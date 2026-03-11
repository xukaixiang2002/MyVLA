"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import draccus
import torch
import torch.nn.functional as F
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


# =====================================================================
#  ExpReS-VLA Components (Syed et al., "ExpReS-VLA")
#  - Compressed Experience Replay via vision embedding extraction
#  - Similarity-based Retrieval Augmented Generation (RAG)
#  - Thresholded Hybrid Contrastive Loss (THCL)
# =====================================================================


class ExperienceBuffer:
    """Circular buffer with FIFO replacement and temporal weighting (Sec IV.A-B).

    Stores L2-normalised vision embeddings (~4 KB each) instead of raw images
    (~150 KB), yielding ~97 % storage compression.
    """

    def __init__(self, capacity: int = 50, decay_rate: float = 0.1):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.embeddings: List[torch.Tensor] = []
        self.timestamps: List[int] = []
        self.current_time: int = 0

    def add(self, embedding: torch.Tensor) -> None:
        if len(self.embeddings) >= self.capacity:
            self.embeddings.pop(0)
            self.timestamps.pop(0)
        self.embeddings.append(embedding.detach().cpu().clone())
        self.timestamps.append(self.current_time)

    def get_temporal_weights(self) -> torch.Tensor:
        """w_i = exp(-λ · Δt_i)  (Eq. 8)"""
        return torch.tensor(
            [math.exp(-self.decay_rate * (self.current_time - ts)) for ts in self.timestamps],
            dtype=torch.float32,
        )

    def increment_time(self) -> None:
        self.current_time += 1

    def __len__(self) -> int:
        return len(self.embeddings)


def retrieve_top_k(
    query: torch.Tensor,
    buffer: ExperienceBuffer,
    k: int,
) -> List[torch.Tensor]:
    """Top-k retrieval via dot-product similarity with temporal weighting (Sec IV.C).

    k is adaptively clamped to min(k, |buffer|, |buffer|//10).
    """
    if len(buffer) == 0:
        return []
    k = min(k, len(buffer), max(1, len(buffer) // 10))
    stored = torch.stack(buffer.embeddings)                 # [N, embed_dim]
    sims = torch.matmul(stored, query.detach().cpu())       # [N]
    scores = sims * buffer.get_temporal_weights()           # [N]
    indices = torch.topk(scores, k).indices
    return [buffer.embeddings[i.item()] for i in indices]


class THCLLoss(torch.nn.Module):
    """Thresholded Hybrid Contrastive Loss (Sec IV.D).

    L_total = L_BC + λ · L_THCL

    Adaptively selects triplet loss when L_triplet ≤ β (simple failures)
    or InfoNCE loss when L_triplet > β (complex failure patterns).
    """

    def __init__(
        self,
        anchor_dim: int,
        embed_dim: int,
        proj_dim: int = 512,
        margin: float = 0.5,
        temperature: float = 0.1,
        threshold: float = 1.0,
        weight: float = 0.3,
    ):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.threshold = threshold
        self.weight = weight
        # Projection heads  h = g_φ(o, c) ∈ R^512
        self.anchor_proj = torch.nn.Sequential(
            torch.nn.Linear(anchor_dim, proj_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_dim, proj_dim),
        )
        self.embed_proj = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, proj_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_dim, proj_dim),
        )

    def forward(
        self,
        anchor: torch.Tensor,       # [anchor_dim]  — requires grad
        pos_embeds: torch.Tensor,    # [num_pos, embed_dim]
        neg_embeds: torch.Tensor,    # [num_neg, embed_dim]
    ) -> torch.Tensor:
        h = self.anchor_proj(anchor)
        h_pos = self.embed_proj(pos_embeds).mean(dim=0)
        h_negs = self.embed_proj(neg_embeds)

        # Triplet: max(0, ||h - h⁺||₂ - ||h - h⁻||₂ + α)  (Eq. 17)
        d_pos = torch.norm(h - h_pos, p=2)
        d_neg = torch.norm(h - h_negs[0], p=2)
        triplet = torch.clamp(d_pos - d_neg + self.margin, min=0.0)

        if triplet.item() <= self.threshold:
            return self.weight * triplet

        # InfoNCE (Eq. 18)
        pos_sim = torch.dot(h, h_pos) / self.temperature
        neg_sims = torch.matmul(h_negs, h) / self.temperature
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
        target = torch.zeros(1, dtype=torch.long, device=anchor.device)
        return self.weight * F.cross_entropy(logits.unsqueeze(0), target)


def _unwrap_model(model):
    """Unwrap DDP / PEFT wrappers to access the base VLA."""
    base = model.module if hasattr(model, "module") else model
    return base.base_model if hasattr(base, "peft_config") else base


def extract_vision_embedding(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Extract L2-normalised vision embeddings from the frozen backbone (Sec IV.A).

    e = f(o) — pooled patch features, normalised to unit norm for efficient
    dot-product similarity computation.  Returns shape [batch, embed_dim].
    """
    base = _unwrap_model(model)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            patch_features = base.vision_backbone(pixel_values)
        if isinstance(patch_features, tuple):
            patch_features = patch_features[0]
        pooled = patch_features.mean(dim=1)          # [batch, embed_dim]
        return F.normalize(pooled, p=2, dim=-1)


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "/home/xkx/project/models/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("/home/xkx/project/datasets/")        # Path to Open-X dataset directory
    dataset_name: str = "libero_object_no_noops"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 4                                            # Fine-tuning batch size
    max_steps: int = 20000                                       # Max number of fine-tuning steps
    save_steps: int = 20000                                         # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 4                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = True                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # ExpReS-VLA Parameters (Syed et al.)
    use_expres: bool = False                                         # Enable ExpReS-VLA method
    buffer_capacity: int = 50                                       # Per-buffer capacity (N_s = N_f = 50)
    temporal_decay: float = 0.1                                     # Temporal decay rate λ
    retrieval_k_success: int = 3                                    # Success experiences retrieved per batch
    retrieval_k_failure: int = 2                                    # Failure experiences retrieved per batch
    contrastive_weight: float = 0.3                                 # THCL weight λ in L_total
    triplet_margin: float = 0.5                                     # Margin α for triplet loss
    infonce_temperature: float = 0.1                                # Temperature τ for InfoNCE
    thcl_threshold: float = 1.0                                     # Switching threshold β
    failure_l1_threshold: float = 0.3                               # L1 threshold to classify as failure
    max_grad_norm: float = 1.0                                      # Gradient clipping ||∇||_∞
    use_cosine_schedule: bool = True                                # Cosine annealing LR schedule
    expres_weight_decay: float = 1e-4                               # Weight decay on LoRA params

    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    # Only wrap with DDP if we're in a true multi-process/multi-GPU environment
    if distributed_state.num_processes > 1:
        vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    else:
        # Move model to the appropriate device if not using DDP
        vla = vla.to(device_id)

    # Create Optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.expres_weight_decay if cfg.use_expres else 0.0,
    )

    # ExpReS-VLA: initialise experience buffers, THCL (lazy), and LR scheduler
    thcl_module = None  # lazy-initialised on first batch when embed_dim is known
    scheduler = None
    success_buffer = None
    failure_buffer = None
    if cfg.use_expres:
        success_buffer = ExperienceBuffer(cfg.buffer_capacity, cfg.temporal_decay)
        failure_buffer = ExperienceBuffer(cfg.buffer_capacity, cfg.temporal_decay)
        if cfg.use_cosine_schedule:
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_steps)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    
    # Get the underlying model config regardless of whether it's wrapped with DDP or PEFT
    base_model = vla.module if hasattr(vla, 'module') else vla
    actual_model = base_model.base_model if hasattr(base_model, 'peft_config') else base_model
    
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(actual_model.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_thcl_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):

            # ---- Forward pass ----
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                    output_hidden_states=cfg.use_expres,
                )
                bc_loss = output.loss

            # ---- ExpReS-VLA: embedding extraction, RAG retrieval, THCL (Sec IV) ----
            thcl_loss = torch.tensor(0.0, device=device_id)
            if cfg.use_expres:
                # Extract vision embeddings from frozen backbone (Sec IV.A)
                with torch.no_grad():
                    vis_embed = extract_vision_embedding(
                        vla, batch["pixel_values"].to(torch.bfloat16).to(device_id)
                    )

                # Lazy-initialise THCL on first batch (need embed_dim from backbone)
                if thcl_module is None:
                    _edim = vis_embed.shape[-1]
                    _hdim = _unwrap_model(vla).config.text_config.hidden_size
                    thcl_module = THCLLoss(
                        anchor_dim=_hdim,
                        embed_dim=_edim,
                        margin=cfg.triplet_margin,
                        temperature=cfg.infonce_temperature,
                        threshold=cfg.thcl_threshold,
                        weight=cfg.contrastive_weight,
                    ).to(device_id)
                    optimizer.add_param_group({"params": thcl_module.parameters(), "lr": cfg.learning_rate})

                # RAG retrieval + THCL computation (Sec IV.C-D)
                avg_embed = vis_embed.mean(dim=0)  # query embedding
                if len(success_buffer) > 0 and len(failure_buffer) > 0:
                    pos_list = retrieve_top_k(avg_embed, success_buffer, cfg.retrieval_k_success)
                    neg_list = retrieve_top_k(avg_embed, failure_buffer, cfg.retrieval_k_failure)
                    if pos_list and neg_list:
                        pos_t = torch.stack(pos_list).to(device_id).float()
                        neg_t = torch.stack(neg_list).to(device_id).float()
                        # Anchor: penultimate LLM hidden state (Eq. 17)
                        anchor = output.hidden_states[-2].mean(dim=1).mean(dim=0).float()
                        thcl_loss = thcl_module(anchor, pos_t, neg_t)

            # Combined loss: L_total = L_BC + L_THCL  (Eq. 15)
            loss = bc_loss + thcl_loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            # Get the underlying model regardless of whether it's wrapped with DDP or PEFT
            base_model = vla.module if hasattr(vla, 'module') else vla
            actual_model = base_model.base_model if hasattr(base_model, 'peft_config') else base_model
            action_logits = output.logits[:, actual_model.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # ExpReS-VLA: store experiences in dual buffers (Sec IV.B)
            if cfg.use_expres:
                for i in range(vis_embed.size(0)):
                    if action_l1_loss.item() > cfg.failure_l1_threshold:
                        failure_buffer.add(vis_embed[i])
                    else:
                        success_buffer.add(vis_embed[i])

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())
            recent_thcl_losses.append(thcl_loss.item() if isinstance(thcl_loss, torch.Tensor) else 0.0)

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                log_dict = {
                    "train_loss": smoothened_loss,
                    "action_accuracy": smoothened_action_accuracy,
                    "l1_loss": smoothened_l1_loss,
                }
                if cfg.use_expres:
                    log_dict["thcl_loss"] = sum(recent_thcl_losses) / max(len(recent_thcl_losses), 1)
                    log_dict["success_buffer_size"] = len(success_buffer)
                    log_dict["failure_buffer_size"] = len(failure_buffer)
                wandb.log(log_dict, step=gradient_step_idx)

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                # Gradient clipping (Sec IV.E: ||∇||_∞ ≤ 1.0)
                if cfg.use_expres and cfg.max_grad_norm > 0:
                    params_to_clip = [p for p in vla.parameters() if p.requires_grad]
                    if thcl_module is not None:
                        params_to_clip += [p for p in thcl_module.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                # Increment buffer timestamps each gradient step
                if cfg.use_expres:
                    success_buffer.increment_time()
                    failure_buffer.increment_time()
                progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    
                    # Use _unwrap_model to handle both DDP and PEFT cases when saving
                    model_to_save = _unwrap_model(vla) if cfg.use_lora else vla
                    if hasattr(vla, 'module'):  # Only use .module if it exists (in DDP case)
                        vla.module.save_pretrained(save_dir)
                    else:  # In non-DDP case, save the unwrapped model
                        model_to_save.save_pretrained(save_dir)

                # Wait for processor and adapter weights to be saved by main process
                if distributed_state.num_processes > 1:
                    dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            # Overwrite latest checkpoint
                            merged_vla.save_pretrained(run_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                if distributed_state.num_processes > 1:
                    dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()