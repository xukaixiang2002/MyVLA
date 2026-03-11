# ExpReS-VLA: Specializing Vision-Language-Action Models Through Experience Replay and Retrieval
Shahram Najam Syed¹*, Yatharth Ahuja¹*, Arthur Jakobsson¹, Jeff Ichnowski¹
¹Robotics Institute, Carnegie Mellon University, Pittsburgh, USA
*Equal contribution

## Abstract
Vision-Language-Action (VLA) models like OpenVLA demonstrate impressive zero-shot generalization across robotic manipulation tasks but struggle to adapt to specific deployment environments where consistent high performance on a limited set of tasks is more valuable than broad generalization. We present EXPierence replayed, REtrieval augmented, Specialized VLA (ExpReS-VLA), a method that enables rapid on-device adaptation of pre-trained VLAs to target domains while preventing catastrophic forgetting through compressed experience replay and retrieval-augmented generation. Our approach maintains a memory-efficient buffer by storing extracted embeddings from OpenVLA’s frozen vision backbone, reducing storage requirements by 97% compared to raw image-action pairs. During deployment, ExpReS-VLA retrieves the k most similar past experiences using cosine similarity to augment training batches, while a prioritized experience replay buffer preserves recently successful trajectories. To leverage failed attempts, we introduce Thresholded Hybrid Contrastive Loss (THCL), enabling the model to learn from both successful and unsuccessful demonstrations collected during deployment.

Experiments on the LIBERO simulation benchmark show that ExpReS-VLA improves success rates from 82.6% to 93.1% on spatial reasoning tasks and from 61% to 72.3% on longhorizon tasks compared to base OpenVLA. Physical robot experiments across five manipulation tasks demonstrate that our approach achieves 98% success on both in-distribution and out-of-distribution tasks (with unseen backgrounds and objects), improving from 84.7% and 32% respectively for naive fine-tuning. ExpReS-VLA accomplishes this adaptation in 31 seconds using only 12 demonstrations on a single RTX 5090, making it practical for real-world deployment where robots must quickly specialize to their specific operating environment.

## I. INTRODUCTION
Every deployed robot faces a fundamental paradox: trained on diverse Internet-scale data and robot demonstrations, it must excel at just a handful of tasks in one specific environment. A deployed robot does not require the ability to manipulate all object categories from its 970,000-trajectory training dataset—it requires consistent, high-performance manipulation of the specific objects in its deployment environment.

OpenVLA [1], a 7B-parameter open-source VLA, exemplifies this tension: achieving 70% success across 29 manipulation tasks, yet struggling to reach the 95%+ reliability users demand for their specific objects and lighting conditions. This specialization challenge reveals the gap between how we train vision-language-action models—for broad generalization—and how we deploy them—for consistent specialization in constrained environments.

This specialization challenge manifests as domain shift—subtle differences in lighting, object textures, or spatial layouts that degrade zero-shot performance from acceptable to unusable. While fine-tuning can adapt to specific environments, it can suffer from catastrophic forgetting [2], where learning new tasks erases previously acquired skills. Existing solutions either require extensive computational resources (full model fine-tuning on GPUs clusters [1]) or fail to leverage failed demonstrations that naturally occur during deployment. Moreover, current approaches treat adaptation as an offline process, incompatible with robots that must improve through daily interaction.

We present EXPierence replayed, REtrieval augmented, Specialized VLA (ExpReS-VLA), a method that makes catastrophic forgetting of previously run tasks structurally impossible through frozen encoders and persistent memory buffers, enabling rapid on-device adaptation of pretrained VLAs. Our key insight is that successful domain adaptation can benefit three complementary mechanisms: (1) compressed memory to efficiently store experiences, (2) retrieval-augmented generation to leverage relevant past experiences, and (3) contrastive learning to explicitly avoid past failures. By combining these mechanisms, ExpReS-VLA transforms OpenVLA from a generalist that works adequately everywhere into a specialist that excels in its deployment environment.

ExpReS-VLA addresses three critical challenges in practical VLA deployment. First, we achieve 97% storage reduction for experience replay by storing vision encoder embeddings instead of raw images, enabling efficient memory management for continual learning. Second, we accelerate convergence through retrieval-augmented training that injects contextually similar past experiences into each batch. Third, we introduce Thresholded Hybrid Contrastive Loss (THCL), which adaptively switches between triplet [3] and InfoNCE [4] objectives based on failure complexity, transforming unsuccessful attempts into learning signals.

We evaluate ExpReS-VLA in simulation and real-world experiments. On LIBERO simulation benchmarks, ExpReS-VLA achieves 92.4% success on spatial tasks and 72% on long-horizon tasks, which are improvements of 10% to 11% over base OpenVLA. Physical robot experiments show ExpReS-VLA improves in-distribution success from 84.7% to 98% and out-of-distribution success from 32% to 98%, with the larger OOD gain demonstrating robust adaptation to unseen variations. ExpReS-VLA completes adaptation in 31 seconds using 12 demonstrations on a single RTX 5090 GPU.

This work makes the following contributions:
- RAG-augmented robot learning: First integration of retrieval mechanisms into VLA fine-tuning, improving adaptation speed.
- Compressed experience replay: A 97% memory reduction technique using frozen vision encoders that maintains semantic fidelity while enabling practical deployment.
- THCL for failure exploitation: A novel piecewise loss that prevents repeated mistakes by dynamically selecting appropriate contrastive objectives.
- Rigorous empirical evaluation: Systematic ablations across 40 simulation tasks (5 seeds) and 5 physical manipulation tasks (150 total trials) establishing clear component contributions.

## II. RELATED WORK
### a) Vision-Language-Action Models (VLAs)
Generalist policies like OpenVLA [1] and RT-2 [5], among others [6], [7], [8], [9], [10], [11], [12], demonstrate broad capabilities but remain brittle in real-world deployments. They often fail with domain shifts across embodiment changes, and lack the deployed performance for specialized tasks [13], [14], [15]. These works also primarily focused on model development and this creates a trade-off between generalization and specialization, which our framework addresses. Our focus is to help these models adapt and fine-tune to specific cases where performance requirements demand robustness and accuracy, by effectively balancing prior data and new experiences.

### b) Fine-tuning and Catastrophic Forgetting
Traditionally, domain adaptation relies on fine-tuning approaches that update the entire network on new data to generalize from a source domain to a target domain [16], [5]. However, this conventional approach is impractical for real-time, on-device adaptation in dynamic environments, as it requires extensive training paradigm, significant GPU memory, and a large, stationary dataset. This process is further complicated by catastrophic forgetting [17], [18], a fundamental problem in continual learning where a model’s plasticity (acquiring new knowledge) comes at the cost of its stability (retaining old knowledge).

Recent state-of-the-art methods have sought to overcome these limitations through more efficient techniques. Previous works have tried to overcome this using Elastic Weight Consolidation through regularizing weight changes [19], which work well in static scenarios. Advancing with task-specific columns [20] helps in preserving previous weights but becomes computationally expensive and difficult. Another way to “pack” multiple tasks in a network is to iteratively prune it and free up parameters, but this approach is computationally expensive and provides difficulty in judging the importance of parameters freedup [21]. Long et al. [22] combined parameterized retrieval-augmented generation (P-RAG) [23] with Dirichlet Process Mixture Model (DPMM) [24] to retain prior knowledge effectively. On the other hand, parameter-efficient fine-tuning (PEFT) [25], such as LoRA [26], fine-tunes only a small fraction of the model’s parameters to reduce computational overhead while maintaining performance. Meta-learning, or “learning to learn,” trains models to learn new tasks more efficiently with only a few examples [27].

ExpReS-VLA builds on this foundation by proposing a holistic framework that integrates a compact memory, retrieval-augmented mechanisms, and a principled mixing schedule to enable adaptive fine-tuning on resource-constrained hardware while explicitly combating catastrophic forgetting by provisioning replay of relevant prior data, along with newer encounters, to reinforce training of parameters towards such scenarios to advance the learning.

### c) Experience Replay
Experience replay, a technique for storing and reusing past experiences, is foundational in deep reinforcement learning for improving data efficiency and training stability [28]. In continual learning, it is primarily a method for mitigating catastrophic forgetting, inspired by biological memory consolidation [29], [30] which comes with a trade-off of storing vast raw sensory data from a robot and a significant memory bottleneck, necessitating efficient data and memory management, as we explore in our system.

### d) The RAG Paradigm in Robotics
Retrieval-Augmented Generation (RAG), which enriches model outputs with external data at inference time, is a popular, cost-effective alternative to retraining in NLP [31], [32], [33], and they have been explored in knowledge retrieval systems effectively [34]. While they are challenging to adapt for robotics’ low-latency, multimodal needs, RAG is emerging as a powerful tool for guiding agents by retrieving past interactions as in-context examples [22]. ExpReS-VLA uses RAG as a “warm-start” for on-device fine-tuning. It queries a compact memory buffer for similar past experiences and injects them into the current learning batch, providing a highly relevant initial gradient that significantly reduces adaptation time.

## III. PROBLEM STATEMENT
Given a pre-trained VLA, and a robot in a specific deployment, the goal is to adapt the VLA to improve its task performance as measured by success rates. In this scenario, robots have limited computing resources and memory constraints (either on-board or cost-restricted cloud). Unlike traditional fine-tuning that assumes batch access to stationary data, our setting reflects real-world deployment where robots must adapt through sequential interactions while maintaining previously acquired capabilities.

### A. Mathematical Formulation
Let $\pi_{\theta_{0}}: O ×C \to A$ be a pre-trained VLA model with parameters $\theta_{0} \in \mathbb{R}^{d}$ trained on source domain $D_{0}$. Upon deployment in target domain $D_{new }$ , the robot observes a stream of interactions:
- Observation space $O=\mathbb{R}^{H ×W ×3}$ : RGB image from a fixed third-person camera
- Command space $C=\mathbb{N}^{L_{max }}$ : Tokenized natural language instructions with maximum length $L_{max }$
- Action space $A=\mathbb{R}^{d_{a}}$ : Continuous end-effector control (7-DOF: 3D position, 3D orientation, gripper)

At each timestep t , the robot receives observation $o_{t}$ and command $c_{t}$ , then executes action $a_{t}=\pi_{\theta_{t}}(o_{t}, c_{t})$ . The environment provides binary success signal $s_{t} \in\{0,1\}$ and, for successful trajectories, expert demonstrations $a_{t}^{*}$ .

### B. Learning Objectives
Adaptation involves three competing objectives:
1) **Adaptation Performance**: Minimize cumulative imitation loss on target domain:
$$\mathcal{L}_{adapt }(T)=\sum_{t=1}^{T} 1_{s_{t}=1} \cdot \mathcal{L}_{bc}\left(\pi_{\theta_{t}}\left(o_{t}, c_{t}\right), a_{t}^{*}\right)$$
where 1 is the indicator function and $L_{bc}(\cdot, \cdot)$ is the behavioral cloning loss.

2) **Catastrophic Forgetting Prevention**: Maintain performance on prior tasks stored in replay buffer B
$$\mathcal{F}(T)=\frac{1}{|\mathcal{B}|} \sum_{\left(\tilde{o}, \tilde{c}, \tilde{a}^{*}\right) \in \mathcal{B}} \mathcal{L}_{bc}\left(\pi_{\theta_{T}}(\tilde{o}, \tilde{c}), \tilde{a}^{*}\right)$$

3) **Memory Efficiency**: Operate within strict memory budget M (in MB):
$$Memory(\mathcal{B})=\sum_{i \in \mathcal{B}} size\left(e_{i}\right)+size\left(a_{i}^{*}\right) \leq M$$
where $e_{i}=f(o_{i})$ is the compressed embedding from frozen vision encoder $f: \mathbb{R}^{H ×W ×3} \to \mathbb{R}^{d_{e}}$

The complete optimization problem is thus:
$$\begin{array}{rl} min _{\left\{\theta_{t}\right\}_{t=1}^{T}} & \mathcal{L}_{adapt }(T) \\ s.t. & \mathcal{F}(T) \leq \varepsilon \\ & Memory(\mathcal{B}) \leq M . \end{array}$$

### C. Assumptions
**Inherited from OpenVLA**:
- Open-loop control: We predict entire action sequences from initial observations without real-time visual feedback.
- Static environment: The deployment environment remains consistent during operation (fixed camera position, lighting conditions, workspace layout).

**ExpReS-VLA-specific assumptions**:
- Binary success signals: We assume access to task completion signals for automatic success/failure labeling in simulation; physical robots require manual labeling.
- Single robot embodiment: We focus on adaptation for a single robot type without cross-embodiment transfer.
- Limited demonstrations: Target domain provides sparse expert demonstrations (10–30 trajectories per task), necessitating data-efficient adaptation.
- Consumer hardware: All computation runs on a single consumer-grade GPU with $≤32 ~GB$ memory, constraining model size and batch processing.

These assumptions reflect practical deployment scenarios where robots must specialize to their environments with minimal human supervision and computational resources. Our solution, detailed in Section IV, addresses these challenges through compressed experience replay, retrieval-augmented generation, and contrastive learning from failures.

## IV. METHOD
Starting with a pre-trained VLA model, ExpReS-VLA continuously collects experiences during deployment, stores them in compressed form, and retrieves relevant past experiences to guide future adaptation. This creates a virtuous cycle: the robot attempts tasks, remembers both successes and failures, and learns from similar past situations when encountering new challenges. To enable this cycle on resource-constrained hardware, ExpReS-VLA combines three mechanisms that work synergistically: compressed storage via embedding extraction, similarity-based retrieval for relevant experience selection, and adaptive contrastive learning to leverage failures.

### A. Embedding Extraction and Storage
We extract compact representations from observations using OpenVLA’s pre-trained vision encoder to achieve memory-efficient storage without sacrificing task-relevant information. The encoder $f: \mathbb{R}^{224 ×224 ×3} \to \mathbb{R}^{1024}$ combines features from two complementary vision transformers:
$$e=f(o)=\left[e_{SigLIP } ; e_{DINOv 2}\right],$$
where $e_{SigLIP } \in \mathbb{R}^{768}$ captures semantic content and $e_{DINOv2 } \in \mathbb{R}^{256}$ encodes spatial structure, with $[;]$ denoting concatenation.

This representation preserves critical visual information while achieving substantial compression. Each raw image requires $224 ×224 ×3 ×1=150,528$ bytes in uint8 format. The extracted embedding requires $1024 ×4=4,096$ bytes in float32 format, yielding a compression ratio of 36.7:1.

We store experiences as tuples $\tau=(e, c, a, s)$ where:
- $e \in \mathbb{R}^{1024}$ : Visual embedding from frozen encoder
- $c \in \mathbb{N}^{L}$ : Tokenized language command (variable length $L ≤L_{max }$)
- $a \in\mathbb{R}^{7×T}$ : Action sequence for trajectory length T
- $s \in\{0,1\}$ : Binary success indicator

Freezing the encoder ensures that embeddings are consistent across adaptation cycles. Empirically, when fine-tuning with a non-frozen encoder, we found that the cosine similarity of embeddings of images before and after fine-tuning remained stable (0.98 ± 0.01), confirming that space-savings afforded by storing the embeddings results in minimal loss in embedding-based specialization.

We normalize embeddings to unit norm for two critical reasons: (1) enabling efficient similarity computation via dot products instead of costly cosine calculations, and (2) preventing gradient explosion during contrastive learning by bounding the embedding space to the unit hypersphere. This normalization is performed immediately after extraction.

### B. Dual-Buffer Memory Management
We maintain separate circular buffers for successful and failed trajectories to enable targeted retrieval during adaptation. This separation prevents failed experiences from diluting the behavioral cloning signal while preserving them for contrastive learning. By storing successes and failures independently, we can control the ratio of positive to negative examples in each training batch, ensuring sufficient learning signal from both.

#### Buffer Structure
We implement two fixed-capacity buffers:
$$\mathcal {B}_{s}=\left\{ (e_{i},c_{i},a_{i}^{*},1):i\in [1,N_{s}]\right\} \quad(\text{success buffer})$$
$$\mathcal{B}_{f}=\left\{ (e_{j},c_{j},a_{j},0): j\in \left[1, N_{f}\right]\right\} \quad(\text{failure buffer}).$$

In experiments, we set $N_{s}=N_{f}=50$ to fit within our memory budget while maintaining sufficient diversity.

#### Replacement Policy
We employ FIFO (First-In-First-Out) replacement with temporal weighting. When buffer capacity is reached, we replace the oldest entry but maintain a priority weight for each stored experience:
$$w_{i}=exp \left(-\lambda \cdot \Delta t_{i}\right) ,$$
where $\Delta t_{i}$ is the time since storage (in adaptation cycles) and $\lambda=0.1$ controls decay rate. These weights influence retrieval probability without affecting storage decisions.

#### Success Detection
In simulation, we automatically classify trajectory outcomes using environment feedback:
$$s=\left\{\begin{array}{c} 1 \text{ if } d\left(p_{object }, p_{goal }\right)<\epsilon_{pos } \text{ AND} \\ \left|f_{gripper }-f_{expected }\right|<\epsilon_{force } \text{ AND} \\ t<t_{max } \\ 0 \end{array}\right.$$
where $d(\cdot, \cdot)$ measures Euclidean distance. In experiments, we set $\epsilon_{pos}=5 ~cm$ , $\epsilon_{force }=2 ~N$,and $t_{max }=100$ steps.

### C. Similarity-Based Experience Retrieval
We retrieve relevant experiences from both buffers using cosine similarity in the embedding space. This retrieval augments training batches with contextually similar demonstrations, accelerating adaptation to the target domain.

#### Similarity Computation
Given a query embedding $e_{q}$ from the current observation, we compute similarity scores with all stored experiences:
$$sim(e_{q},e_{i})=e_{q}^{T} e_{i} .$$

Since embeddings are pre-normalized to unit norm, cosine similarity reduces to a simple dot product, eliminating the need to compute norms at query time.

#### Top-k Selection
We retrieve the k most similar experiences from each buffer:
$$\mathcal {R}_{s}={top}-k\left\{ (e_{i},c_{i},a_{i}^{*})\in \mathcal {B}_{s}:sim(e_{q},e_{i})\right\}$$
$$\mathcal{R}_{f}= top- k\left\{\left(e_{j}, c_{j}, a_{j}\right) \in \mathcal{B}_{f}: sim\left(e_{q}, e_{j}\right)\right\}$$

We set $k=min (5,|B| / 10)$ based on empirical ablation studies that showed this configuration balances diversity with relevance. Retrieving 5 experiences provided sufficient context without overwhelming the training batch, while the adaptive scaling (10% of buffer size) ensures meaningful retrieval even with partially filled buffers during initial deployment.

#### Weighted Sampling
Retrieved experiences are weighted by both similarity and temporal recency:
$$p_{i}=\frac{sim\left(e_{q}, e_{i}\right) \cdot w_{i}}{\sum_{j \in \mathcal{R}} sim\left(e_{q}, e_{j}\right) \cdot w_{j}},$$
where wi is the temporal weight from Section 4.2. This weighting prioritizes recent, similar experiences while maintaining some diversity through probabilistic sampling.

#### Batch Construction
Each training batch combines current observations with retrieved experiences:
$$\mathcal{D}_{train }=\left\{\left(o_{curr }, c_{curr }, a_{curr }\right)\right\} \cup sample\left(\mathcal{R}_{s}, 3\right) \cup sample\left(\mathcal{R}_{f}, 2\right)$$

The 3:2 ratio of success to failure retrievals balances positive demonstrations with negative examples for contrastive learning. We reconstruct full observations from embeddings using a learned decoder when necessary, though we find that operating directly on embeddings suffices for most adaptation objectives.

### D. Thresholded Hybrid Contrastive Loss (THCL)
We introduce THCL to learn from both successful and failed demonstrations by dynamically selecting between two contrastive objectives based on the difficulty of distinguishing failures from successes.

#### Loss Formulation
THCL combines behavioral cloning with adaptive contrastive learning:
$$\mathcal{L}_{total }=\mathcal{L}_{BC}+\lambda \mathcal{L}_{THCL}$$
where $L_{BC}=-log p(a^{*} | o, c)$ is the standard imitation loss and $\lambda=0.3$ weights the contrastive term.

#### Adaptive Switching Mechanism
The contrastive component switches between two formulations:
$$\mathcal{L}_{THCL }= \begin{cases}\mathcal{L}_{triplet } & if \mathcal{L}_{triplet } \leq \beta \\ \mathcal{L}_{InfoNCE } & otherwise. \end{cases}$$

This piecewise selection adapts to the complexity of negative examples. Simple failures trigger triplet loss (efficient), while complex failure patterns invoke InfoNCE (more expressive).

#### Triplet Loss
For single negative examples, we enforce margin constraints:
$$\mathcal{L}_{triplet }=max \left(0,\left\| h-h^{+}\right\| _{2}-\left\| h-h^{-}\right\| _{2}+\alpha\right),$$
where $h=g_{\phi}(o, c) \in \mathbb{R}^{512}$ is the penultimate layer representation, $h^{+}$ corresponds to successful actions, $h^{-}$ to failures, and margin $\alpha=0.5$ . We use L2 distance rather than cosine similarity here as the representation space $g_{\phi}$ is not normalized, allowing the model to learn appropriate scales.

#### InfoNCE Loss
For multiple negatives, we maximize the likelihood of positive examples:
$$\mathcal{L}_{InfoNCE}=-log \frac{exp \left(h^{T} h^{+} / \tau\right)}{exp \left(h^{T} h^{+} / \tau\right)+\sum_{i=1}^{K} exp \left(h^{T} h_{i}^{-} / \tau\right)}$$
with temperature $\tau=0.1$ and $K=|R_{f}|$ negative samples from the failure retrieval set. Lower temperature increases discrimination between positives and hard negatives.

#### Threshold Calibration
We set switching threshold $\beta=1.0$ based on empirical analysis of 1000 training batches. Distribution of losses shows:
- 78% of batches satisfy $L_{triplet } ≤1.0$ (use triplet)
- 22% exceed threshold (use InfoNCE)

This ratio indicates that most failure modes are distinguishable with simple constraints, while genuinely ambiguous cases benefit from multi-negative comparison.

### E. Online Learning Pipeline
We trigger adaptation when performance degrades below acceptable thresholds and execute a structured training protocol that balances rapid improvement with computational constraints.

#### Adaptation Triggers
We adopt OpenVLA’s LoRA configuration [1]: rank 32, BFloat16 precision, and adaptation of query/value projections only, yielding 98.3M trainable parameters (1.4% of 7B). We initiate fine-tuning when:
$$\frac{1}{N_{w}} \sum_{i=t-N_{w}}^{t} s_{i}<\theta_{adapt },$$
where $N_{w}=10$ is the window size, $s_{i}$ is the success indicator for attempt i,and $\theta_{adapt }=0.8$ . This criterion ensures adaptation only occurs after consistent performance degradation, avoiding premature updates from isolated failures.

#### Training Procedure
We execute the following optimization:
**Algorithm 1 Online Adaptation**
1: Initialize LoRA parameters
2: Extract embeddings for collected trajectories
3: Update buffers Bs, Bf with new experiences
4: for epoch = 1 to 2 do
5:    for each trajectory τ in collected data do
6:        Retrieve similar experiences via Eq. 11–12
7:        Construct augmented batch Dtrain
8:        Compute Ltotal using THCL (Eq. 15)
9:        Update: {B, A} ←{B, A} −η∇Ltotal
10:    end for
11: end for
12: Deploy updated model πθt+1

#### Hyperparameter Configuration
- Learning rate: $\eta=2 ×10^{-5}$ with cosine decay
- Batch size: 1 with gradient accumulation over 8 steps
- Gradient clipping: $\|\nabla\|_{\infty} ≤1.0$
- Weight decay: $1 ×10^{-4}$ on LoRA parameters only
- Mixed precision: BFloat16 for forward pass, Float32 for gradients

## V. EXPERIMENTS
We evaluate ExpReS-VLA across simulation and physical robot experiments to demonstrate: (1) consistent performance improvements over baselines, (2) effective utilization of failed demonstrations through contrastive learning, and (3) practical deployment feasibility on consumer hardware. All experiments use OpenVLA as the base model with identical hyperparameters detailed in Section IV.

### A. Experimental Setup
All experiments run on a single NVIDIA RTX 5090 (32GB) GPU using mixed precision (BFloat16) with PyTorch 2.0, demonstrating the feasibility of on-device adaptation without distributed computing infrastructure. We evaluate each method across two complementary settings: simulation experiments on the LIBERO [35] benchmark comprising 4 task suites with 10 tasks each, evaluated over 50 rollouts per task across 5 random seeds for statistical reliability, and physical robot experiments using a 7-DOF Franka Emika Panda arm performing 5 manipulation tasks with 30 trials for in-distribution conditions and 10 trials for out-of-distribution variants.

We compare ExpReS-VLA against four baselines to establish performance bounds. Diffusion Policy and Octo results are taken directly from the OpenVLA paper [1] to ensure fair comparison, representing state-of-the-art imitation learning from scratch and fine-tunable generalist policies respectively. We additionally evaluate OpenVLA trained from random initialization to measure the benefit of pretraining, and OpenVLA with naive fine-tuning (without our memory mechanisms) to isolate the contribution of our approach.

To understand component contributions, we conduct systematic ablations by removing individual elements: ExpReS-VLA(-C) excludes the contrastive loss to measure the impact of learning from failures, ExpReS-VLA(-R) removes RAG retrieval to assess the value of similarity-based experience selection, and ExpReS-VLA(-E) eliminates experience replay to quantify the importance of memory retention. These ablations reveal which components are essential versus complementary for achieving robust adaptation.

### B. Simulation Results
Table I presents results on the LIBERO benchmark, where ExpReS-VLA achieves the highest average success rate of 88.7%, outperforming the best baseline (OpenVLA) by 10.8 percentage points. The ablation studies reveal clear component contributions: removing both contrastive learning and RAG retrieval (ExpReS-VLA(-CR)) yields minimal improvement over base OpenVLA at 80.7%, adding experience replay and contrastive learning without RAG (ExpReS-VLA(-EC)) reaches 85.3%, while removing only contrastive learning (ExpReS-VLA(-C)) achieves 87.3%. This progression demonstrates that RAG retrieval provides the largest individual gain of 6.6 percentage points, followed by experience replay at 4.6 points, with contrastive learning adding the final 1.4 points to reach full performance.

**TABLE I: LIBERO benchmark results showing success rates (%) with standard errors across 5 seeds. ExpReS-VLA achieves highest performance across all task categories, with particular improvements in spatial reasoning (10% absolute gain) and long-horizon tasks (11% gain).**

| Method | LIBERO- Spatial | LIBERO- Object | LIBERO- Goal | LIBERO- Long | Average |
| --- | --- | --- | --- | --- | --- |
| Diffusion Policy* | 78.3±1.1 | 92.5±0.7 | 68.3±1.2 | 50.5±1.3 | 72.4 |
| Octo fine-tuned* | 78.9±1.0 | 85.7±0.9 | 84.6±0.9 | 51.1±1.3 | 75.1 |
| OpenVLA (base) | 82.6±2.1 | 88.9±0.7 | 79.0±1.4 | 61.0+0.5 | 77.9 |
| ExpReS-VLA(-CR) | 82.6±1.0 | 85.8±0.1 | 88.4+1.2 | 66.0±0.3 | 80.7 |
| ExpReS-VLA(-EC) | 90.2±0.3 | 91.0±0.5 | 88.6+3.4 | 71.4+5.7 | 85.3 |
| ExpReS-VLA(-C) | 92.4±2.9 | 91.8±0.1 | 93.0±0.4 | 72.0+3.5 | 87.3 |
| ExpReS-VLA (full) | 93.1+2.9 | 93.9±0.1 | 95.4+0.4 | 72.3+3.5 | 88.7 |

Performance improvements are most pronounced on complex tasks requiring spatial reasoning and long-horizon planning. LIBERO-Goal shows the largest absolute gain of 16.4 percentage points over base OpenVLA, while LIBERO-Long improves by 11.3 points, suggesting LWRA’s effectiveness on multi-step problems that traditionally challenge imitation learning methods. The synergy between components is evident in the full model outperforming any ablation variant, confirming that memory retention, similarity-based retrieval, and contrastive learning from failures work complementarily rather than redundantly.

### C. Physical Robot Results
Table II presents physical robot experiments that validate our approach in real-world conditions. ExpReS-VLA achieves 98% success on both in-distribution and out-of-distribution tasks, demonstrating remarkable consistency across varying conditions. The most striking result is the catastrophic failure of naive fine-tuning on OOD scenarios, dropping from 84.7% to 32% success rate when encountering unseen backgrounds, objects, or variations. In contrast, ExpReS-VLA maintains 98% performance on these same OOD conditions, confirming that our memory and retrieval mechanisms prevent the overfitting that plagues standard fine-tuning approaches.

**TABLE II: Physical robot experiments showing success counts out of 30 trials for in-distribution tasks and 10 trials for out-of-distribution (OOD) variants. ExpReS-VLA maintains near-perfect performance on both conditions.**

| Task | Trials | OpenVLA (scratch) | OpenVLA (naive FT) | ExpRes- VLA (-C) | ExpRes- VLA |
| --- | --- | --- | --- | --- | --- |
| **In-Distribution Tasks** | | | | | |
| Place white mug in bowl | 30 | 18/30 | 21/30 | 27/30 | 29/30 |
| Stack all bowls | 30 | 25/30 | 27/30 | 30/30 | 30/30 |
| Push bowl near glass | 30 | 24/30 | 24/30 | 30/30 | 29/30 |
| Knock pringles can | 30 | 30/30 | 30/30 | 30/30 | 30/30 |
| Move 7UP next to Pepsi | 30 | 17/30 | 25/30 | 26/30 | 29/30 |
| Total (In-Dist) | 150 | 114/150 | 127/150 | 143/150 | 147/150 |
| Success Rate | | 76.0% | 84.7% | 95.3% | 98.0% |
| **Out-of-Distribution Tasks** | | | | | |
| Place mug (new bg) | 10 | 6/10 | 2/10 | 8/10 | 9/10 |
| Stack bowls (unseen) | 10 | 4/10 | 1/10 | 10/10 | 10/10 |
| Push bowl (new bg) | 10 | 3/10 | 5/10 | 10/10 | 10/10 |
| Knock can (diff size) | 10 | 10/10 | 7/10 | 10/10 | 10/10 |
| Move Diet 7UP | 10 | 1/10 | 1/10 | 10/10 | 10/10 |
| Total (OOD) | 50 | 24/50 | 16/50 | 48/50 | 49/50 |
| Success Rate | | 48.0% | 32.0% | 96.0% | 98.0% |

The contribution of contrastive learning becomes particularly evident in OOD scenarios, where adding THCL improves performance from 96% to 98%. While this 2 percentage point gain may appear modest, it represents halving the failure rate from 4% to 2%, crucial for deployment where even rare failures can be costly. All methods were trained on identical data—just 12 demonstrations collected in 31 seconds on our RTX 5090—highlighting that ExpReS-VLA’s advantages stem from better utilization of limited data rather than requiring additional supervision. The consistent performance across diverse tasks, from precise placement operations to dynamic pushing movements, indicates that our approach provides general-purpose robustness rather than task-specific improvements.

#### Qualitative Analysis
Failed cases for baseline methods typically involve: (1) repeated unsuccessful grasping attempts at previously failed positions, (2) confusion between similar objects, and (3) inability to recover from initial mistakes. ExpReS-VLA avoids these failure modes through explicit contrastive learning from past failures. Notably, in the “Push bowl near glass” task, ExpReS-VLA(-C) achieved perfect 30/30 success while the full ExpReS-VLA model recorded 29/30. Video analysis revealed that the single failure was caused by a transient shadow artifact from overhead lighting during that specific trial, creating a visual distortion that the contrastive loss may have made the model more sensitive to. This isolated incident suggests that while contrastive learning generally improves robustness, it can occasionally increase sensitivity to spurious visual features, a trade-off worth investigating in future work.

## VI. CONCLUSION
We presented ExpReS-VLA, a framework that reconciles the fundamental tension between broad VLA generalization and specialized deployment performance. Our key observation is that catastrophic forgetting is not an inherent limitation of neural adaptation—it’s an artifact of poor memory management. By maintaining frozen vision encoders and compressed experience buffers, ExpReS-VLA makes forgetting architecturally impossible while enabling rapid specialization. The success of retrieval-augmented training demonstrates that robots don’t need massive datasets for adaptation; they need smart reuse of relevant past experiences. Most importantly, our results show that learning from failures through contrastive objectives transforms inevitable mistakes from wasted attempts into valuable training signals.

### Limitations
ExpReS-VLA requires manual success labeling for physical robots, limiting fully autonomous deployment. Our experiments focus on a single embodiment (7-DOF arm) in static environments; cross-embodiment transfer remains unexplored. The fixed-capacity buffers may not scale to long-term deployment spanning months, and THCL occasionally increases sensitivity to visual artifacts, suggesting task-specific tuning may be beneficial. Future work should address automatic success detection, cross-embodiment transfer, and dynamic buffer management for lifelong learning scenarios.

## REFERENCES
[1] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, et al., “Openvla: An open-source vision-language-action model,” arXiv preprint arXiv:2406.09246, 2024.
[2] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell, “Overcoming catastrophic forgetting in neural networks,” in Proceedings of the National Academy of Sciences (PNAS), vol. 114, no. 13, 2017, pp. 3521–3526.
[3] F. Schroff, D. Kalenichenko, and J. Philbin, “Facenet: A unified embedding for face recognition and clustering,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 815–823.
[4] A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.
[5] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, et al., “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” arXiv preprint arXiv:2307.15818, 2023.
[6] X. Li, M. Liu, H. Zhang, C. Yu, J. Xu, H. Wu, C. Cheang, et al., “Vision-language foundation models as effective robot imitators,” arXiv preprint arXiv:2311.01378, 2023.
[7] A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al., “Open x-embodiment: Robotic learning datasets and rt-x models: Open xembodiment collaboration 0,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 6892–6903.
[8] J. Huang, S. Yong, X. Ma, X. Linghu, P. Li, Y. Wang, Q. Li, S.-C. Zhu, B. Jia, and S. Huang, “An embodied generalist agent in 3d world,” arXiv preprint arXiv:2311.12871, 2023.
[9] J. Wen, Y. Zhu, J. Li, M. Zhu, Z. Tang, K. Wu, Z. Xu, N. Liu, R. Cheng, C. Shen, et al., “Tinyvla: Towards fast, data-efficient visionlanguage-action models for robotic manipulation,” IEEE Robotics and Automation Letters, 2025.
[10] O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al., “Octo: An open-source generalist robot policy,” arXiv preprint arXiv:2405.12213, 2024.
[11] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al., “π0.5: a vision-language-action model with open-world generalization,” arXiv preprint arXiv:2504.16054, 2025.
[12] NVIDIA, N. C. Johan Bjorck andFernando Casta˜neda, X. Da, R. Ding, L. J. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, J. Jang, Z. Jiang, J. Kautz, K. Kundalia, L. Lao, Z. Li, Z. Lin, K. Lin, G. Liu, E. Llontop, L. Magne, A. Mandlekar, A. Narayan, S. Nasiriany, S. Reed, Y. L. Tan, G. Wang, Z. Wang, J. Wang, Q. Wang, J. Xiang, Y. Xie, Y. Xu, Z. Xu, S. Ye, Z. Yu, A. Zhang, H. Zhang, Y. Zhao, R. Zheng, and Y. Zhu, “GR00T N1: An open foundation model for generalist humanoid robots,” in ArXiv Preprint, March 2025.
[13] M. J. Kim, C. Finn, and P. Liang, “Fine-tuning vision-language-action models: Optimizing speed and success,” arXiv preprint arXiv:2502.19645, 2025.
[14] Q. Gu, Y. Ju, S. Sun, I. Gilitschenski, H. Nishimura, M. Itkina, and F. Shkurti, “Safe: Multitask failure detection for vision-language-action models,” 2025. [Online]. Available: https://arxiv.org/abs/2506.09937
[15] Y. Li, Y. Deng, J. Zhang, J. Jang, M. Memmel, R. Yu, C. R. Garrett, F. Ramos, D. Fox, A. Li, A. Gupta, and A. Goyal, “Hamster: Hierarchical action models for open-world robot manipulation,” 2025. [Online]. Available: https://arxiv.org/abs/2502.05485
[16] K. Crammer, M. Kearns, and J. Wortman, “Learning from multiple sources,” 2008.
[17] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks: The sequential learning problem,” ser. Psychology of Learning and Motivation, G. H. Bower, Ed. Academic Press, 1989, vol. 24, pp. 109–165. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0079742108605368
[18] E. L. Aleixo, J. G. Colonna, M. Cristo, and E. Fernandes, “Catastrophic forgetting in deep learning: A comprehensive taxonomy,” arXiv preprint arXiv:2312.10549, 2023.
[19] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. GrabskaBarwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell, “Reply to husz´ar: The elastic weight consolidation penalty is empirically valid,” Proceedings of the National Academy of Sciences, vol. 115, no. 11, pp. E2498–E2498, 2018. [Online]. Available: https://www.pnas.org/doi/abs/10.1073/pnas.1800157115
[20] A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell, “Progressive neural networks,” arXiv preprint arXiv:1606.04671, 2016.
[21] A. Mallya and S. Lazebnik, “Packnet: Adding multiple tasks to a single network by iterative pruning,” in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2018, pp. 7765–7773.
[22] Y. Long, K. Chen, L. Jin, and M. Shang, “Drae: Dynamic retrievalaugmented expert networks for lifelong learning and task adaptation in robotics,” arXiv preprint arXiv:2507.04661, 2025.
[23] W. Su, Y. Tang, Q. Ai, J. Yan, C. Wang, H. Wang, Z. Ye, Y. Zhou, and Y. Liu, “Parametric retrieval augmented generation,” in Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2025, pp. 1240–1250.
[24] Z. Ghahramani and M. Beal, “Variational inference for bayesian mixtures of factor analysers,” Advances in neural information processing systems, vol. 12, 1999.
[25] L. Xu, H. Xie, S.-Z. J. Qin, X. Tao, and F. L. Wang, “Parameter-efficient fine-tuning methods for pretrained language models: A critical review and assessment,” 2023. [Online]. Available: https://arxiv.org/abs/2312.12148
[26] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” 2021. [Online]. Available: https://arxiv.org/abs/2106.09685
[27] T. Hospedales, A. Antoniou, P. Micaelli, and A. Storkey, “Metalearning in neural networks: A survey,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 9, pp. 5149–5169, 2021.
[28] D. Rolnick, A. Ahuja, J. Schwarz, T. P. Lillicrap, and G. Wayne, “Experience replay for continual learning,” 2019. [Online]. Available: https://arxiv.org/abs/1811.11682
[29] G. M. van de Ven, N. Soures, and D. Kudithipudi, Continual learning and catastrophic forgetting. Elsevier, 2025, p. 153–168. [Online]. Available: http://dx.doi.org/10.1016/B978-0-443-15754-7.00073-0
[30] W. Hu, Z. Lin, B. Liu, C. Tao, Z. T. Tao, D. Zhao, J. Ma, and R. Yan, “Overcoming catastrophic forgetting for continual learning via model adaptation,” in International conference on learning representations, 2019.
[31] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. K¨uttler, M. Lewis, W.-t. Yih, T. Rockt¨aschel, et al., “Retrievalaugmented generation for knowledge-intensive nlp tasks,” Advances in neural information processing systems, vol. 33, pp. 9459–9474, 2020.
[32] Z. Guo, L. Xia, Y. Yu, T. Ao, and C. Huang, “Lightrag: Simple and fast retrieval-augmented generation,” arXiv preprint arXiv:2410.05779, 2024.
[33] K. Sawarkar, A. Mangal, and S. R. Solanki, “Blended rag: Improving rag (retriever-augmented generation) accuracy with semantic search and hybrid query-based retrievers,” in 2024 IEEE 7th international conference on multimedia information processing and retrieval (MIPR). IEEE, 2024, pp. 155–161.
[34] A. Santoro, S. Bartunov, M. Botvinick, D. Wierstra, and T. Lillicrap, “Meta-learning with memory-augmented neural networks,” in International conference on machine learning. PMLR, 2016, pp. 1842–1850.
[35] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, “Libero: Benchmarking knowledge transfer for lifelong robot learning,” 2023. [Online]. Available: https://arxiv.org/abs/2306.03310
