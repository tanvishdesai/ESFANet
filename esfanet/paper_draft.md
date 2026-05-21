### 1. Title

**Uncertainty-Gated Evidence Fusion for Trustworthy Histopathology Classification with Evidential Deep Learning Agents**

### 2. Abstract

Reliable medical image classification requires not only high predictive performance but also calibrated uncertainty and interpretable failure behavior. This work proposes **DST-AgenticNet**, a lightweight multi-agent classification architecture for breast histopathology patches in which modality-specialized “agents” produce **evidence** (Dirichlet parameters) rather than logits, and a learned **gating network** performs **evidence-level fusion** to dynamically weight agents per sample. Concretely, the model combines (i) an RGB agent based on a pretrained ResNet-18 encoder, (ii) an edge agent operating on fixed Sobel magnitude maps, and (iii) a frequency agent operating on FFT amplitude summaries. All agents are trained with an evidential deep learning objective that couples data fit with a KL regularizer to avoid overconfident predictions. Experiments comparing fusion strategies show that uncertainty-gated fusion achieves strong classification performance while maintaining low calibration error: gated fusion reaches **0.9680 accuracy**, **0.9605 macro-F1**, and **0.0107 ECE**, while a “monopoly” gating variant attains **0.0057 ECE** with slightly reduced accuracy/F1. The analysis of learned gate weights reveals interpretable global trust allocations and a failure mode in which the gate collapses to a single dominant agent. These results highlight evidence-level dynamic gating as a practical path toward calibrated and interpretable multimodal decision-making in safety-critical imaging.

### 3. Introduction

Breast histopathology patch classification is a core component of computer-aided diagnosis workflows and is often framed as binary discrimination between malignant and non-malignant tissue. While modern deep networks can achieve strong accuracy, medical deployment requires additional properties: **(i)** reliable uncertainty estimates to support risk-aware decisions, and **(ii)** interpretable mechanisms that can explain which information sources the model trusts for a given prediction. Standard deterministic softmax classifiers are known to be miscalibrated and can produce high-confidence errors, motivating alternatives that expose uncertainty in a principled way [1], [3].

Multimodal and multi-expert designs offer another avenue for robustness: different “views” (e.g., texture, structure, spectral content) can complement each other when one view is degraded or non-informative. However, typical fusion approaches average or concatenate features/logits, which may implicitly assume equal reliability across views and can amplify overconfidence when all modalities are treated as equally trustworthy. Recent work on uncertainty-aware fusion emphasizes dynamic weighting of experts using uncertainty signals [5], [6], but many such methods rely on sampling-based Bayesian uncertainty, increasing compute and complicating deployment.

This paper proposes **DST-AgenticNet**, a multi-agent architecture that fuses **evidence** (Dirichlet concentration parameters) produced by several modality-specialized agents trained with an evidential deep learning objective [1]. A compact gating network predicts per-sample agent weights from the concatenated evidences and performs a weighted evidence sum. This design aligns the fusion mechanism with the semantics of **evidential** uncertainty: the gate modulates the amount of evidence contributed by each agent rather than post-hoc scaling of probabilities. Empirically, the approach yields competitive classification performance with low expected calibration error (ECE), and qualitatively exposes interpretable “trust dynamics” via gate weights.

**Contributions.**  
- **Evidence-level dynamic fusion**: a gating mechanism that weights per-agent **evidence** (Dirichlet parameters) rather than logits/probabilities, enabling uncertainty-aware decision-level fusion.  
- **Multi-agent evidential modeling**: three complementary agents (RGB, edge, frequency) trained using an evidential classification loss with annealed KL regularization [1].  
- **Practical training stabilizers and analysis**: explicit investigation of **gate collapse (“monopoly”)** vs. collaborative weighting and the resulting calibration/accuracy trade-offs.  
- **Evaluation with calibration**: reporting accuracy, macro-F1, and ECE to assess both performance and reliability.

### 4. Related Work

#### 4.1 Evidential deep learning for calibrated uncertainty

Evidential deep learning (EDL) replaces point-probability outputs with parameters of a Dirichlet distribution, enabling uncertainty estimates from a single forward pass [1]. Surveys consolidate training strategies and theoretical interpretations of evidence and belief in safety-critical contexts [3]. Extensions to continuous targets further demonstrate that evidential modeling can scale to complex vision tasks while avoiding sampling-based Bayesian inference [8]. Our work builds directly on the classification EDL formulation [1], using evidence outputs and KL regularization with annealing to reduce overconfidence.

#### 4.2 Uncertainty-aware multimodal fusion and mixture-of-experts

Multimodal fusion methods span data-, feature-, and decision-level designs, with attention-based fusion and uncertainty-aware weighting emerging as practical strategies [4]. Uncertainty-encoded mixture-of-experts architectures explicitly compute expert weights from uncertainty signals to improve robustness under modality degradation [5], and probabilistic fusion rules such as uncertainty-aware noisy-or have been studied to prevent brittle dependence on a single modality [6]. In multi-agent perception, attention mechanisms weight contributions across agents for improved downstream performance [2]. DST-AgenticNet aligns with these ideas but differs in two key aspects: it uses **EDL-derived evidence** to represent both confidence and uncertainty [1], and it performs **evidence-level** (Dirichlet-parameter) fusion rather than feature/logit fusion.

#### 4.3 Dempster–Shafer theory and evidential fusion in deep networks

Dempster–Shafer theory (DST) provides a framework for combining uncertain evidence from multiple sources. Prior work integrates DS layers with deep networks for set-valued classification and evidence combination [7] and demonstrates DST-based fusion in ensemble CNNs for robust recognition [9]. In medical imaging, mutual evidential learning and evidential fusion strategies have been proposed to enhance uncertainty calibration and robustness [10]. Our method is inspired by these evidence-combination principles: each agent produces evidence and the fusion module performs a weighted evidence aggregation consistent with an evidential interpretation, while remaining simple and differentiable for end-to-end training.

#### 4.4 Agentic and multi-agent trust framing

Surveys of agentic AI emphasize the need for trust and risk management in systems composed of multiple interacting components [12], [13]. Multi-agent settings highlight challenges in heterogeneous collaboration, which motivates mechanisms for adaptive weighting and coordination [14]. While our model is not a generative tool-using agent system, it adopts an **agentic** abstraction—multiple specialized “experts” coordinated by a learned policy—and evaluates trust dynamics via calibration metrics and learned contribution weights [12]. This framing connects uncertainty-aware fusion to broader trustworthy AI considerations.

### 5. Proposed Methodology

#### 5.1 Overall architecture overview

DST-AgenticNet is a supervised binary classifier that maps an input RGB histopathology patch \(x \in \mathbb{R}^{3 \times H \times W}\) to (i) per-agent evidence vectors and (ii) a fused evidence vector used for prediction. The model comprises three agents:

- **RGB agent**: a pretrained ResNet-18 feature extractor (final pooling output) followed by an MLP head that outputs nonnegative evidence via Softplus.
- **Edge agent**: a fixed Sobel operator generates a single-channel edge magnitude map, processed by a small CNN and an evidence head (Softplus).
- **Frequency agent**: a 2D FFT on the grayscale image yields amplitude spectra; a pooled spectral vector is mapped to evidence via an MLP head (Softplus).

Let \(e^{(a)}(x) \in \mathbb{R}_{\ge 0}^{K}\) denote the evidence produced by agent \(a \in \{1,2,3\}\) for \(K\) classes (here \(K=2\)). These evidences are stacked into a tensor \(E(x) \in \mathbb{R}_{\ge 0}^{A \times K}\) with \(A=3\).

#### 5.2 Evidential predictive distribution

Following EDL [1], evidence is converted to Dirichlet concentration parameters:
\[
\alpha^{(a)}(x) = e^{(a)}(x) + \mathbf{1}.
\]
Given any evidence vector \(e\), the predictive class probabilities are computed from the Dirichlet mean:
\[
p(y=k \mid x) = \frac{\alpha_k}{\sum_{j=1}^{K} \alpha_j}.
\]
This probability is used both for prediction and for calibration evaluation (ECE).

#### 5.3 Uncertainty-gated evidence fusion (proposed)

The fusion module predicts per-sample agent weights from the concatenated evidences:
\[
w(x) = \mathrm{softmax}\left(g\left(\mathrm{vec}(E(x))\right)\right) \in \Delta^{A-1},
\]
where \(g(\cdot)\) is a two-layer MLP with ReLU and \(\Delta^{A-1}\) is the \(A\)-simplex. The fused evidence is a weighted sum:
\[
e^{(\mathrm{fuse})}(x) = \sum_{a=1}^{A} w_a(x)\, e^{(a)}(x).
\]
The key design choice is that the gate weights **evidence** rather than probabilities/logits. The fused evidence is then converted to \(\alpha^{(\mathrm{fuse})}(x) = e^{(\mathrm{fuse})}(x) + \mathbf{1}\) and used for prediction and loss computation.

We compare the proposed fusion against two ablations implemented in the same codebase:
- **RGB-only**: predict using only \(e^{(\mathrm{RGB})}\).
- **Naive fusion**: use unweighted evidence summation \(e^{(\mathrm{fuse})} = \sum_a e^{(a)}\).

#### 5.4 Training objective and optimization

All evidence heads are trained with an evidential loss adapted from [1]. Given one-hot label \(y \in \{0,1\}^K\) and belief vector \(\hat{p}=\alpha/S\) with \(S=\sum_k \alpha_k\), the loss includes a data-fit term with an uncertainty-dependent variance component and a KL regularizer encouraging appropriate uncertainty:
\[
\mathcal{L}_{\mathrm{EDL}} = \mathcal{L}_{\mathrm{mse}}(y, \hat{p}, \alpha) + \lambda(t)\,\mathrm{KL}\!\left(\tilde{\alpha}\,\|\,\mathbf{1}\right),
\]
where \(\lambda(t)\) is an annealing coefficient increasing with epoch \(t\), and \(\tilde{\alpha}\) is a label-dependent transformation used in the code implementation [1].

To prevent weak experts from collapsing during fusion training, an auxiliary “deep supervision” term is applied to individual agents (scaled by 0.2 in the provided implementation) so that each agent continues to receive a learning signal even when its gate weight is small.

Finally, the implementation explores regularizers intended to mitigate **gate collapse**. One variant nudges the gate toward non-degenerate weighting by penalizing deviation from a uniform target weight distribution via an MSE penalty, which empirically produces more collaborative global weights. A contrasting “monopoly” regime is characterized by near-one weight on a single agent, yielding a qualitatively different trust distribution.

### 6. Experiments and Results

#### 6.1 Experimental Setup

**Task and data.** The implementation trains on a breast histopathology patch dataset organized as PNG images with class labels parsed from filenames (binary classification). Data are split with stratification into **70% train**, **15% validation**, and **15% test**.

**Preprocessing.** Images are resized to \(96 \times 96\), converted to tensors, and normalized using ImageNet mean and standard deviation.

**Training.** The code trains for **20 epochs** with batch size **128**, using **AdamW** with learning rate **1e-4**. Training is performed on a **single CUDA-capable GPU** when available (otherwise CPU).

**Metrics.** We report:
- **Accuracy**
- **Macro-F1**
- **Expected Calibration Error (ECE)** computed with 15 confidence bins from the Dirichlet-mean probabilities.

#### 6.2 Quantitative Results

Table 1 reproduces the results from `results.txt` exactly.

**Table 1. Test performance and calibration across fusion strategies (from `results.txt`).**

| Method | Accuracy | Macro-F1 | ECE (↓) |
|---|---:|---:|---:|
| GATED_FUSION | 0.9680 | 0.9605 | 0.0107 |
| GATED_FUSION (monopoly; collapse) | 0.9650 | 0.9570 | 0.0057 |
| NAIVE_FUSION | 0.9682 | 0.9608 | 0.0118 |
| RGB_ONLY | 0.9687 | 0.9614 | 0.0122 |

#### 6.3 Results Discussion

**Performance vs. calibration trade-off.** The reported results are close in accuracy and macro-F1 across all strategies, suggesting that the classification task is solvable even with a single strong RGB backbone. However, calibration differs: gated fusion improves ECE relative to RGB-only (0.0107 vs. 0.0122), indicating better alignment between predicted confidence and empirical correctness. The “monopoly” variant achieves a lower ECE (0.0057) but is treated as a **failure mode**: it corresponds to near-deterministic selection of a single agent (RGB), which undermines the intended multi-agent robustness and trust decomposition.

**Interpretable trust dynamics.** The learned gating weights provide a direct diagnostic of which agent the model trusts. In a collaborative regime, global average weights are distributed across RGB, edge, and frequency agents (approximately 0.395 / 0.310 / 0.296), and the per-sample weight distributions show non-trivial variation across images, consistent with adaptive fusion. In contrast, the monopoly regime exhibits near-deterministic selection of the RGB agent (approximately 0.998 weight), which reduces the model’s effective diversity and can weaken robustness to modality-specific failures.

**Why “better numbers” can still be failure.** A lower ECE in the monopoly regime is not, by itself, evidence of better multi-agent behavior: if the gate collapses, the system degenerates into an RGB-only predictor with extra overhead. This collapse removes the opportunity for edge/frequency agents to compensate when RGB is corrupted or insufficient. In other words, monopoly can be **well-calibrated yet fragile**, because calibration is measured on the same test distribution and does not guarantee resilience to agent-specific degradations or distribution shifts.

**Why evidence-level gating matters.** Unlike post-hoc probability scaling, gating at the evidence level modulates the Dirichlet concentration parameters before probabilities are computed. This is consistent with an evidential interpretation of uncertainty [1], and aligns with prior uncertainty-aware mixture-of-experts designs that gate experts based on reliability signals [5], [6]. The evidential objective further constrains the model to avoid overconfident predictions by penalizing inappropriate evidence accumulation through the KL regularizer [1].

### 7. Limitations

**Dataset and evaluation scope.** The current experiments focus on a single binary histopathology classification setting with a fixed train/validation/test split, and the reported results do not establish cross-dataset generalization.

**Fusion operator simplicity.** The fusion step is implemented as a weighted evidence sum. While differentiable and stable, this is a simplified approximation of richer DST combination operators that explicitly model conflict [7], [9].

**Gate collapse risk.** The system can exhibit a failure mode where the gate collapses to a single dominant agent (“monopoly”), reducing the benefits of multi-agent specialization. Even if some metrics (e.g., ECE) improve under collapse on a fixed test set, this behavior is undesirable because it eliminates complementary information sources and can reduce robustness to modality-specific corruption. Collapse should therefore be treated as a failure mode and actively monitored/mitigated.

**Compute and implementation constraints.** Although EDL avoids sampling-based uncertainty estimation [1], the model still includes multiple forward paths (agents), increasing inference cost compared to a single-network baseline. Hardware specifics (GPU model, memory) were not logged in the provided implementation and therefore are not reported.

### 8. Conclusion and Future Work

This paper introduced DST-AgenticNet, an uncertainty-aware multi-agent classifier that performs **evidence-level dynamic fusion** using a learned gate over evidential agents. The approach leverages evidential deep learning to obtain uncertainty from a single forward pass [1] and exposes interpretable trust dynamics through gating weights. Empirical results show strong classification performance with low calibration error, and reveal a meaningful collapse-vs-collaboration behavior that can be analyzed and regularized.

Future work includes: (i) evaluating generalization across additional histopathology datasets and shifts, (ii) integrating explicit conflict modeling inspired by DST layers [7], [9], and (iii) extending the gating mechanism with uncertainty features (e.g., total evidence or subjective uncertainty mass) to more directly align gating with reliability signals, as suggested by uncertainty-aware fusion literature [5], [6], [10], [11].

### 9. References

[1] M. Sensoy et al., “Evidential Deep Learning to Quantify Classification Uncertainty,” 2018.  
[2] Ahmed et al., “Attention Based Feature Fusion For Multi-Agent Collaborative Perception,” 2023.  
[3] Y. Gao et al., “A Comprehensive Survey on Evidential Deep Learning and Its Applications,” 2024.  
[4] Li et al., “Multimodal Alignment and Fusion: A Survey,” 2025.  
[5] Lou et al., “Uncertainty-Encoded Multi-Modal Fusion for Robust Object Detection in Autonomous Driving,” 2023.  
[6] Tian et al., “UNO: Uncertainty-aware Noisy-Or Multimodal Fusion for Unanticipated Input Degradation,” 2020.  
[7] Tong et al., “An evidential classifier based on Dempster-Shafer theory and deep learning,” 2021.  
[8] A. Amini et al., “Deep Evidential Regression,” 2020.  
[9] Yaghoubi et al., “CNN-DST: ensemble deep learning based on Dempster-Shafer theory for vibration-based fault recognition,” 2021.  
[10] He et al., “Mutual Evidential Deep Learning for Medical Image Segmentation,” 2025.  
[11] Fang et al., “Dynamic Uncertainty-aware Multimodal Fusion for Outdoor Health Monitoring,” 2025.  
[12] Raza et al., “TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management,” 2025.  
[13] Ali and Dornaika, “Agentic AI: A Comprehensive Survey of Architectures, Applications, and Future Directions,” 2025.  
[14] Feng et al., “Multi-Agent Embodied AI: Advances and Future Directions,” 2025.  
[15] Huang et al., “Latent Distribution Decoupling: A Probabilistic Framework for Uncertainty-Aware Multimodal Emotion Recognition,” 2025.  


