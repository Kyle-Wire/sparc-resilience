# Physics-Constrained Machine Learning Research

Sources that shaped the **SharedTrunk + CityHead architecture**, the **10-term PDE loss**, **SIREN layers**, **MC-Dropout uncertainty**, **sparse spatial attention**, **JEPA pretraining**, and **physics guardrails** in Stage 2.

---

## SIREN — Sinusoidal Representation Networks

> **Sitzmann, V., Martel, J.N.P., Bergman, A.W., Lindell, D.B., & Wetzstein, G. (2020).** "Implicit Neural Representations with Periodic Activation Functions." *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 7462–7473. https://arxiv.org/abs/2006.09661

The `SharedTrunk` physics encoder uses SIREN layers (`SIRENLayer` in `sparc/models/spatial_attention.py`). SIREN's sinusoidal activations make the network naturally differentiable and its derivatives representable — ideal for PDE-supervised training where spatial gradients of the temperature field must satisfy physical constraint equations. The ω-scaled initialization ensures activations remain in the stable range of the sine function.

---

## Physics-Informed Neural Networks (PINNs)

> **Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).** "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045

> **Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021).** "Physics-informed machine learning." *Nature Reviews Physics*, 3, 422–440. https://doi.org/10.1038/s42254-021-00314-5

The 10-term PDE loss in `sparc/training/pde_loss.py` embeds physical equations directly into the training objective rather than treating physics as a post-hoc check. This follows the PINN paradigm: the neural network is constrained to be a physically plausible function by including PDE residuals in the loss, alongside data fit. SPARC's eight core terms cover heat diffusion (α∇²T − S), surface energy balance (Q* − Q_H − Q_E), directional consistency, anisotropy alignment, gradient-flux (Fourier's law), Gaussian curvature regularization, and α-field smoothness/prior terms.

---

## Surface Energy Balance

> **Oke, T.R. (1988).** "The urban energy balance." *Progress in Physical Geography*, 12(4), 471–508. https://doi.org/10.1177/030913338801200401

> **Kustas, W.P., & Norman, J.M. (1996).** "Use of remote sensing for evapotranspiration monitoring over land surfaces." *Hydrological Sciences Journal*, 41(4), 495–516. https://doi.org/10.1080/02626669609491522

The energy-balance PDE term in the loss (Q* − Q_H − Q_E ≈ 0) encodes conservation of energy at the urban surface: net radiation must equal sensible heat flux plus latent heat flux plus storage. This is the fundamental urban climatology equation and grounds the neural model in thermodynamic reality.

---

## Heat Diffusion / Thermal Diffusivity

> **Carslaw, H.S., & Jaeger, J.C. (1959).** *Conduction of Heat in Solids*, 2nd ed. Oxford University Press.

> **Fourier, J.B.J. (1822).** *Théorie analytique de la chaleur.* *(Historical foundation of Fourier's law, α∇²T heat equation.)*

The core diffusion term α∇²T − S = 0 and gradient-flux term (Fourier's law: **q** = −α∇T) form the physical backbone of the PDE loss. The spatially-varying process-rate field α(x) — learned by `ProcessRateNet` in `sparc/models/process_rate_net.py` — encodes local thermal diffusivity, which varies across impervious surfaces, canopy, and water bodies.

---

## MC-Dropout for Predictive Uncertainty

> **Gal, Y., & Ghahramani, Z. (2016).** "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, 1050–1059. https://arxiv.org/abs/1506.02142

> **Gal, Y. (2016).** *Uncertainty in Deep Learning.* PhD Thesis, University of Cambridge. http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf

MC-Dropout is left active at inference in the `CityHead`. 500 stochastic forward passes produce per-point predictive mean, standard deviation, and credible intervals. This follows Gal & Ghahramani (2016): keeping dropout active turns each forward pass into a sample from an approximate Bayesian posterior, making the ensemble spread a calibrated uncertainty estimate.

---

## Sparse Spatial Attention

> **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., & Polosukhin, I. (2017).** "Attention Is All You Need." *Advances in Neural Information Processing Systems (NeurIPS)*, 30. https://arxiv.org/abs/1706.03762

> **Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020).** "Reformer: The Efficient Transformer." *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2001.04451

`SparseSpatialAttention` in `sparc/models/spatial_attention.py` implements multi-head attention over a KNN spatial neighborhood. By restricting attention to k nearest neighbors, complexity drops from O(N²) to O(N · max_neighbors) — tractable at the 50k-point scale of the Providence UHI dataset. The attention weights are interpretable as a learned spatial-influence surface: each point's weight distribution shows which neighboring points influenced its prediction.

---

## Staged Curriculum Learning

> **Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).** "Curriculum Learning." *Proceedings of the 26th International Conference on Machine Learning (ICML)*, 41–48. https://doi.org/10.1145/1553374.1553380

The 10-term PDE loss uses a staged curriculum: heat diffusion activates at epoch 1, energy balance near epoch 10, directional/anisotropy terms near epoch 20, and the full stack by epoch 30 — each new term ramping linearly over 5 epochs. This follows curriculum learning principles: starting with simple constraints and progressively introducing more complex ones prevents optimizer shock and enables the network to build a physically consistent latent before being asked to satisfy the full constraint set simultaneously.

---

## JEPA — Joint Embedding Predictive Architecture

> **LeCun, Y. (2022).** "A Path Towards Autonomous Machine Intelligence." *OpenReview*. https://openreview.net/forum?id=BZ5a1r-kVsf

> **Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023).** "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. https://arxiv.org/abs/2301.08243

> **Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., & Ballas, N. (2024).** "V-JEPA: Latent Video Prediction for Visual Representation Learning." *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2404.08471

When `lambda_jepa > 0`, SPARC pretrains the SharedTrunk using a V-JEPA-2-AC–style objective. The `EMATrunk` (EMA copy of the online trunk, stop-gradient) provides target embeddings; the `LatentPredictor` maps `(context, ActionEmbedding)` to predicted target latents. The `ActionEmbedding` encodes treatment metadata (one-hot, |Δx|, sign, Δt), making the trunk action-conditioned — important for counterfactual reasoning in Stage 4.

---

## VICReg — Variance-Invariance-Covariance Regularization

> **Bardes, A., Ponce, J., & LeCun, Y. (2022).** "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2105.04906

The JEPA pretraining loss combines cosine similarity with VICReg to prevent representational collapse (all embeddings converging to a constant). VICReg's three terms enforce: variance within a batch (embeddings are spread), invariance between augmented views (same location → same representation), and covariance decorrelation (embedding dimensions are independent). This is essential for the trunk to learn a rich, non-degenerate latent space from unlabeled spatial patches.

---

## Differentiable Surrogates

> **Ba, J.L., Kiros, J.R., & Hinton, G.E. (2016).** "Layer Normalization." *arXiv*, 1607.06450. https://arxiv.org/abs/1607.06450

> **Kingma, D.P., & Ba, J. (2015).** "Adam: A Method for Stochastic Optimization." *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1412.6980

Each classical base model (GWR, GWRF, GGPGAM) is mirrored by a differentiable PyTorch surrogate pre-trained against its out-of-fold predictions. These surrogates (`sparc/models/surrogates.py`) enable the meta-learner to back-propagate through "classical model outputs" end-to-end — the gradient flows from the PDE loss through the meta-learner into the surrogate weights, implicitly refining the classical model's behavior via a differentiable proxy.

---

## Poisson / Advection-Diffusion PDE Solve (Tier 2 Scenarios)

> **LeVeque, R.J. (2007).** *Finite Difference Methods for Ordinary and Partial Differential Equations.* SIAM. ISBN: 978-0898716290.

> **Hundsdorfer, W., & Verwer, J.G. (2003).** *Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations.* Springer. ISBN: 978-3-642-05709-1.

The Tier 2 scenario engine (`pde_solve`) solves a forward Poisson or advection-diffusion equation under the proposed intervention's new forcing conditions. This captures spatial spillovers — a tree-planting intervention at point A cools point B through diffusive heat transport — which the simpler Tier 0/1 methods cannot represent.
