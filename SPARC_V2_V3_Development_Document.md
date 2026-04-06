# SPARC Version 2 — Locked Development Document

**SPARC Labs LLC | April 2026**
**Status: Locked for Development**
**Repository:** github.com/Kyle-Wire/sparc-resilience

---

## Preamble

This document is the locked technical specification for SPARC Version 2. It supersedes the separate V2/V3 phased plan. Rather than building intermediate components that would be replaced, this document specifies the single right architecture — designed once, built once, with every component serving the final system from day one.

V1 established that spatial causal inference is feasible and useful for climate resilience applications. V2 makes it end-to-end differentiable, fully Bayesian, physics-informed, and capable of probabilistic adaptation performance estimation across all 13 existing domains.

The core product promise does not change. For any adaptation measure in any location, SPARC tells you:

1. **What will likely happen** to the outcome variable
2. **How confident we are** — decomposed by source of uncertainty
3. **How the intervention ripples** through related variables
4. **Where it works** and where it doesn't

Every number backed by a probability. Every probability backed by a causal mechanism. Every mechanism assigned a data-supported confidence level.

---

## Guiding Principles

**Build it once, build it right.** No intermediate components that get replaced. Every file written in V2 serves the final architecture.

**Physics constrains, data informs.** Physics priors are inviolable constraints on sign and rough magnitude. Within those constraints the data is trusted. Where they conflict the tension is reported as a scientific finding.

**Interpretability is not optional.** Every architectural decision preserves the ability to ask: what is this model doing and why? Differentiable surrogates maintain coefficient surfaces. Attention weights are mappable. Process rate fields are validatable against literature.

**Uncertainty has sources.** Three distinct sources — graph structure (MC³), parameter values (NUTS), process rate (auxiliary network) — each implying a different practical response.

**End-to-end differentiability.** A single gradient connects the physics-informed loss to every component. Joint optimization under a unified physical objective.

**Backward compatibility.** All new behavior is opt-in via `project.yml`. V1 behavior fully reproducible by setting `meta_learner: lightgbm`, `inference: frequentist`, `process_rate: disabled`.

---

## What Does Not Change

- CLI interface (`sparc init / validate / run / scenario / report`)
- Streamlit UI — new config fields surface automatically
- All 13 existing domain templates
- Stage 0 correlogram analysis and auto-configuration
- Stage 1 GWEN variable selection
- DoWhy refutation tests in Stage 3
- Mode 1 and Mode 2 scenario simulation in Stage 4
- `project.yml` schema — all new fields optional with V1 defaults

---

## Timeline

| Component | Target |
|---|---|
| Surrogates + process rate net + spatial attention + neural meta-learner | Q2 2026 |
| Joint loss + full optimization stack | Q2 2026 |
| MC³ with parallel tempering | Q3 2026 |
| NUTS with autograd | Q3 2026 |
| Multi-resolution transfer learning | Q4 2026 |
| MAML spatial CV | Q4 2026 |
| JAMES Paper 1 submission | Q1 2027 |

---

## Paper Targets

| Paper | Content | Journal | Submission |
|---|---|---|---|
| 1 | MC³ + NUTS for spatial causal DAG inference. UHI application. Spatially-varying process rate validation. | JAMES | Q1 2027 |
| 2 | End-to-end differentiable spatial causal inference. Multi-resolution transfer. MAML generalization. | Nature Computational Science | Q3 2027 |

---

---

# PART ONE — OPTIMIZATION FRAMEWORK

---

## 1.1 — Why Optimization Strategy Matters Here

SPARC's loss surface is unlike standard machine learning problems. Three specific properties make naive optimization fail:

**Heterogeneous parameter scales.** Neural meta-learner weights operate at 1e-2 to 1e-4. Process rate network outputs operate at 1e-7 m²/s (thermal diffusivity). Causal coefficients operate at 1e-2 z-score units per percentage point. A single global learning rate cannot handle this — some components take enormous steps while others stagnate.

**Physics loss ill-conditioning.** The Laplacian term in the physics loss divides by resolution² (900 at 30m resolution). This amplifies gradient magnitude through that term by 900×, creating severe gradient imbalance relative to the MSE term. Without careful management this causes the physics loss to dominate and the model to ignore the data.

**SIREN activation geometry.** Sine activations in the physics encoder create a highly non-convex loss landscape with many sharp local minima corresponding to different frequency harmonics of the PDE solution. Standard gradient descent can tunnel into a wrong harmonic and get stuck. The gradient norm at the sine inflection points (±π/2 arguments) can be very large.

The optimization stack below is designed specifically for these three properties.

---

## 1.2 — Primary Optimizer: AdamW with Per-Component Learning Rates

**Why AdamW over standard SGD:** Adam maintains per-parameter adaptive learning rates via first and second moment estimates (m_t and v_t). The v_t term is a diagonal approximation to the Hessian — it automatically accounts for different parameter scales without requiring explicit per-parameter tuning. AdamW decouples weight decay from the gradient update, which is important for physics-informed models where L2 regularization on weights should not interact with the physics constraint.

**Why per-component learning rates:** Different components have fundamentally different loss landscape geometry. The SIREN physics encoder needs a smaller learning rate because its sinusoidal activations create sharper local geometry. The process rate network operates in physical units and needs a tighter rate to prevent large jumps in physical parameter space.

```python
def build_optimizer(model, process_rate_net, surrogates):
    """
    Per-component learning rates via AdamW parameter groups.
    Each group adapts independently — heterogeneous scales handled
    without manual intervention.
    """
    param_groups = [
        # Neural meta-learner streams
        {
            'params': model.base_enc.parameters(),
            'lr': 1e-3,
            'name': 'base_encoder'
        },
        {
            'params': model.physics_enc.parameters(),
            'lr': 5e-4,      # SIREN needs slower lr — sharp local geometry
            'name': 'physics_encoder'
        },
        {
            'params': model.spatial_enc.parameters(),
            'lr': 1e-3,
            'name': 'spatial_encoder'
        },
        {
            'params': model.alpha_emb.parameters(),
            'lr': 1e-3,
            'name': 'alpha_embedding'
        },
        {
            'params': model.fusion.parameters(),
            'lr': 1e-3,
            'name': 'fusion'
        },
        {
            'params': model.regression_head.parameters(),
            'lr': 1e-3,
            'name': 'regression_head'
        },
        {
            'params': model.exceedance_heads.parameters(),
            'lr': 1e-3,
            'name': 'exceedance_heads'
        },
        # Process rate network — physical units, tight lr
        {
            'params': process_rate_net.parameters(),
            'lr': 1e-4,
            'name': 'process_rate'
        },
        # Differentiable surrogates
        {
            'params': surrogates['gwr'].parameters(),
            'lr': 1e-3,
            'name': 'diff_gwr'
        },
        {
            'params': surrogates['gwrf'].parameters(),
            'lr': 1e-3,
            'name': 'diff_gwrf'
        },
        {
            'params': surrogates['ggpgam'].parameters(),
            'lr': 1e-3,
            'name': 'diff_ggpgam'
        },
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=1e-4,
        betas=(0.9, 0.999),   # momentum (beta1) and adaptive scale (beta2)
        eps=1e-8
    )

    return optimizer
```

**The role of betas:** `beta1=0.9` is the momentum coefficient — it determines how much velocity from previous gradient steps is retained. `beta2=0.999` governs the second moment (curvature estimate) accumulation. Together they implement a form of adaptive momentum that naturally handles the heterogeneous gradient landscape across SPARC's components.

---

## 1.3 — Momentum: Implementation and Physical Interpretation

Momentum in the optimization context accumulates a velocity vector in the direction of persistent gradient flow. For SPARC this is physically meaningful: the physics loss and data fidelity loss often point in slightly different directions at each batch. Momentum averages their gradient signals over time rather than thrashing between them at each step.

**Standard momentum vs. Nesterov:**

```
Standard momentum:
    v_t     = beta * v_{t-1} + grad(theta_t)
    theta_t+1 = theta_t - lr * v_t

    Problem: computes gradient at current position, then applies
             accumulated momentum. Can overshoot near optima.

Nesterov accelerated gradient (NAG):
    v_t     = beta * v_{t-1} + grad(theta_t - lr * beta * v_{t-1})
    theta_t+1 = theta_t - lr * v_t

    Computes gradient at the momentum-predicted future position.
    Looks ahead before stepping.
    Converges faster, more stable near optima.
    Corrects overshoot before it happens.
```

**Enabling Nesterov in practice:** Adam does not have a native Nesterov mode, but the equivalent behavior is achieved by using the `amsgrad=True` variant of Adam which uses the maximum of past squared gradients for the denominator — this provides stronger convergence guarantees and more conservative step sizes in directions of high curvature, which approximates the look-ahead behavior of Nesterov for adaptive methods.

Alternatively, for domains where Adam does not converge well (empirically, some wildfire and coastal domains with highly skewed feature distributions), SGD with explicit Nesterov can be used:

```python
# SGD with Nesterov momentum as fallback optimizer for specific domains
sgd_optimizer = torch.optim.SGD(
    param_groups,
    momentum=0.9,
    nesterov=True,          # look-ahead gradient
    weight_decay=1e-4
)
```

---

## 1.4 — Solution Set for Exploding Gradients

Three distinct sources of gradient explosion in SPARC, each with a targeted solution.

### Source 1 — SIREN Physics Encoder

Sine activations have gradient `d/dx[sin(ω·x)] = ω·cos(ω·x)`. At `ω=30` (the standard SIREN frequency), this multiplies gradients by up to 30 at each neuron. In a 3-layer SIREN network this creates potential gradient amplification of 30³ = 27,000 before clipping.

**Solution: SIREN-specific initialization + pre-activation layer normalization**

```python
class SIRENLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega=30.0, is_first=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega = omega

        # SIREN initialization — critical for stable training
        # First layer: uniform(-1/in_dim, 1/in_dim)
        # Hidden layers: uniform(-sqrt(6/in_dim)/omega, sqrt(6/in_dim)/omega)
        if is_first:
            nn.init.uniform_(self.linear.weight, -1/in_dim, 1/in_dim)
        else:
            bound = np.sqrt(6 / in_dim) / omega
            nn.init.uniform_(self.linear.weight, -bound, bound)

        # Pre-activation layer norm prevents gradient accumulation
        self.layer_norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        x = self.layer_norm(x)           # normalize before sine
        return torch.sin(self.omega * self.linear(x))
```

### Source 2 — Physics Loss Laplacian Scaling

The finite-difference Laplacian divides by `resolution²`. At 30m resolution this is 900. If the physics loss receives a gradient of magnitude g, the gradient flowing back through the Laplacian computation is amplified by 1/900² ≈ 1.2×10⁻⁶ per unit change in prediction — but in the forward pass the physics residual is amplified by 900, creating large forward values that then receive large gradients.

**Solution: Physics loss gradient scaling + normalized residual**

```python
def compute_physics_residual(T_pred, alpha, neighbor_idx, source_term, resolution):
    valid = (neighbor_idx != -1).all(dim=1)
    T_n = T_pred[neighbor_idx[valid, 0]]
    T_s = T_pred[neighbor_idx[valid, 1]]
    T_e = T_pred[neighbor_idx[valid, 2]]
    T_w = T_pred[neighbor_idx[valid, 3]]

    # Raw Laplacian
    laplacian = (T_n + T_s + T_e + T_w - 4*T_pred[valid]) / (resolution**2)

    # Normalize residual by its running standard deviation
    # Prevents physics loss from dominating when poorly initialized
    physics_residual = laplacian - (source_term[valid] / alpha[valid])

    # Normalize by residual magnitude — keeps physics loss on same scale as MSE
    residual_scale = physics_residual.detach().std().clamp(min=1e-6)
    normalized_residual = physics_residual / residual_scale

    return normalized_residual, laplacian, valid
```

### Source 3 — Cross-Component Gradient Interference

When gradients flow backward through the joint loss, the physics gradient, MSE gradient, and surrogate gradient can interfere constructively at certain parameter configurations — each individually acceptable but summing to an explosive total.

**Solution: Global gradient norm clipping applied after backward, before optimizer step**

```python
def training_step(model, process_rate_net, surrogates, batch,
                  optimizer, clip_norm=1.0):
    optimizer.zero_grad()

    # Forward pass
    outputs = model(**batch)
    loss, loss_components = compute_joint_loss(outputs, batch)

    # Backward pass
    loss.backward()

    # Global gradient norm clipping
    # Clips the TOTAL gradient norm across ALL parameters simultaneously
    # Preserves gradient direction — only reduces magnitude
    # Applied before optimizer step — optimizer never sees explosive gradients
    all_params = (
        list(model.parameters()) +
        list(process_rate_net.parameters()) +
        list(surrogates['gwr'].parameters()) +
        list(surrogates['gwrf'].parameters()) +
        list(surrogates['ggpgam'].parameters())
    )
    total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=clip_norm)

    # Log gradient norm for monitoring — flag if consistently near clip threshold
    if total_norm > clip_norm * 0.9:
        log_gradient_warning(total_norm, clip_norm)

    optimizer.step()

    return loss.item(), loss_components, total_norm.item()
```

**Choosing `clip_norm`:** The CMA-ES hyperparameter search (Section 1.6) includes `clip_norm` in its search space. A reasonable starting range is [0.5, 5.0]. If the model consistently clips (total_norm >> clip_norm), the physics loss lambda is too large for the current training stage — this is the most common failure mode and is addressed by the training curriculum (Section 1.7).

**Per-component physics gradient clipping as additional safeguard:**

```python
# Applied specifically to physics loss gradient before combining with other losses
def physics_safe_loss(physics_residual, lambda_physics, clip_value=0.1):
    physics_loss = torch.mean(physics_residual**2)
    scaled = lambda_physics * physics_loss

    # Detach if physics gradient magnitude exceeds safe threshold
    # Prevents physics loss from overwhelming data fidelity in early training
    if scaled.detach().item() > clip_value:
        scaled = scaled / (scaled.detach() / clip_value)

    return scaled
```

---

## 1.5 — Learning Rate Scheduling

```python
def build_scheduler(optimizer, n_epochs, warmup_epochs=10):
    """
    Cosine annealing with warm restarts.
    T_0=10: first restart at epoch 10
    T_mult=2: each subsequent cycle is 2x longer
    Cycles: 10, 20, 40, 80 epochs

    Warm restarts escape local minima — when lr jumps back up,
    the model can escape basins it was stuck in.
    Particularly effective combined with SWA in final stage.
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    return scheduler

def warmup_scheduler(optimizer, warmup_epochs, base_scheduler):
    """
    Linear warmup for first warmup_epochs.
    Prevents large initial gradient steps from destabilizing SIREN weights.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1.0

    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, base_scheduler], milestones=[warmup_epochs]
    )
```

---

## 1.6 — CMA-ES Hybrid Hyperparameter Optimization

Gradient descent finds local optima in hyperparameter space just as it does in parameter space. CMA-ES provides global exploration of the hyperparameter landscape before committing to gradient-based training — finding configurations that gradient tuning cannot reach from a random starting point.

**Why CMA-ES specifically:** CMA-ES maintains and adapts a full covariance matrix over the search space. This means it learns which hyperparameters co-vary — if high `lambda_physics` requires low `lambda_smooth` for stability, CMA-ES learns this correlation and proposes candidates that respect it. Standard grid search and random search cannot capture these dependencies.

**What CMA-ES searches over:**

```python
# Hyperparameter search space — all in log space for scale invariance
cma_search_space = {
    # Loss weights
    'log_lambda_physics':       (-4, -1),     # 1e-4 to 1e-1
    'log_lambda_smooth':        (-5, -2),     # 1e-5 to 1e-2
    'log_lambda_alpha_smooth':  (-5, -2),
    'log_lambda_prior':         (-5, -2),
    'log_lambda_base':          (-3,  0),     # 1e-3 to 1.0
    'log_lambda_neighbor':      (-4, -1),

    # Learning rates
    'log_physics_enc_lr':       (-4, -3),     # 1e-4 to 1e-3
    'log_process_rate_lr':      (-5, -4),     # 1e-5 to 1e-4

    # Architecture
    'log_clip_norm':            (-1,  1),     # 0.1 to 10.0
    'dropout':                   (0.05, 0.3),

    # Training dynamics
    'log_swa_lr_max':           (-3, -1),
    'beta1':                     (0.85, 0.95),
}

def cma_es_objective(log_params, model_class, data, domain_config, n_eval_epochs=15):
    """
    Objective function for CMA-ES.
    Trains model for n_eval_epochs with given hyperparams.
    Returns negative spatial CV R² (CMA-ES minimizes).
    Parallelized across population — each candidate trains independently.
    """
    params = {k: np.exp(v) if 'log_' in k else v
              for k, v in zip(cma_search_space.keys(), log_params)}

    model = model_class(**params)
    optimizer = build_optimizer(model, **params)

    # Short training run
    for epoch in range(n_eval_epochs):
        for batch in spatial_minibatch_sampler(data, batch_size=2048):
            training_step(model, batch, optimizer, **params)

    # Evaluate on held-out spatial fold
    cv_r2 = evaluate_spatial_cv(model, data, domain_config)
    return -cv_r2   # minimize negative R²

def run_cma_es(model_class, data, domain_config):
    import cma

    x0 = [np.mean(bounds) for bounds in cma_search_space.values()]
    sigma0 = 0.5   # initial step size

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': 20,
        'bounds': [
            [b[0] for b in cma_search_space.values()],
            [b[1] for b in cma_search_space.values()]
        ],
        'maxiter': 50,
        'tolx': 1e-4,       # stop when step size < 1e-4
        'tolfun': 1e-3,     # stop when function value change < 1e-3
        'verbose': -9       # suppress output during parallel eval
    })

    # Parallel population evaluation
    from concurrent.futures import ProcessPoolExecutor

    while not es.stop():
        candidates = es.ask()
        with ProcessPoolExecutor(max_workers=4) as executor:
            fitnesses = list(executor.map(
                lambda c: cma_es_objective(c, model_class, data, domain_config),
                candidates
            ))
        es.tell(candidates, fitnesses)
        es.disp()

    best_params = {
        k: np.exp(v) if 'log_' in k else v
        for k, v in zip(cma_search_space.keys(), es.result.xbest)
    }
    return best_params

# CMA-ES runs once per domain before main training
# Output: optimal hyperparameter configuration saved to domain_config
# Main training then uses these fixed hyperparameters
```

**CMA-ES also used for:**

- Process rate network prior initialization: which prior_mean minimizes early training instability across the domain's land cover distribution
- MC³ initialization strategy: which starting DAG structure leads to fastest chain convergence for this domain's variable set
- NUTS step size initialization: starting step size that achieves target acceptance rate quickly, reducing dual averaging burn-in

---

## 1.7 — Stochastic Gradient Descent via Spatial Mini-Batches

Full-batch gradient descent over 54k points is computationally expensive and can get stuck in large flat regions of the loss surface. Mini-batch SGD introduces stochastic noise that helps escape local optima and reduces per-step computation.

**The critical constraint for SPARC:** Each mini-batch must contain a point's spatial neighbors or the neighbor index lookup in the physics loss fails. Random sampling cannot be used.

```python
def spatial_minibatch_sampler(data, coords, neighbor_idx, batch_size=2048):
    """
    Generates spatially contiguous mini-batches.
    Each batch: sample centroid, include all points within radius r
    where the resulting batch_size ≈ target batch_size.

    Ensures: every point's N/S/E/W neighbors are in the same batch.
    Avoids: neighbor lookups crossing batch boundaries (invalid physics loss).
    """
    N = len(coords)

    while True:
        # Sample centroid
        centroid_idx = np.random.randint(0, N)
        centroid = coords[centroid_idx]

        # Find all points within radius
        distances = np.linalg.norm(coords - centroid, axis=1)

        # Binary search for radius that gives target batch size
        radii = np.sort(distances)
        target_idx = min(batch_size, N) - 1
        radius = radii[target_idx]

        batch_idx = np.where(distances <= radius)[0]

        # Verify all neighbors present
        neighbor_set = set(batch_idx)
        valid_in_batch = np.array([
            all(neighbor_idx[i, k] == -1 or neighbor_idx[i, k] in neighbor_set
                for k in range(4))
            for i in batch_idx
        ])

        # Use only points whose neighbors are all in batch
        clean_batch = batch_idx[valid_in_batch]

        if len(clean_batch) >= batch_size // 2:
            yield clean_batch
```

**Batch size selection:** The capacity sweep (Section 1.9) will identify your interpolation threshold in terms of hidden_dim. Batch size interacts with this — larger batches provide better gradient estimates but less stochastic noise to escape local optima. For SPARC domains with ~50k points, `batch_size=2048` (approximately 4% of data per step) provides a good noise/accuracy tradeoff. Increase to 4096 for domains with >200k points.

---

## 1.8 — Stochastic Weight Averaging

After standard training converges, SWA finds flat optima that generalize better to unseen spatial locations. The intuition: sharp optima in the training loss landscape often correspond to narrow basins that do not generalize spatially. Flat optima — the broad valleys between sharp peaks — tend to generalize better because they are robust to the distribution shift between training and test spatial folds.

```python
def apply_swa(model, process_rate_net, train_loader, optimizer,
              swa_epochs=20, swa_lr_min=1e-4, swa_lr_max=1e-2):
    """
    Cyclic learning rate during SWA causes model to traverse
    the flat optimum region. Weight averaging produces a point
    near the center of this region — best generalization.
    """
    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

    swa_model = AveragedModel(model)
    swa_process_rate = AveragedModel(process_rate_net)

    # Cyclic scheduler for SWA phase
    swa_scheduler = SWALR(
        optimizer,
        swa_lr=swa_lr_max,
        anneal_epochs=swa_epochs // 2,
        anneal_strategy='cos'
    )

    for epoch in range(swa_epochs):
        for batch in train_loader:
            training_step(model, process_rate_net, batch, optimizer,
                          clip_norm=1.0)

        # Update SWA model averages
        swa_model.update_parameters(model)
        swa_process_rate.update_parameters(process_rate_net)
        swa_scheduler.step()

    # Update batch statistics using training data
    # Required for LayerNorm (and BatchNorm if present)
    with torch.no_grad():
        for batch in train_loader:
            swa_model(**batch)

    return swa_model, swa_process_rate
    # Typical improvement: +0.5 to +1.5 R² at no architectural cost
```

---

## 1.9 — Hidden Dimension Capacity Sweep (Double Descent)

Before fixing `hidden_dim`, run a sweep to find the double descent regime. The interpolation threshold — where test R² is worst — should be identified and avoided. Optimal hidden_dim lies beyond this threshold where the model is overparameterized and the physics loss acts as implicit regularization.

```python
def capacity_sweep(model_class, data, domain_config, hidden_dims=None):
    """
    Trains models at each hidden_dim, records spatial CV R².
    Expected shape: underfit → interpolation dip → second descent.
    Set hidden_dim at value on the right side of the dip.
    """
    if hidden_dims is None:
        hidden_dims = [64, 128, 256, 512, 1024]

    results = {}
    for dim in hidden_dims:
        model = model_class(hidden_dim=dim, **domain_config)
        optimizer = build_optimizer(model)

        # Short training run for sweep — full training after selection
        for epoch in range(30):
            for batch in spatial_minibatch_sampler(data, batch_size=2048):
                training_step(model, batch, optimizer)

        cv_r2 = evaluate_spatial_cv(model, data)
        results[dim] = cv_r2
        print(f"hidden_dim={dim}: CV R²={cv_r2:.4f}")

    # Find interpolation threshold (minimum R²)
    threshold_dim = min(results, key=results.get)
    optimal_dim   = max(
        (dim for dim in hidden_dims if dim > threshold_dim),
        key=results.get
    )
    print(f"Interpolation threshold at hidden_dim={threshold_dim}")
    print(f"Selected hidden_dim={optimal_dim} (second descent regime)")

    return optimal_dim, results

# Output: capacity_sweep.png saved to stage2/figures/
# Shows hidden_dim vs CV R² — paper-quality diagnostic figure
```

---

## 1.10 — Domain-Specific Optimization Recommendations

Different domains present different optimization challenges based on their data distributions, spatial autocorrelation structures, and physical process characteristics.

| Domain | Primary challenge | Recommended adjustment |
|---|---|---|
| UHI | High spatial autocorrelation, non-stationary coefficients | batch_size=2048, clip_norm=1.0, standard schedule |
| ForceSMIP | Global grid, weak spatial gradient, large N | batch_size=4096, lower lambda_physics (0.05), longer warmup (20 epochs) |
| Coastal | Sharp gradients at coastline, non-stationary process rate | clip_norm=0.5, higher lambda_smooth (0.05), SIREN omega=15 |
| Wildfire | Highly skewed outcome distribution, extreme events | Add Huber loss component, clip_norm=0.5, consider SGD+Nesterov |
| Groundwater | Smooth spatial variation, well-behaved gradients | Standard settings, increase hidden_dim to upper double descent regime |
| Air quality | Complex wind-driven dispersion, directional anisotropy | Larger max_neighbors (256) in spatial attention, longer burnin |
| Drought | Temporal lagging, KBDI cumulative variable | Preprocessing: standardize KBDI by domain; standard optimizer |

---

---

# PART TWO — STAGE 2 ARCHITECTURE

---

## 2.1 — Differentiable Base Model Surrogates

**File:** `sparc/models/surrogates.py`

Three differentiable neural approximations of GWR, GWRF, and GGPGAM. Each surrogate is trained simultaneously with the meta-learner under the joint objective. Interpretability of each original model is preserved through architectural design — coefficient surfaces remain mappable, local nonlinearity remains characterizable, additive structure remains decomposable.

### Differentiable GWR

```python
class DifferentiableGWR(nn.Module):
    """
    Approximates GWR by learning spatially-varying coefficients
    as a function of location.

    Interpretability: beta (N, n_vars) maps directly to GWR coefficient
    surfaces. Map beta[:, k] across study area = spatially-varying
    coefficient for predictor k. Identical interpretation to true GWR.
    Correlation with true GWR output required > 0.95 (validation).
    """
    def __init__(self, n_vars, n_spatial_features, hidden_dim=64):
        super().__init__()
        self.coeff_net = nn.Sequential(
            nn.Linear(n_spatial_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_vars)
        )
        # Initialize near zero — start from GWR-like flat surface
        nn.init.xavier_uniform_(self.coeff_net[-1].weight, gain=0.01)

    def forward(self, X, spatial_features):
        beta = self.coeff_net(spatial_features)    # (N, n_vars)
        y_pred = (X * beta).sum(dim=1, keepdim=True)  # (N, 1)
        return y_pred, beta

    def coefficient_map(self, spatial_features, predictor_idx):
        """Returns spatially-varying coefficient for one predictor."""
        with torch.no_grad():
            beta = self.coeff_net(spatial_features)
        return beta[:, predictor_idx].numpy()
```

### Differentiable GWRF

```python
class DifferentiableGWRF(nn.Module):
    """
    Approximates GWRF using spatially-conditioned nonlinear network.
    Local nonlinearity: each location gets a different nonlinear
    transformation of the same input features.
    Interpretability: spatial modulation vector characterizes local
    nonlinear behavior — can be projected into interpretable components.
    """
    def __init__(self, n_vars, n_spatial_features, hidden_dim=64):
        super().__init__()
        # Spatial conditioning: location → modulation vector
        self.spatial_conditioner = nn.Sequential(
            nn.Linear(n_spatial_features, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, hidden_dim)
        )
        # Prediction: features + spatial modulation → outcome
        self.predictor = nn.Sequential(
            nn.Linear(n_vars + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, X, spatial_features):
        spatial_mod = self.spatial_conditioner(spatial_features)
        combined = torch.cat([X, spatial_mod], dim=1)
        return self.predictor(combined)
```

### Differentiable GGPGAM

```python
class DifferentiableGGPGAM(nn.Module):
    """
    Approximates GGPGAM using sum of smooth per-predictor networks.
    Interpretability: each shape function f_k is plottable as a
    univariate smooth — identical interpretation to GAM partial effects.
    Additive structure preserved: total = sum of per-predictor effects.
    """
    def __init__(self, n_vars, hidden_dim=32):
        super().__init__()
        # One small network per predictor — additive structure guaranteed
        self.shape_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(n_vars)
        ])
        self.intercept = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        effects = [f(X[:, k:k+1]) for k, f in enumerate(self.shape_functions)]
        return sum(effects) + self.intercept

    def partial_effect(self, X, predictor_idx, n_grid=100):
        """Returns partial effect curve for one predictor."""
        x_range = torch.linspace(X[:, predictor_idx].min(),
                                 X[:, predictor_idx].max(), n_grid).unsqueeze(1)
        with torch.no_grad():
            effect = self.shape_functions[predictor_idx](x_range)
        return x_range.numpy(), effect.numpy()
```

### Surrogate Validation Protocol

Before using surrogates in joint training, validate against true base model outputs.

```python
def validate_surrogates(surrogates, true_base_models, data, threshold=0.95):
    """
    Each surrogate must achieve R² > threshold vs. true model predictions.
    If failed: increase hidden_dim, increase lambda_base, retrain.
    True model retained as fallback if surrogate consistently fails.
    """
    results = {}
    for name, (surrogate, true_model) in zip(
        ['gwr', 'gwrf', 'ggpgam'],
        zip(surrogates.values(), true_base_models.values())
    ):
        with torch.no_grad():
            true_pred   = true_model(data['X'], data.get('spatial'))
            surr_pred   = surrogate(data['X'], data.get('spatial'))

        r2 = 1 - ((true_pred - surr_pred)**2).sum() / ((true_pred - true_pred.mean())**2).sum()
        results[name] = r2.item()

        status = 'PASS' if r2 > threshold else 'FAIL'
        print(f"Surrogate {name}: R²={r2:.4f} vs true model [{status}]")

    return results
```

---

## 2.2 — Process Rate Network

**File:** `sparc/models/process_rate_net.py`

Domain-agnostic auxiliary network estimating the spatially-varying physical process rate. For UHI: thermal diffusivity α (m²/s). For coastal: hydraulic diffusivity κ. For wildfire: fire spread rate λ. Same architecture, same training curriculum, same loss structure across all domains — domain config supplies the physical interpretation, bounds, and material priors.

```python
class ProcessRateNet(nn.Module):
    """
    Learns the mapping: land cover composition → local process rate.

    Physical bounds hard-enforced via sigmoid scaling — no clamping,
    no gradient killing, always physically valid output.

    Initialized near domain prior mean — starts from physics,
    deviates where data supports it. Deviation from prior is a
    scientific finding, not a training artifact.
    """
    def __init__(self, n_inputs, domain_config):
        super().__init__()
        pr = domain_config['process_rate']
        self.log_min  = np.log(pr['bounds'][0])
        self.log_max  = np.log(pr['bounds'][1])
        self.var_name = pr['name']    # 'thermal_diffusivity', etc.
        self.units    = pr['units']

        self.network = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

        # Initialize near domain prior mean — critical for stable early training
        nn.init.constant_(self.network[-1].bias, np.log(pr['prior_mean']))
        nn.init.xavier_uniform_(self.network[-1].weight, gain=0.1)

    def forward(self, land_cover_features):
        raw = self.network(land_cover_features)

        # Sigmoid hard-constrains to physical bounds
        # Smooth at boundaries — no gradient killing
        log_rate = (
            self.log_min +
            (self.log_max - self.log_min) * torch.sigmoid(raw)
        )
        return torch.exp(log_rate)    # (N, 1) always positive, always in bounds

    def compute_mixture_prior(self, land_cover, material_table):
        """
        Linear mixture of material property values weighted by surface fractions.
        Physically motivated prior — same approach as urban energy balance models.
        Network can deviate from this prior but deviation costs lambda_prior.
        """
        prior = torch.zeros(len(land_cover), 1)
        fraction_sum = torch.zeros(len(land_cover), 1)

        for material, alpha_value in material_table.items():
            col_idx = self.material_column_indices[material]
            fraction = land_cover[:, col_idx:col_idx+1] / 100.0
            prior += fraction * alpha_value
            fraction_sum += fraction

        # Normalize by actual fraction sum (handles pixels not summing to 100%)
        prior = prior / fraction_sum.clamp(min=1e-6)
        return prior

    def material_validation_report(self, land_cover, land_cover_classes,
                                   literature_values):
        """
        Compares learned process rate to published material property values.
        Primary validation result for Paper 1.
        Agreement near 1:1 line = validation.
        Systematic deviation = scientific finding.
        """
        with torch.no_grad():
            learned = self.forward(land_cover).numpy()

        report = {}
        for class_name, class_mask in land_cover_classes.items():
            learned_mean = learned[class_mask].mean()
            learned_std  = learned[class_mask].std()
            lit_value    = literature_values.get(class_name, np.nan)

            report[class_name] = {
                'learned_mean': learned_mean,
                'learned_std':  learned_std,
                'literature':   lit_value,
                'difference':   learned_mean - lit_value,
                'z_score':      (learned_mean - lit_value) / (learned_std + 1e-10)
            }
        return report
```

**Domain configuration for all 13 templates:**

```yaml
# UHI domain
process_rate:
  enabled: true
  name: thermal_diffusivity
  units: m2_per_s
  bounds: [1.0e-7, 9.0e-7]
  prior_mean: 5.0e-7
  inputs: [Pct_Impervious, Pct_Canopy, Pct_Water, NDVI, soil_moisture]
  material_priors:
    impervious: 7.5e-7
    canopy:     3.0e-7
    water:      1.4e-7
    soil:       3.5e-7

# Coastal domain
process_rate:
  enabled: true
  name: hydraulic_diffusivity
  units: m2_per_s
  bounds: [1.0e-5, 1.0e-2]
  prior_mean: 1.0e-3
  inputs: [substrate_type, porosity, grain_size, tidal_range, elevation]
  material_priors:
    sand:   1.0e-3
    clay:   1.0e-5
    gravel: 1.0e-2
    rock:   1.0e-6

# Wildfire domain
process_rate:
  enabled: true
  name: fire_spread_rate
  units: m_per_s
  bounds: [1.0e-4, 5.0e-1]
  prior_mean: 1.0e-2
  inputs: [fuel_moisture, slope, aspect, wind_speed, fuel_type]
  material_priors:
    chaparral: 0.05
    grassland: 0.15
    forest:    0.03
    bare:      0.001

# Air quality domain
process_rate:
  enabled: true
  name: turbulent_diffusivity
  units: m2_per_s
  bounds: [1.0, 100.0]
  prior_mean: 10.0
  inputs: [wind_speed, boundary_layer_height, urban_roughness, stability_class]
  material_priors:
    urban_dense:  5.0
    urban_sparse: 15.0
    suburban:     25.0
    rural:        50.0
```

---

## 2.3 — Sinusoidal Spatial Encoding

**File:** `sparc/features/sinusoidal_encoding.py`

Replaces Laplacian eigenmaps. Faster to compute (no eigendecomposition), Fourier-interpretable, naturally captures spatial periodicity at multiple scales.

```python
class SinusoidalSpatialEncoding(nn.Module):
    """
    Encodes (x, y) coordinates as sinusoidal features at multiple frequencies.
    Related to the Fourier transform of the spatial field — each frequency
    captures periodicity at a different spatial scale.

    Low frequencies: broad regional patterns (climate gradients)
    High frequencies: fine-scale local patterns (urban morphology)

    This is the spatial equivalent of transformer positional encoding.
    """
    def __init__(self, n_frequencies=64, learnable_freqs=False):
        super().__init__()
        self.n_frequencies = n_frequencies

        if learnable_freqs:
            # Allow frequencies to adapt to domain spatial structure
            self.log_freqs = nn.Parameter(
                torch.linspace(0, np.log(10000), n_frequencies // 2)
            )
        else:
            self.register_buffer(
                'log_freqs',
                torch.linspace(0, np.log(10000), n_frequencies // 2)
            )

    def forward(self, coords):
        """
        coords: (N, 2) array of [x, y] coordinates in projected CRS
        Returns: (N, 4 * n_frequencies // 2) encoding
        """
        freqs = torch.exp(self.log_freqs)   # (n_frequencies // 2,)

        # Normalize coordinates to [0, 2π] range for stability
        x = coords[:, 0:1]
        y = coords[:, 1:2]

        encoding = torch.cat([
            torch.sin(x * freqs),
            torch.cos(x * freqs),
            torch.sin(y * freqs),
            torch.cos(y * freqs),
        ], dim=1)

        return encoding    # (N, 4 * n_frequencies // 2) = (N, 128) at default
```

---

## 2.4 — Sparse Spatial Attention

**File:** `sparc/models/spatial_attention.py`

Replaces fixed spatial encoder with learned attention over geographic neighborhood. Each point learns which other points are relevant — not just geographic proximity but feature similarity. This is a data-driven GWR kernel.

Full attention over 54k points is O(N²) — infeasible. Sparse KNN attention limits each point to attending over its max_neighbors nearest points — O(N × max_neighbors).

```python
class SIRENLayer(nn.Module):
    """
    Sinusoidal activation layer optimal for PDE solutions.
    Used exclusively in physics encoder stream.
    Gradient management: LayerNorm before activation + SIREN init.
    """
    def __init__(self, in_dim, out_dim, omega=30.0, is_first=False):
        super().__init__()
        self.linear     = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.omega      = omega

        if is_first:
            nn.init.uniform_(self.linear.weight, -1/in_dim, 1/in_dim)
        else:
            bound = np.sqrt(6 / in_dim) / omega
            nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.layer_norm(x)
        return torch.sin(self.omega * self.linear(x))


class SparseSpatialAttention(nn.Module):
    """
    Multi-head attention over KNN spatial neighborhood.
    Attention weights: interpretable as spatial influence surface.
    Each point's attention weight distribution shows which other
    points influenced its prediction and by how much.
    """
    def __init__(self, d_model, n_heads=4, max_neighbors=128, dropout=0.1):
        super().__init__()
        self.d_model       = d_model
        self.n_heads       = n_heads
        self.max_neighbors = max_neighbors

        self.attention = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.pos_encoding = SinusoidalSpatialEncoding(d_model // 2)
        self.output_norm  = nn.LayerNorm(d_model)

    def build_knn_index(self, coords, k=None):
        """
        Precomputed once per dataset. Reused every forward pass.
        Returns (N, max_neighbors) array of neighbor indices.
        """
        from sklearn.neighbors import NearestNeighbors
        k = k or self.max_neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nbrs.fit(coords)
        _, indices = nbrs.kneighbors(coords)
        return indices[:, 1:]   # exclude self

    def forward(self, X, coords, knn_index):
        """
        X:         (N, d_model) feature vectors
        coords:    (N, 2) geographic coordinates
        knn_index: (N, max_neighbors) precomputed neighbor indices
        """
        # Inject geographic position
        pos = self.pos_encoding(coords)
        X_pos = X + pos[:, :self.d_model]   # trim to d_model if needed

        # Sparse attention: each point attends to its KNN neighborhood
        # X_pos[knn_index]: (N, max_neighbors, d_model)
        neighbors = X_pos[knn_index]   # (N, max_neighbors, d_model)

        # Reshape for MultiheadAttention: (N, 1, d_model) queries
        queries  = X_pos.unsqueeze(1)    # (N, 1, d_model)

        attended, weights = self.attention(
            queries,        # (N, 1, d_model)
            neighbors,      # (N, max_neighbors, d_model)
            neighbors       # (N, max_neighbors, d_model)
        )

        # weights: (N, 1, max_neighbors) → interpretable influence surface
        attended = self.output_norm(attended.squeeze(1) + X_pos)

        return attended, weights.squeeze(1)

    def attention_influence_map(self, weights, knn_index, focus_point_idx, N):
        """
        For a given point: which other points influenced its prediction?
        Returns a spatial influence surface of shape (N,).
        High values = strong influence on focus point's prediction.
        """
        influence = torch.zeros(N)
        neighbor_indices = knn_index[focus_point_idx]
        influence[neighbor_indices] = weights[focus_point_idx].mean(dim=0).cpu()
        return influence.numpy()
```

---

## 2.5 — Neural Meta-Learner

**File:** `sparc/models/neural_meta.py`

Four-stream fusion network with SIREN physics encoder, sparse spatial attention, process rate embedding, and dual prediction head.

```python
class SPARCMetaLearner(nn.Module):
    """
    Four-stream neural meta-learner.

    Stream 1 (Base):    Encodes differentiable surrogate predictions
    Stream 2 (Physics): SIREN encoder — natural PDE solution basis
    Stream 3 (Spatial): Sparse attention — learned spatial influence
    Stream 4 (Alpha):   Process rate embedding — local thermal regime

    Dual output:
        Regression head:   continuous outcome prediction
        Exceedance heads:  P(outcome > threshold) per configured threshold
                           Trained directly via cross-entropy —
                           calibrated exceedance probabilities as primary output

    NUTS interface:
        predict_for_nuts() — called by NUTS likelihood evaluation
        Exposes prediction function without gradient computation
    """
    def __init__(self, n_base_models, n_physics_features,
                 d_spatial, hidden_dim, dropout, thresholds,
                 n_heads=4, max_neighbors=128):
        super().__init__()

        # Stream 1: Base model predictions
        self.base_enc = nn.Sequential(
            nn.Linear(n_base_models, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )

        # Stream 2: Physics features — SIREN for PDE basis
        self.physics_enc = nn.Sequential(
            SIRENLayer(n_physics_features, 32, omega=30.0, is_first=True),
            SIRENLayer(32, 32, omega=30.0)
        )

        # Stream 3: Spatial context — learned attention
        self.spatial_enc = SparseSpatialAttention(
            d_model=d_spatial,
            n_heads=n_heads,
            max_neighbors=max_neighbors,
            dropout=dropout
        )
        # Project attention output to standard width
        self.spatial_proj = nn.Linear(d_spatial, 64)

        # Stream 4: Process rate embedding
        self.alpha_emb = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU()
        )

        # Fusion
        fusion_in = 32 + 32 + 64 + 16   # = 144
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU()
        )

        # Dual prediction heads
        self.regression_head  = nn.Linear(64, 1)
        self.exceedance_heads = nn.ModuleList([
            nn.Linear(64, 1) for _ in thresholds
        ])
        self.thresholds = thresholds

    def forward(self, base_preds, physics_feats, X_spatial, coords,
                knn_index, alpha):
        """
        base_preds:    (N, n_base_models) — DiffGWR, DiffGWRF, DiffGGPGAM outputs
        physics_feats: (N, n_physics)
        X_spatial:     (N, d_spatial)
        coords:        (N, 2)
        knn_index:     (N, max_neighbors) precomputed
        alpha:         (N, 1) from ProcessRateNet
        """
        b = self.base_enc(base_preds)

        p = self.physics_enc(physics_feats)

        s_attended, attn_weights = self.spatial_enc(X_spatial, coords, knn_index)
        s = self.spatial_proj(s_attended)

        a = self.alpha_emb(alpha)

        h = self.fusion(torch.cat([b, p, s, a], dim=1))

        T_pred   = self.regression_head(h)
        exceedance = [
            torch.sigmoid(head(h))
            for head in self.exceedance_heads
        ]

        return T_pred, exceedance, attn_weights

    def predict_with_uncertainty(self, *args, n_samples=100, **kwargs):
        """MC Dropout inference — keep dropout active."""
        self.train()
        predictions = torch.stack([
            self.forward(*args, **kwargs)[0]
            for _ in range(n_samples)
        ])
        return predictions.mean(dim=0), predictions.std(dim=0)

    def predict_for_nuts(self, X_dict):
        """Interface for NUTS likelihood evaluation — no gradient."""
        with torch.no_grad():
            T_pred, *_ = self.forward(**X_dict)
        return T_pred.numpy()
```

---

## 2.6 — Joint Loss Function

**File:** `sparc/training/loss.py`

Single unified objective across all components. Eight loss terms organized by function.

```python
def sparc_joint_loss(
    # Predictions
    T_pred, exceedance_preds, y_true, thresholds,
    # Process rate
    alpha, alpha_prior,
    # Spatial structure
    neighbor_idx, source_term, resolution,
    # Surrogate alignment
    surrogate_preds, surrogate_targets,
    # Lambda weights (from CMA-ES)
    lambda_physics, lambda_smooth,
    lambda_alpha_smooth, lambda_prior,
    lambda_base, lambda_neighbor,
    # Training phase (for curriculum)
    epoch
):
    loss_components = {}

    # 1. Data fidelity — regression MSE
    mse = F.mse_loss(T_pred.squeeze(), y_true)
    loss_components['mse'] = mse.item()

    # 2. Exceedance probabilities — cross entropy per threshold
    # Direct training of P(outcome > threshold)
    ce = sum(
        F.binary_cross_entropy(
            e.squeeze(),
            (y_true > t).float()
        )
        for e, t in zip(exceedance_preds, thresholds)
    ) * 0.5
    loss_components['cross_entropy'] = ce.item()

    # 3. Physics residual — heat diffusion PDE (or domain equivalent)
    valid = (neighbor_idx != -1).all(dim=1)
    T = T_pred.squeeze()
    T_n = T[neighbor_idx[valid, 0]]
    T_s = T[neighbor_idx[valid, 1]]
    T_e = T[neighbor_idx[valid, 2]]
    T_w = T[neighbor_idx[valid, 3]]

    laplacian = (T_n + T_s + T_e + T_w - 4*T[valid]) / (resolution**2)
    physics_residual = laplacian - (source_term[valid] / alpha.squeeze()[valid])

    # Normalize residual to prevent physics loss dominating
    residual_scale = physics_residual.detach().std().clamp(min=1e-6)
    physics = lambda_physics * (physics_residual / residual_scale).pow(2).mean()
    loss_components['physics'] = physics.item()

    # 4. Prediction smoothness — penalize implausible spatial spikes
    smooth = lambda_smooth * laplacian.pow(2).mean()
    loss_components['smooth'] = smooth.item()

    # 5. Alpha field smoothness — process rate cannot jump discontinuously
    if valid.sum() > 0:
        a = alpha.squeeze()
        a_n = a[neighbor_idx[valid, 0]]
        a_s = a[neighbor_idx[valid, 1]]
        a_e = a[neighbor_idx[valid, 2]]
        a_w = a[neighbor_idx[valid, 3]]
        alpha_lap = (a_n + a_s + a_e + a_w - 4*a[valid]) / (resolution**2)
        alpha_smooth = lambda_alpha_smooth * alpha_lap.pow(2).mean()
    else:
        alpha_smooth = torch.tensor(0.0)
    loss_components['alpha_smooth'] = alpha_smooth.item()

    # 6. Alpha prior regularization — decayed by curriculum
    prior_weight = get_prior_weight(epoch, lambda_prior)
    prior_reg = prior_weight * (alpha - alpha_prior).pow(2).mean()
    loss_components['alpha_prior'] = prior_reg.item()

    # 7. Surrogate fidelity — each surrogate tracks true base model
    surrogate_loss = lambda_base * sum(
        F.mse_loss(pred, target)
        for pred, target in zip(surrogate_preds, surrogate_targets)
    )
    loss_components['surrogate'] = surrogate_loss.item()

    # 8. Spatial neighborhood consistency
    point_loss = (T_pred.squeeze() - y_true).pow(2)
    neighbor_loss = torch.zeros_like(point_loss)
    for k in range(4):
        valid_k = neighbor_idx[:, k] != -1
        neighbor_loss[valid_k] += 0.25 * point_loss[neighbor_idx[valid_k, k]]
    neighborhood = lambda_neighbor * (point_loss + 0.3 * neighbor_loss).mean()
    loss_components['neighborhood'] = neighborhood.item()

    total = mse + ce + physics + smooth + alpha_smooth + prior_reg + surrogate_loss + neighborhood
    loss_components['total'] = total.item()

    return total, loss_components


def get_prior_weight(epoch, lambda_prior_base):
    """
    Decay prior regularization over training.
    Early: prior guides alpha near mixture model.
    Late:  data trusted to pull alpha away from prior.
    """
    if epoch < 10:
        return lambda_prior_base
    elif epoch < 30:
        return lambda_prior_base
    else:
        # Decay toward 10% of base weight
        decay = max(0.1, 1.0 - (epoch - 30) / 100)
        return lambda_prior_base * decay
```

---

## 2.7 — Training Curriculum

**File:** `sparc/training/curriculum.py`

Four-stage training schedule. Each stage has a specific purpose — rushing to full joint optimization without proper initialization leads to unstable training or local minima trapping.

```
STAGE A — Representation Warmup (epochs 0–10)
    lambda_physics      = 0
    lambda_prior        = 0.01
    lambda_base         = 0.5     ← surrogates track true base models closely
    lambda_neighbor     = 0.01
    lambda_smooth       = 0
    lambda_alpha_smooth = 0

    CMA-ES runs during this stage (parallel process)
    Goal: stable initialization. Surrogates converge to base models.
          Alpha network initializes near mixture prior.
          Attention patterns emerge from spatial data.
          SIREN weights settle into a valid harmonic.

STAGE B — Physics Activation (epochs 10–30)
    lambda_physics      = ramp(0 → target)
    lambda_alpha_smooth = ramp(0 → target)
    lambda_prior        = 0.01
    lambda_base         = 0.2     ← surrogates allowed moderate deviation
    lambda_smooth       = ramp(0 → target)

    Nesterov momentum builds velocity through consistent gradient directions.
    Gradient clipping active (max_norm from CMA-ES, typically 0.5–2.0).
    Goal: physics constraint activates. Alpha begins differentiating
          by surface type. Exceedance heads calibrate.

STAGE C — Joint Optimization (epoch 30+)
    All lambdas at CMA-ES-optimized values.
    lambda_prior decays toward 10% of base (curriculum decay).
    lambda_base  = 0.1    ← surrogates semi-independent, data-driven.
    Cosine annealing with warm restarts cycles lr.
    Goal: end-to-end joint optimization under unified physics objective.
          Gradient flows from PDE loss through all components.

STAGE D — Stochastic Weight Averaging (final 20% of epochs)
    Cyclic lr between swa_lr_min and swa_lr_max (from CMA-ES).
    Weight averaging across cycles.
    Goal: flat optimum. Better spatial generalization.
          Typically +0.5–1.5 R² at no architectural cost.
```

```python
def get_lambda_schedule(epoch, target_lambdas, warmup_end=10, ramp_end=30):
    """Returns current lambda values for each loss term."""
    schedule = {}

    for term, target in target_lambdas.items():
        if term in ['physics', 'smooth', 'alpha_smooth']:
            # Ramp from 0 over warmup + ramp period
            if epoch < warmup_end:
                schedule[term] = 0.0
            elif epoch < ramp_end:
                t = (epoch - warmup_end) / (ramp_end - warmup_end)
                schedule[term] = target * t
            else:
                schedule[term] = target

        elif term == 'prior':
            # Constant then decay
            schedule[term] = get_prior_weight(epoch, target)

        elif term == 'base':
            # Step down at stage transitions
            if epoch < warmup_end:
                schedule[term] = 0.5
            elif epoch < ramp_end:
                schedule[term] = 0.2
            else:
                schedule[term] = 0.1

        else:
            schedule[term] = target

    return schedule
```

---

## 2.8 — Multi-Resolution Transfer Learning

**File:** `sparc/models/downscaler.py`

Progressive training across spatial resolutions. The causal DAG structure and physics priors transfer across resolutions. The same causal mechanisms operative at global scale (ForceSMIP) are also operative at urban scale (UHI) — physics priors enforce this consistency.

```python
class SpatialTransferLearner(nn.Module):
    """
    Transfers learned representations from coarser resolution
    to finer resolution via frozen backbone + trainable refinement.

    Resolution levels:
        Level 1: ForceSMIP 2.5° global (ERA5)
        Level 2: Regional 1km (regional climate models)
        Level 3: Urban 100m (satellite + reanalysis fusion)
        Level 4: Local 30m (high-resolution remote sensing)

    Transfer strategy:
        Freeze backbone encoder (coarse representations)
        Fine-tune: resolution adapter + refinement head
        Physics priors held constant across resolutions
        Causal structure (MC³ DAG) frozen from Level 1
    """
    def __init__(self, backbone, fine_features_dim, hidden_dim=64):
        super().__init__()
        # Frozen backbone from coarser resolution
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Adapter: maps coarse representation to target resolution context
        self.resolution_adapter = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )

        # Refinement: combines coarse representation with fine features
        self.refinement = nn.Sequential(
            nn.Linear(128 + fine_features_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, X_coarse, X_fine, coords_coarse, coords_fine, knn_index):
        # Extract coarse representation (no gradient — frozen)
        with torch.no_grad():
            coarse_repr = self.backbone.encode(X_coarse, coords_coarse, knn_index)

        # Adapt to target resolution
        adapted = self.resolution_adapter(coarse_repr)

        # Refine with fine-resolution features
        combined = torch.cat([adapted, X_fine], dim=1)
        return self.refinement(combined)


# Training schedule across resolution levels
resolution_levels = {
    1: {
        'domain':      'forcesmip',
        'resolution':  250000,       # 2.5° in meters
        'data_source': 'ERA5 global',
        'freeze':      [],           # nothing frozen at Level 1
        'description': 'Global causal structure + climate forcing patterns'
    },
    2: {
        'domain':      'regional',
        'resolution':  1000,
        'data_source': 'Regional climate models',
        'freeze':      ['mc3_dag'],  # freeze DAG from Level 1
        'description': 'Regional coefficient adjustment, mesoscale patterns'
    },
    3: {
        'domain':      'uhi',
        'resolution':  100,
        'data_source': 'Satellite + reanalysis',
        'freeze':      ['mc3_dag', 'backbone_encoder'],
        'description': 'Local land cover effects, urban morphology corrections'
    },
    4: {
        'domain':      'uhi_hires',
        'resolution':  30,
        'data_source': 'High-resolution remote sensing',
        'freeze':      ['mc3_dag', 'backbone_encoder'],
        'description': 'Fine-scale spatial heterogeneity'
    }
}
```

---

## 2.9 — MAML Spatial Cross-Validation

**File:** `sparc/training/maml.py`

Model-Agnostic Meta-Learning adapted for spatial cross-validation. The model explicitly optimizes for performance on held-out spatial folds — spatial generalization is the objective, not just evaluation.

```python
def maml_spatial_cv_loss(model, process_rate_net, data, spatial_folds,
                          inner_lr=1e-3, n_inner_steps=5, lambda_dict=None):
    """
    Outer loop: optimize base parameters for spatial generalization.
    Inner loop: adapt to each training fold.
    Test fold loss (not training fold) is what drives meta-gradient.

    Effect: model learns initialization from which a few gradient steps
            produce good predictions at any spatial location.
            Spatial generalization built into the loss function.
    """
    from copy import deepcopy

    meta_loss = 0.0

    for train_idx, test_idx in spatial_folds:
        # Clone parameters for inner loop adaptation
        fast_model = deepcopy(model)
        fast_prate = deepcopy(process_rate_net)
        inner_opt = torch.optim.SGD(
            list(fast_model.parameters()) + list(fast_prate.parameters()),
            lr=inner_lr,
            momentum=0.9,
            nesterov=True
        )

        # Inner loop: adapt to this fold's training data
        for _ in range(n_inner_steps):
            train_batch = data[train_idx]
            alpha = fast_prate(train_batch['land_cover'])
            T_pred, exceedance, _ = fast_model(**train_batch, alpha=alpha)

            train_loss, _ = sparc_joint_loss(
                T_pred, exceedance, train_batch['y'],
                alpha=alpha, **lambda_dict
            )
            inner_opt.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(fast_model.parameters()) + list(fast_prate.parameters()),
                max_norm=1.0
            )
            inner_opt.step()

        # Outer loop: evaluate on held-out spatial fold
        # This is what drives the meta-gradient on the BASE model
        test_batch = data[test_idx]
        alpha_test = fast_prate(test_batch['land_cover'])
        T_pred_test, exceedance_test, _ = fast_model(**test_batch, alpha=alpha_test)

        test_loss, _ = sparc_joint_loss(
            T_pred_test, exceedance_test, test_batch['y'],
            alpha=alpha_test, **lambda_dict
        )
        meta_loss += test_loss

    return meta_loss / len(spatial_folds)
```

---

---

# PART THREE — STAGE 3: BAYESIAN CAUSAL INFERENCE

---

## 3.1 — MC³ with Parallel Tempering

**File:** `sparc/causal/mc3.py`

MC³ samples a posterior distribution over DAG structures. Standard parallel chains (4 chains at the same temperature) can get stuck in disconnected posterior modes — particularly for domains where multiple causal structures have similar BGe scores. Parallel tempering runs chains at different temperatures and allows state swaps, enabling exploration of the full graph space.

### DAGStructure

```python
class DAGStructure:
    def __init__(self, adjacency, var_names):
        self.adjacency = adjacency    # (n_vars, n_vars) binary
        self.var_names = var_names
        self.n_vars = len(var_names)

    def get_parents(self, node_idx):
        return list(np.where(self.adjacency[:, node_idx] == 1)[0])

    def is_acyclic(self):
        """Kahn's algorithm O(V+E)."""
        in_degree = self.adjacency.sum(axis=0).copy()
        queue = list(np.where(in_degree == 0)[0])
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for neighbor in np.where(self.adjacency[node] == 1)[0]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return visited == self.n_vars

    def n_edges(self):
        return self.adjacency.sum()

    def copy(self):
        return DAGStructure(self.adjacency.copy(), self.var_names.copy())

    @classmethod
    def empty(cls, domain_config):
        n = len(domain_config['var_names'])
        return cls(np.zeros((n, n), dtype=int), domain_config['var_names'])

    @classmethod
    def from_config(cls, domain_config):
        """Expert DAG from project.yml — used as one initialization."""
        adj = np.zeros((len(domain_config['var_names']),) * 2, dtype=int)
        for (src, tgt) in domain_config['causal']['dag_edges']:
            i = domain_config['var_names'].index(src)
            j = domain_config['var_names'].index(tgt)
            adj[i, j] = 1
        return cls(adj, domain_config['var_names'])
```

### PhysicsInformedGraphPrior

```python
class PhysicsInformedGraphPrior:
    def __init__(self, domain_config):
        self.domain = domain_config
        self.p_edge = domain_config.get('mc3', {}).get('prior_edge_probability', 0.25)

    def log_prior(self, G):
        adj = G.adjacency
        n_vars = G.n_vars
        n_possible = n_vars * (n_vars - 1)

        # 1. Sparsity prior
        n_edges = adj.sum()
        log_p = (n_edges * np.log(self.p_edge) +
                 (n_possible - n_edges) * np.log(1 - self.p_edge))

        # 2. Expected edge bonuses from physics config
        for edge_spec in self.domain.get('physics', {}).get('expected_edges', []):
            i = G.var_names.index(edge_spec['from'])
            j = G.var_names.index(edge_spec['to'])
            strength = edge_spec.get('strength', 1.0)
            if adj[i, j] == 1:
                log_p += strength * 2.0
            else:
                log_p -= strength * 0.5

        # 3. Hard physics violations
        if self._violates_physics(G):
            return -np.inf

        return log_p

    def _violates_physics(self, G):
        adj = G.adjacency
        cfg = self.domain.get('physics', {})
        outcome_idx = G.var_names.index(cfg.get('outcome_var', G.var_names[-1]))

        # Outcome cannot cause anything
        if adj[outcome_idx, :].sum() > 0:
            return True

        # Temporal ordering
        for (early, late) in cfg.get('temporal_order', []):
            ei = G.var_names.index(early)
            li = G.var_names.index(late)
            if adj[li, ei] == 1:
                return True

        # Explicitly forbidden edges
        for (src, tgt) in cfg.get('forbidden_edges', []):
            si = G.var_names.index(src)
            ti = G.var_names.index(tgt)
            if adj[si, ti] == 1:
                return True

        return False
```

### Parallel Tempering MC³

```python
class ParallelTemperingMC3:
    """
    Runs MC³ chains at multiple temperatures simultaneously.
    Hot chains explore freely — cold chains converge to posterior.
    Periodic swap proposals allow hot chain discoveries to
    propagate to the cold (target) chain.
    """
    temperatures = [1.0, 1.5, 2.0, 3.0]   # cold → hot
    swap_interval = 50

    def __init__(self, data, domain_config, n_burnin=3000, n_samples=10000, thin=5):
        self.data = data
        self.domain_config = domain_config
        self.n_burnin = n_burnin
        self.n_samples = n_samples
        self.thin = thin
        self.prior = PhysicsInformedGraphPrior(domain_config)

    def log_posterior(self, G, temperature=1.0):
        lp = self.prior.log_prior(G)
        if not np.isfinite(lp):
            return -np.inf
        ml = self.dag_marginal_likelihood(G)
        # Temperature scales likelihood — hot chains have flatter posteriors
        return lp + ml / temperature

    def swap_step(self, chains, step):
        """Propose swaps between adjacent temperature chains."""
        if step % self.swap_interval != 0:
            return chains

        for i in range(len(chains) - 1):
            # Metropolis-Hastings acceptance for swap
            log_accept = (
                (1/self.temperatures[i] - 1/self.temperatures[i+1]) *
                (chains[i+1]['log_post'] - chains[i]['log_post'])
            )
            if np.log(np.random.uniform()) < log_accept:
                chains[i], chains[i+1] = chains[i+1], chains[i]

        return chains

    def run(self):
        """
        4 temperature chains, each with dispersed initialization.
        Only cold chain (T=1.0) samples collected for posterior.
        """
        inits = [
            DAGStructure.empty(self.domain_config),
            DAGStructure.from_config(self.domain_config),
            DAGStructure.random_sparse(self.domain_config, p=0.2),
            DAGStructure.physics_maximal(self.domain_config)
        ]

        chains = [
            {'G': init, 'log_post': self.log_posterior(init, T), 'T': T}
            for init, T in zip(inits, self.temperatures)
        ]

        posterior_samples = []
        total_steps = self.n_burnin + self.n_samples * self.thin

        for step in range(total_steps):
            # MH step for each chain at its temperature
            for chain in chains:
                chain = self._mh_step(chain)

            # Parallel tempering swaps
            chains = self.swap_step(chains, step)

            # Collect samples from cold chain only (T=1.0)
            cold_chain = chains[0]
            if step >= self.n_burnin and (step - self.n_burnin) % self.thin == 0:
                posterior_samples.append(cold_chain['G'].copy())

        return MC3Results(posterior_samples, self.domain_config)
```

---

## 3.2 — MC³ Results

```python
class MC3Results:
    def posterior_edge_probabilities(self):
        n_vars = self.domain_config['n_vars']
        counts = np.zeros((n_vars, n_vars))
        for G in self.samples:
            counts += G.adjacency
        return counts / len(self.samples)

    def median_probability_dag(self):
        probs = self.posterior_edge_probabilities()
        adj = (probs > 0.5).astype(int)
        G = DAGStructure(adj, self.domain_config['var_names'])
        if not G.is_acyclic():
            adj = self._greedy_acyclicity_repair(adj, probs)
        return DAGStructure(adj, self.domain_config['var_names'])

    def edge_confidence_report(self):
        probs = self.posterior_edge_probabilities()
        report = {
            'strong':     [],   # P > 0.90
            'moderate':   [],   # 0.70–0.90
            'weak':       [],   # 0.50–0.70
            'uncertain':  [],   # 0.30–0.50
            'absent':     [],   # P < 0.30
            'surprising': [],   # P > 0.70, not in expert DAG
            'missing':    []    # P < 0.30, in expert DAG
        }
        expert_adj = DAGStructure.from_config(self.domain_config).adjacency
        for i in range(self.domain_config['n_vars']):
            for j in range(self.domain_config['n_vars']):
                if i == j:
                    continue
                p = probs[i, j]
                edge = (self.domain_config['var_names'][i],
                        self.domain_config['var_names'][j], p)
                if p > 0.90:   report['strong'].append(edge)
                elif p > 0.70: report['moderate'].append(edge)
                elif p > 0.50: report['weak'].append(edge)
                elif p > 0.30: report['uncertain'].append(edge)
                else:          report['absent'].append(edge)

                if p > 0.70 and expert_adj[i, j] == 0:
                    report['surprising'].append(edge)
                if p < 0.30 and expert_adj[i, j] == 1:
                    report['missing'].append(edge)
        return report

    def model_averaged_effects(self, causal_estimator, n_draws=500):
        thin_idx = np.linspace(0, len(self.samples)-1, n_draws, dtype=int)
        coefficient_samples = defaultdict(list)

        for idx in thin_idx:
            G = self.samples[idx]
            coeffs = causal_estimator.estimate(G, self.data)
            for edge, coeff in coeffs.items():
                coefficient_samples[edge].append(coeff)

        averaged = {}
        for edge, samples in coefficient_samples.items():
            samples = np.array(samples)
            n_present = len(samples)
            n_absent  = n_draws - n_present
            with_zeros = np.concatenate([samples, np.zeros(n_absent)])

            averaged[edge] = {
                'bma_mean':           with_zeros.mean(),
                'bma_std':            with_zeros.std(),
                'ci_5':               np.percentile(with_zeros, 5),
                'ci_95':              np.percentile(with_zeros, 95),
                'edge_prob':          n_present / n_draws,
                'effect_if_present':  samples.mean() if len(samples) > 0 else 0.0
            }
        return averaged
```

---

## 3.3 — NUTS Sampler

**File:** `sparc/causal/nuts.py`

No-U-Turn Sampler with dual averaging for step size adaptation. Gradient of log posterior computed via PyTorch autograd — essentially free given existing neural architecture. Conditions on MC³ median probability DAG.

```python
class NUTSSampler:
    """
    No-U-Turn Sampler — adaptive Hamiltonian Monte Carlo.
    Automatically determines trajectory length by detecting
    when the leapfrog path starts turning back on itself.
    Dual averaging adapts step size toward target acceptance rate.
    """
    def __init__(self, log_posterior_fn, target_accept=0.80,
                 n_burnin=1000, n_samples=3000):
        self.log_posterior = log_posterior_fn
        self.target_accept = target_accept
        self.n_burnin = n_burnin
        self.n_samples = n_samples

        # Dual averaging parameters for step size adaptation
        self.mu     = np.log(10 * 0.1)    # initial step size target
        self.gamma  = 0.05
        self.t0     = 10
        self.kappa  = 0.75
        self.delta  = target_accept
        self.step_size = 0.1              # initial step size

    def hamiltonian(self, theta, r):
        return self.log_posterior(theta) - 0.5 * (r**2).sum()

    def leapfrog(self, theta, r, step_size):
        theta = theta.clone().detach().requires_grad_(True)
        log_p = self.log_posterior(theta)
        grad  = torch.autograd.grad(log_p, theta)[0]

        r_half = r + step_size/2 * grad
        theta_new = (theta + step_size * r_half).detach().requires_grad_(True)

        log_p_new = self.log_posterior(theta_new)
        grad_new  = torch.autograd.grad(log_p_new, theta_new)[0]

        r_new = r_half + step_size/2 * grad_new
        return theta_new.detach(), r_new.detach()

    def build_tree(self, theta, r, direction, depth, step_size, H0):
        if depth == 0:
            theta_new, r_new = self.leapfrog(theta, r * direction, step_size)
            H_new = self.hamiltonian(theta_new, r_new)
            n_prime     = int(H_new > H0 - 1000)
            s_prime     = int(H_new > H0 - self.delta_max)
            log_alpha   = min(0.0, H_new - H0)
            return theta_new, r_new, theta_new, r_new, theta_new, n_prime, s_prime, log_alpha, 1

        (theta_minus, r_minus, theta_plus, r_plus,
         theta_prime, n_prime, s_prime, log_alpha, n_alpha) = self.build_tree(
            theta, r, direction, depth-1, step_size, H0
        )

        if s_prime == 1:
            if direction == -1:
                (theta_minus, r_minus, _, _,
                 theta_pp, n_pp, s_pp, la_pp, na_pp) = self.build_tree(
                    theta_minus, r_minus, direction, depth-1, step_size, H0
                )
            else:
                (_, _, theta_plus, r_plus,
                 theta_pp, n_pp, s_pp, la_pp, na_pp) = self.build_tree(
                    theta_plus, r_plus, direction, depth-1, step_size, H0
                )

            # Multinomial sampling for proposal
            if n_pp > 0 and np.random.uniform() < n_pp / max(n_prime + n_pp, 1):
                theta_prime = theta_pp

            n_prime  += n_pp
            log_alpha = np.logaddexp(log_alpha, la_pp)
            n_alpha  += na_pp

            # U-turn criterion
            diff = (theta_plus - theta_minus).detach()
            s_prime = (
                s_pp and
                (diff @ r_minus.detach() >= 0) and
                (diff @ r_plus.detach() >= 0)
            )

        return (theta_minus, r_minus, theta_plus, r_plus,
                theta_prime, n_prime, s_prime, log_alpha, n_alpha)

    def dual_averaging_update(self, step, log_accept_prob, step_size,
                               H_bar, m_bar):
        """
        Adapts step size during burnin toward target acceptance rate.
        Primal-dual averaging — stable convergence of step size.
        """
        t = step + 1
        H_bar_new = ((1 - 1/(t + self.t0)) * H_bar +
                     (1/(t + self.t0)) * (self.delta - np.exp(log_accept_prob)))
        log_e = self.mu - np.sqrt(t)/self.gamma * H_bar_new
        log_e_bar_new = (t**(-self.kappa) * log_e +
                         (1 - t**(-self.kappa)) * np.log(m_bar))

        return np.exp(log_e), H_bar_new, np.exp(log_e_bar_new)

    def sample(self, theta_init):
        theta = theta_init.clone().detach()
        chain = []
        H_bar, m_bar = 0.0, 1.0
        self.delta_max = 1000.0

        for step in range(self.n_burnin + self.n_samples):
            r = torch.randn_like(theta)
            H0 = self.hamiltonian(theta, r).item()

            # NUTS tree building
            theta_minus = theta_plus = theta
            r_minus = r_plus = r
            n, s, depth = 1, 1, 0
            theta_prime = theta

            while s == 1:
                direction = 1 if np.random.uniform() > 0.5 else -1
                (theta_minus, r_minus, theta_plus, r_plus,
                 theta_pp, n_pp, s_pp, log_alpha, n_alpha) = self.build_tree(
                    theta_minus if direction == -1 else theta_plus,
                    r_minus if direction == -1 else r_plus,
                    direction, depth, self.step_size, H0
                )

                if s_pp and np.random.uniform() < min(1.0, n_pp / n):
                    theta_prime = theta_pp

                n += n_pp
                s  = s_pp
                depth += 1

                if depth > 10:   # safety cap on tree depth
                    break

            # Accept/reject
            theta = theta_prime

            # Dual averaging during burnin
            if step < self.n_burnin:
                mean_log_alpha = log_alpha - np.log(max(n_alpha, 1))
                self.step_size, H_bar, m_bar = self.dual_averaging_update(
                    step, mean_log_alpha, self.step_size, H_bar, m_bar
                )
            else:
                chain.append(theta.clone())

        return torch.stack(chain)
```

### Blocked NUTS Parameter Sampling

```
Parameter blocks for NUTS:

causal_main:
    Parameters: treatment → outcome edge weights
    Includes: canopy_effect, impervious_effect, albedo_effect
    Prior: sign constraints (-inf if violated) + Gaussian(mu_lit, sigma_lit)
    Note: sign constraints enforced in log_posterior — zero gradient
          through forbidden regions, clean rejection

causal_mediated:
    Parameters: mediated pathway coefficients
    Includes: ndvi_effect, canopy_to_ndvi
    Prior: Gaussian with wider sigma — mediation less certain

process_rate:
    Parameters: alpha_mean, alpha_spatial_variance
    Prior: log-normal centered on domain mixture prior
    Bounds: enforced in log_posterior via -inf outside range

spatial:
    Parameters: rho (spatial lag), sigma2 (noise variance)
    rho prior: Beta(2, 2) — weakly symmetric on [0, 1)
    sigma2 prior: InverseGamma — conjugate for Gaussian likelihood

hyperparameters:
    Parameters: prior_trust_weight
    Prior: Beta(5, 2) — weakly informative, slightly favoring literature
    Interpretation: 1.0 = data controls, 0.0 = physics priors control
```

### Convergence Diagnostics

```python
def compute_diagnostics(chain, param_names):
    """
    Run after NUTS sampling. Flag non-convergence prominently
    in Stage 3 report — same prominence as DoWhy refutation failures.
    """
    n, p = chain.shape
    diagnostics = {}

    for i, name in enumerate(param_names):
        samples = chain[:, i].numpy()

        # R-hat via split-chain method
        chain_a, chain_b = samples[:n//2], samples[n//2:]
        W = (np.var(chain_a) + np.var(chain_b)) / 2
        B = (n//2) * np.var([np.mean(chain_a), np.mean(chain_b)])
        var_hat = (1 - 1/(n//2)) * W + B/(n//2)
        r_hat = np.sqrt(var_hat / W)

        # ESS via autocorrelation (Geyer estimator)
        f = np.fft.fft(samples - samples.mean(), n=2*n)
        acf = np.fft.ifft(f * np.conj(f)).real[:n] / (samples.var() * n)
        pairs = acf[1::2] + acf[2::2]
        cutoff = next((i for i, p in enumerate(pairs) if p < 0), len(pairs))
        ess = n / (1 + 2 * acf[1:2*cutoff+1].sum())

        diagnostics[name] = {
            'posterior_mean': samples.mean(),
            'posterior_std':  samples.std(),
            'ci_5':           np.percentile(samples, 5),
            'ci_95':          np.percentile(samples, 95),
            'r_hat':          r_hat,
            'ess':            max(ess, 1.0),
            'converged':      r_hat < 1.05 and ess > 100,
            'well_converged': r_hat < 1.01 and ess > 400
        }

    return diagnostics
```

---

---

# PART FOUR — STAGE 4: BAYESIAN SCENARIO SIMULATION

---

## 4.1 — Joint Posterior Simulation

Stage 4 Mode 3 draws jointly from MC³ (graph structure) and NUTS (parameter values). Every scenario draw uses a different plausible causal structure AND a different plausible parameter vector. Uncertainty is propagated from both sources simultaneously.

```python
def bayesian_scenario_simulation(
    intervention, mc3_results, nuts_chain,
    X_baseline, land_cover, physics_constraints,
    process_rate_net, threshold,
    n_scenario_samples=1000
):
    """
    For each draw:
        G     ← MC³ posterior (causal structure)
        theta ← NUTS chain (parameter values)
        alpha ← process_rate_net posterior (process rate)

    Computes intervention delta under each (G, theta, alpha) triple.
    Returns full distributional result.
    """
    thin_idx = np.linspace(0, len(nuts_chain)-1,
                           n_scenario_samples, dtype=int)

    outcome_distribution = np.zeros((n_scenario_samples, len(X_baseline)))
    uncertainty_sources  = []

    for i, idx in enumerate(thin_idx):
        # Sample causal structure from MC³ posterior
        G = mc3_results.sample_dag()

        # Sample parameters from NUTS chain
        theta = nuts_chain[idx]

        # Physics constraints ALWAYS enforced — regardless of draw
        theta = apply_physics_constraints(theta, physics_constraints)

        # Compute intervention effect under this (G, theta) pair
        delta = compute_intervention_delta(
            intervention, X_baseline, G, theta, land_cover,
            process_rate_net
        )
        outcome_distribution[i] = delta

        # Track uncertainty sources for decomposition
        uncertainty_sources.append({
            'graph_entropy':  mc3_results.graph_entropy(G),
            'param_variance': theta.var().item()
        })

    return {
        'posterior_mean':     outcome_distribution.mean(axis=0),
        'posterior_std':      outcome_distribution.std(axis=0),
        'ci_5':               np.percentile(outcome_distribution, 5,  axis=0),
        'ci_50':              np.percentile(outcome_distribution, 50, axis=0),
        'ci_95':              np.percentile(outcome_distribution, 95, axis=0),
        'exceedance_prob':    (outcome_distribution > threshold).mean(axis=0),
        'uncertainty_decomp': decompose_uncertainty(uncertainty_sources),
        'samples':            outcome_distribution   # full distribution
    }


def decompose_uncertainty(sources):
    """
    Decomposes total variance into structural and parametric components.
    Tells the planner: is uncertainty from not knowing the causal
    structure (need better theory) or not knowing the parameters
    (need more data)?
    """
    graph_var  = np.var([s['graph_entropy'] for s in sources])
    param_var  = np.var([s['param_variance'] for s in sources])
    total_var  = graph_var + param_var + 1e-10

    return {
        'graph_structure_pct': 100 * graph_var / total_var,
        'parameter_pct':       100 * param_var / total_var,
        'dominant_source':     'graph_structure' if graph_var > param_var else 'parameters',
        'interpretation': (
            'Uncertainty dominated by causal structure — the mechanism is not '
            'well-established for this domain. Causal theory needs development.'
            if graph_var > param_var else
            'Uncertainty dominated by parameter estimation — the mechanism is '
            'known but coefficient magnitudes are uncertain. More observations '
            'would narrow the credible interval.'
        )
    }
```

---

## 4.2 — Posterior Predictive Checks

```python
def posterior_predictive_checks(model, nuts_chain, data, weights, n_draws=200):
    """
    Generates synthetic data from posterior and compares to observed.
    Bayesian p-values near 0.5 = good fit.
    Near 0 or 1 = systematic model misspecification.
    Moran's I check is the critical spatial diagnostic.
    """
    thin_idx = np.linspace(0, len(nuts_chain)-1, n_draws, dtype=int)
    test_statistics = defaultdict(list)

    for idx in thin_idx:
        theta = nuts_chain[idx]
        y_synthetic = model.generate(theta, data['X'])

        residuals = data['y'] - y_synthetic
        test_statistics['mean'].append(y_synthetic.mean())
        test_statistics['std'].append(y_synthetic.std())
        test_statistics['skewness'].append(scipy.stats.skew(y_synthetic))
        test_statistics['morans_i'].append(
            compute_morans_i(residuals, weights)
        )

    obs = {
        'mean':     data['y'].mean(),
        'std':      data['y'].std(),
        'skewness': scipy.stats.skew(data['y']),
        'morans_i': compute_morans_i(data['y'] - model.predict_mean(nuts_chain, data), weights)
    }

    bayesian_pvalues = {
        stat: np.mean(np.array(test_statistics[stat]) > obs[stat])
        for stat in obs
    }

    return {
        'bayesian_pvalues': bayesian_pvalues,
        'target':           '~0.5 for each statistic',
        'flag':             {k: 'FAIL' if abs(v - 0.5) > 0.4 else 'PASS'
                             for k, v in bayesian_pvalues.items()}
    }
```

---

---

# PART FIVE — COMPLETE FILE STRUCTURE

---

```
sparc/
│
├── models/
│   ├── surrogates.py            DifferentiableGWR, DifferentiableGWRF,
│   │                            DifferentiableGGPGAM, validate_surrogates()
│   ├── process_rate_net.py      ProcessRateNet, compute_mixture_prior(),
│   │                            material_validation_report()
│   ├── spatial_attention.py     SIRENLayer, SparseSpatialAttention,
│   │                            attention_influence_map()
│   ├── neural_meta.py           SPARCMetaLearner, predict_with_uncertainty(),
│   │                            predict_for_nuts()
│   └── downscaler.py            SpatialTransferLearner, resolution_levels config
│
├── causal/
│   ├── mc3.py                   DAGStructure, DAGProposal, PhysicsInformedGraphPrior,
│   │                            bge_local_score(), ParallelTemperingMC3, MC3Results
│   └── nuts.py                  NUTSSampler, dual_averaging_update(),
│                                compute_diagnostics(), blocked parameter sampling
│
├── training/
│   ├── loss.py                  sparc_joint_loss() (8 terms),
│   │                            get_prior_weight(), compute_physics_residual()
│   ├── optimizer.py             build_optimizer(), build_scheduler(),
│   │                            warmup_scheduler(), training_step()
│   ├── cma_es.py                cma_es_objective(), run_cma_es(),
│   │                            cma_search_space definition
│   ├── maml.py                  maml_spatial_cv_loss()
│   └── curriculum.py            get_lambda_schedule(), 4-stage schedule,
│                                apply_swa(), capacity_sweep()
│
├── features/
│   └── sinusoidal_encoding.py   SinusoidalSpatialEncoding (replaces eigenmaps)
│
├── interventions/
│   └── scenario_simulator.py    bayesian_scenario_simulation(),
│                                decompose_uncertainty(),
│                                posterior_predictive_checks()
│                                (Modes 1 and 2 unchanged)
│
└── [unchanged from V1]
    ├── config/
    ├── data/
    ├── evaluation/
    ├── run/
    ├── ui/
    └── templates/    (all 13 + process_rate block added to each)
```

---

---

# PART SIX — COMPLETE OUTPUT REGISTRY

---

## Stage 2 Outputs

| Output | File | Description | Phase |
|---|---|---|---|
| Neural prediction surface | `stage2/prediction_surface.tif` | Point-wise outcome prediction | Core |
| MC Dropout std | `stage2/uncertainty_std.tif` | Epistemic uncertainty | Core |
| CI 5th / 95th percentile | `stage2/uncertainty_ci*.tif` | Credible prediction bounds | Core |
| Physics residual field | `stage2/physics_residual.tif` | PDE satisfaction per point | Core |
| Stream attribution | `stage2/stream_attribution.*` | Base vs. physics vs. spatial | Core |
| Exceedance probability maps | `stage2/exceedance_p[threshold].tif` | P(outcome > threshold) — trained directly | Core |
| Model performance table | `stage2/model_performance.csv` | V1 vs. V2 comparison | Core |
| Integrated gradients | `stage2/feature_importance_gradients.csv` | Per-feature attribution | Core |
| Surrogate validation | `stage2/surrogate_validation.csv` | R² vs. true base models | Core |
| Learned process rate field | `stage2/alpha_field.tif` | Per-point α across study area | Core |
| Process rate prior field | `stage2/alpha_prior_field.tif` | Physics mixture prior | Core |
| Process rate residual | `stage2/alpha_residual.tif` | Learned minus prior | Core |
| Material validation report | `stage2/alpha_material_validation.csv` | Learned α vs. literature | Core |
| Attention influence maps | `stage2/attention_influence_[idx].tif` | Spatial influence for selected points | Core |
| Capacity sweep | `stage2/figures/double_descent_sweep.png` | hidden_dim vs. CV R² | Core |
| Alpha convergence | `stage2/figures/alpha_convergence.png` | Process rate training stability | Core |
| Alpha vs. prior scatter | `stage2/figures/alpha_validation.png` | Paper validation figure | Core |
| Calibration curve | `stage2/figures/calibration_curve.png` | MC Dropout coverage vs. nominal | Core |
| CMA-ES convergence | `stage2/figures/cma_es_convergence.png` | Hyperparameter search history | Core |
| SWA loss curve | `stage2/figures/swa_loss.png` | Training + SWA phase | Core |

## Stage 3 Outputs

| Output | File | Description |
|---|---|---|
| Posterior edge probabilities | `stage3/posterior_edge_probs.csv` | P(edge\|data) all variable pairs |
| Edge confidence report | `stage3/edge_confidence.csv` | Strong/moderate/weak/absent/surprising/missing |
| Median probability DAG | `stage3/median_probability_dag.json` | Best single graph |
| BMA causal coefficients | `stage3/bma_coefficients.csv` | Model-averaged effects |
| Parameter posteriors | `stage3/parameter_posteriors.csv` | Mean, std, CI 5/95 per parameter |
| Convergence diagnostics | `stage3/convergence_diagnostics.csv` | R-hat, ESS per parameter |
| Posterior predictive checks | `stage3/ppc.csv` | Bayesian p-values — 4 test statistics |
| DoWhy refutation tests | `stage3/refutation_tests.csv` | Unchanged from V1 |
| Trace plots | `stage3/figures/trace_plots.png` | Visual chain convergence |
| Posterior densities | `stage3/figures/posterior_densities.png` | Marginal distributions |
| Autocorrelation plots | `stage3/figures/autocorrelation.png` | Mixing speed by lag |
| Edge probability heatmap | `stage3/figures/dag_posterior_heatmap.png` | n_vars × n_vars matrix |
| Median DAG diagram | `stage3/figures/median_probability_dag.png` | Colored by confidence tier |
| R-hat summary | `stage3/figures/rhat_summary.png` | Bar chart, threshold at 1.05 |
| ESS summary | `stage3/figures/ess_summary.png` | Bar chart, threshold at 100 |

## Stage 4 Outputs (per scenario)

| Output | File | Description |
|---|---|---|
| Mode 1 prediction | `stage4/[scenario]/mode1_prediction.tif` | Ensemble re-prediction (unchanged) |
| Mode 2 causal | `stage4/[scenario]/mode2_causal.tif` | DAG + MGWR (unchanged) |
| Posterior mean effect | `stage4/[scenario]/mode3_posterior_mean.tif` | Expected intervention delta |
| Posterior std | `stage4/[scenario]/mode3_posterior_std.tif` | Combined structural + parameter uncertainty |
| CI 5th / 50th / 95th | `stage4/[scenario]/mode3_ci*.tif` | Full distributional result |
| Exceedance probability | `stage4/[scenario]/mode3_exceedance_prob.tif` | P(effect > threshold) |
| Uncertainty decomposition | `stage4/[scenario]/uncertainty_decomp.csv` | Graph vs. parameter source breakdown |
| Diminishing returns | `stage4/[scenario]/diminishing_returns.png` | Dose-response curve with credible bands |
| BMA scenario summary | `stage4/bma_scenario_summary.csv` | All scenarios model-averaged |

---

---

# PART SEVEN — VISUALIZATION SPECIFICATIONS

---

## Key Maps

**Process rate residual (α learned − α prior):** Diverging colormap centered at zero. This is a scientific finding map — systematic positive residual over a land cover class means that surface diffuses heat faster than naive material mixing predicts. Annotate the 5 largest deviation zones with land cover context and coordinates. Include in Paper 1 as a key figure.

**Exceedance probability P(effect > threshold):** Sequential colormap white → deep red. Contour lines at P=0.50 and P=0.80. This is the most policy-relevant map SPARC produces. A planner looks at this and immediately sees where an intervention reliably meets the performance standard vs. where it is uncertain. Caption should read: "Probability that [intervention] produces [outcome change] exceeding [threshold] at each location."

**Edge probability heatmap:** Square n_vars × n_vars heatmap. White = P=0, deep blue = P=1.0. Expert DAG edges drawn as red outlines — agreement and disagreement immediately visible. Annotate cells where P > 0.5 with the probability value. Surprising inclusions (P > 0.70, not in expert DAG) annotated in amber.

**Median probability DAG:** Standard DAG diagram with edge color by confidence tier: green (strong, P > 0.90), blue (moderate, P > 0.70), yellow (weak, P > 0.50). Edge thickness proportional to BMA coefficient magnitude. Dashed edges for surprising inclusions. Present expert DAG as adjacent panel for direct comparison.

**Trace plots:** One panel per parameter, 4 colored lines (one per temperature-1 chain). Post-burnin only. Good mixing: chains interleaved like blowing grass. Convergence failure: chains stuck in separate regions. Label burnin cutoff with vertical line.

**Diminishing returns curve:** x-axis = intervention magnitude (units of predictor). y-axis = posterior mean outcome delta. Solid line = posterior mean. Shaded band = 5th to 95th percentile. Vertical dashed line at inflection point. Width of band at each x-value shows how uncertainty grows with extrapolation. Caption: "Diminishing returns become apparent beyond [inflection point]. Credible interval widens substantially beyond the data support region."

**Uncertainty decomposition:** Stacked horizontal bar chart per spatial zone (neighborhood / planning district / census tract). Three segments: graph structure uncertainty (orange), parameter uncertainty (blue). Process rate uncertainty (purple) available as separate bar if relevant. Legend clearly labels each segment. Zones sorted by total uncertainty. Include interpretation: "Parameter uncertainty dominates → more field observations needed. Graph structure uncertainty dominates → causal mechanism needs theoretical development."

---

---

# PART EIGHT — project.yml REFERENCE

---

Complete additions to `project.yml` for V2. All fields optional — V1 defaults preserved.

```yaml
# V2 model configuration
models:
  meta_learner: neural           # 'lightgbm' restores V1 behavior
  neural:
    hidden_dim: 256              # set after capacity sweep
    dropout: 0.1
    n_heads: 4                   # spatial attention heads
    max_neighbors: 128           # sparse attention neighborhood
    mc_dropout_samples: 100
    exceedance_thresholds: [0.25, 0.50, 0.75]   # in outcome units

# Process rate network
process_rate:
  enabled: true
  name: thermal_diffusivity      # domain-specific — see domain template
  units: m2_per_s
  bounds: [1.0e-7, 9.0e-7]
  prior_mean: 5.0e-7
  inputs: [Pct_Impervious, Pct_Canopy, Pct_Water, NDVI, soil_moisture]
  material_priors:
    impervious: 7.5e-7
    canopy:     3.0e-7
    water:      1.4e-7
    soil:       3.5e-7

# Optimization
optimization:
  run_cma_es: true               # run hyperparameter search before training
  cma_es_popsize: 20
  cma_es_maxiter: 50
  clip_norm: 1.0                 # overridden by CMA-ES if run_cma_es: true
  swa_epochs: 20
  capacity_sweep: true           # run hidden_dim sweep before training
  capacity_sweep_dims: [64, 128, 256, 512, 1024]

# Training curriculum
training:
  n_epochs: 100
  batch_size: 2048
  warmup_epochs: 10
  ramp_epochs: 30
  lambda_physics: 0.1            # overridden by CMA-ES if run_cma_es: true
  lambda_smooth: 0.01
  lambda_alpha_smooth: 0.01
  lambda_prior: 0.01
  lambda_base: 0.2
  lambda_neighbor: 0.05

# Causal inference
causal:
  inference: bayesian            # 'frequentist' restores V1 DML behavior

  mc3:
    enabled: true
    temperatures: [1.0, 1.5, 2.0, 3.0]
    n_burnin: 3000
    n_samples: 10000
    thin: 5
    swap_interval: 50
    prior_edge_probability: 0.25
    physics_edge_bonuses: true

  nuts:
    n_burnin: 1000
    n_samples: 3000
    target_accept: 0.80
    condition_on: median_probability_dag
    blocking: true

# Multi-resolution transfer (optional)
transfer:
  enabled: false
  pretrained_backbone: null      # path to pretrained model from coarser resolution
  freeze_backbone: true
  fine_features_dim: 32

# MAML spatial CV
maml:
  enabled: false                 # enable after base training converges
  inner_lr: 0.001
  n_inner_steps: 5

# Scenario simulation
scenarios:
  exceedance_threshold: 0.5     # for P(effect > threshold) maps
  n_posterior_samples: 1000     # joint MC³ + NUTS draws per scenario
```

---

---

# PART NINE — BACKWARD COMPATIBILITY

---

V1 behavior is fully reproducible in V2:

```yaml
models:
  meta_learner: lightgbm         # bypasses neural meta-learner

process_rate:
  enabled: false                 # bypasses auxiliary network

optimization:
  run_cma_es: false
  capacity_sweep: false

causal:
  inference: frequentist         # bypasses MC³ and NUTS, uses V1 DML

maml:
  enabled: false

transfer:
  enabled: false
```

All 13 existing domain templates, CLI, Streamlit UI, Stage 0, Stage 1, DoWhy refutation tests, Mode 1 and Mode 2 simulation are unchanged. New `project.yml` fields are optional with V1 defaults.

---

---

# PART TEN — WHAT V2 TELLS A PLANNER

---

For any adaptation measure in any location:

| Question | Primary output | Supporting outputs |
|---|---|---|
| What will likely happen? | `mode3_posterior_mean.tif` | `bma_coefficients.csv`, `diminishing_returns.png` |
| How confident are we? | `mode3_posterior_std.tif`, `uncertainty_decomp.csv` | `parameter_posteriors.csv`, convergence dashboard |
| How does it ripple? | `bma_coefficients.csv` (mediation) | `median_probability_dag.png`, `edge_confidence.csv` |
| Where does it work? | `mode3_exceedance_prob.tif` | `stream_attribution.tif`, `alpha_residual.tif` |

**What V2 answers that V1 cannot:**

| Question | V1 | V2 |
|---|---|---|
| How much will this intervention cool the area? | Point estimate ± bootstrap CI | Full posterior distribution |
| Is my assumed causal structure correct? | F1 vs. expert DAG | Posterior P(edge) for every variable pair |
| Which pathways am I wrong about? | Not answerable | Missing edges (P < 0.30, in expert DAG) |
| What pathways did I miss? | Not answerable | Surprising edges (P > 0.70, not in expert DAG) |
| How thermally distinct are my surfaces? | Not answerable | Learned α field validated against material literature |
| Is my model structurally misspecified? | Not answerable | Bayesian p-values on 4 test statistics |
| What is the probability this meets the standard? | Not answerable | P(effect > threshold) spatial map — trained directly |
| What is driving my uncertainty? | Not answerable | Uncertainty decomposition: graph structure vs. parameters |
| Should I collect more data or refine theory? | Not answerable | Dominant uncertainty source gives the answer |

---

*Document version: April 2026*
*Status: Locked for V2 development*
*Next review: Upon JAMES Paper 1 submission — Q1 2027*