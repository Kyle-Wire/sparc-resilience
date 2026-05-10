# Continual & Transfer Learning Research

Sources that shaped the **V3 continual learning infrastructure** — EWC, experience replay, the City Registry, transfer training, and the Welford online scaler — described in `SPARC_V4_Roadmap_Integrated.md` and implemented in `sparc/training/`, `sparc/registry/`, and `sparc/run/`.

---

## Catastrophic Forgetting

> **McCloskey, M., & Cohen, N.J. (1989).** "Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem." *Psychology of Learning and Motivation*, 24, 109–165. https://doi.org/10.1016/S0079-7421(08)60536-8

> **French, R.M. (1999).** "Catastrophic forgetting in connectionist networks." *Trends in Cognitive Sciences*, 3(4), 128–135. https://doi.org/10.1016/S1364-6613(99)01294-2

The core problem motivating V3: when a neural network trained on City A is fine-tuned on City B, it forgets City A — the weights shift to minimize City B loss at the expense of City A performance. SPARC's continual learning stack exists to prevent this, enabling the SharedTrunk to accumulate physics knowledge across cities without forgetting earlier ones.

---

## Elastic Weight Consolidation (EWC)

> **Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017).** "Overcoming catastrophic forgetting in neural networks." *Proceedings of the National Academy of Sciences*, 114(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114

EWC is implemented in `sparc/training/ewc.py` and is cited directly in that file's module docstring. After training on a city, SPARC computes the diagonal Fisher information matrix for the SharedTrunk parameters. The Fisher approximates which weights were most important for that city's task. During subsequent training on a new city, the EWC penalty term discourages large changes to those high-Fisher weights:

```
L_total = L_task + λ_ewc · Σ_i  F_i · (θ_i − θ*_i)²
```

This is the exact formulation from Kirkpatrick et al. (2017), using the squared L2 deviation weighted by the diagonal Fisher.

---

## Experience Replay / Coreset Selection

> **Rebuffi, S.A., Kolesnikov, A., Sperl, G., & Lampert, C.H. (2017).** "iCaRL: Incremental Classifier and Representation Learning." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2001–2010. https://arxiv.org/abs/1611.07725

> **Lopez-Paz, D., & Ranzato, M. (2017).** "Gradient Episodic Memory for Continual Learning." *Advances in Neural Information Processing Systems (NeurIPS)*, 30. https://arxiv.org/abs/1706.08840

`CoresetSelector` in `sparc/training/replay.py` selects 400 representative points from each city's training data via a greedy K-medoids facility-location approximation. During training on a new city, the replay loss computes predictions on the coreset and penalizes deviation from the previously correct outputs — anchoring the model to its prior knowledge without storing all prior data.

The K-medoids approach is preferred over random sampling (as in GEM/A-GEM) because spatial data is highly redundant: nearby points are nearly identical, so random sampling wastes coreset budget on clusters. K-medoids selects maximally spread representatives.

---

## K-Medoids Clustering

> **Kaufman, L., & Rousseeuw, P.J. (1990).** *Finding Groups in Data: An Introduction to Cluster Analysis.* Wiley. https://doi.org/10.1002/9780470316801

> **Park, H.S., & Jun, C.H. (2009).** "A simple and fast algorithm for K-medoids clustering." *Expert Systems with Applications*, 36(2), 3336–3341. https://doi.org/10.1016/j.eswa.2008.01.039

The greedy K-medoids facility-location approximation in `CoresetSelector._greedy_kmedoids()` iteratively picks the point that maximally reduces the minimum-distance-to-nearest-medoid across the dataset. This is an O(N·k) greedy approximation with bounded approximation ratio, trading optimality for speed on the 50k-point datasets typical of a SPARC city.

---

## Transfer Learning for Spatial Models

> **Pan, S.J., & Yang, Q. (2010).** "A Survey on Transfer Learning." *IEEE Transactions on Knowledge and Data Engineering*, 22(10), 1345–1359. https://doi.org/10.1109/TKDE.2009.191

> **Weiss, K., Khoshgoftaar, T.M., & Wang, D. (2016).** "A survey of transfer learning." *Journal of Big Data*, 3, 9. https://doi.org/10.1186/s40537-016-0043-6

The SharedTrunk / CityHead split is a deliberate transfer learning architecture. The SharedTrunk encodes portable physics knowledge (thermal diffusivity patterns, PDE structure) that transfers across cities, while the CityHead is retrained per deployment. This inductive bias means a model trained on Providence can warm-start on Boston, reaching a better R² in fewer epochs than cold-start training.

`sparc/training/transfer_training.py` implements three training modes:
- `train_cold_start()` — full fresh training
- `train_warm_start()` — frozen trunk from source city, train head only
- `train_warm_start_finetune()` — warm start, then unfreeze trunk at a configured epoch

---

## Online / Welford Statistics

> **Welford, B.P. (1962).** "Note on a method for calculating corrected sums of squares and products." *Technometrics*, 4(3), 419–420. https://doi.org/10.1080/00401706.1962.10490022

> **Chan, T.F., Golub, G.H., & LeVeque, R.J. (1979).** "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances." *Technical Report STAN-CS-79-773*, Stanford University.

`WelfordScaler` in `sparc/data/welford.py` maintains running mean and variance across cities without storing any past data, using the Welford online algorithm. This enables standardization of features across a growing multi-city dataset without ever materializing all cities' data simultaneously — a key privacy and memory property of the federated registry design.

---

## Federated / Privacy-Preserving Learning

> **McMahan, H.B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017).** "Communication-Efficient Learning of Deep Networks from Decentralized Data." *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*, 54, 1273–1282. https://arxiv.org/abs/1602.05629

The V4 Central Registry design follows federated learning principles: only model artifacts (trunk weights, Fisher matrices, coresets of 400 standardized points, Welford statistics) leave the user's machine — raw geospatial data never does. The coreset is too small and transformed to reconstruct original observations. This mirrors the FedAvg aggregation principle: merge model-level summaries without centralizing data.

---

## Continual Learning Surveys

> **Parisi, G.I., Kemker, R., Part, J.L., Kanan, C., & Wermter, S. (2019).** "Continual lifelong learning with neural networks: A review." *Neural Networks*, 113, 54–71. https://doi.org/10.1016/j.neunet.2019.01.012

> **De Lange, M., Aljundi, R., Masana, M., Parrini, S., Javed, K., Babiloni, F., Tuytelaars, T., & Van de Walle, R. (2022).** "A continual learning survey: Defying forgetting in classification tasks." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(7), 3366–3385. https://doi.org/10.1109/TPAMI.2021.3057446

These surveys contextualize SPARC's approach: combining regularization (EWC), replay (coreset), and architectural separation (SharedTrunk + CityHead) is a common best-practice cocktail for continual learning — no single method dominates, so combining all three provides complementary protection against forgetting.
