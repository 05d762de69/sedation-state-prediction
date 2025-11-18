# Sedation State Prediction using EEG Graph-Theoretical Features

Machine-learning pipeline for predicting sedation states (responsiveness in earlier versions) during propofol sedation based on EEG-derived graph-theoretical measures (e.g., small-worldness, clustering, mean degree).

## Overview

This project investigates whether network topology metrics derived from EEG connectivity can predict sedation depth during propofol anesthesia.
This project reuses the open-access EEG dataset from [Chennu (2016)](https://doi.org/10.1371/journal.pcbi.1004669) and extends their graph-theoretical analysis of sedation-related connectivity changes by applying machine-learning models (Random Forests and Convolutional Neural Networks (CNNs)) to graph-theoretical features of large-scale brain networks.

## Data

The dataset originates from [Chennu et al. (2016)](https://doi.org/10.1371/journal.pcbi.1004669), an open-access EEG study on propofol-induced changes in functional connectivity.  
Raw EEG data are available through the PLOS repository and are **not redistributed here** in compliance with data-sharing policies.

Derived features (dwPLI matrices, graph metrics) are generated via scripts in `src/connectivity.py` and `src/graph_metrics`.

## Methods

### Feature Extraction

EEG epochs were processed to compute connectivity using the weighted Phase Lag Index (wPLI) across five canonical frequency bands (δ = 1–4 Hz, θ = 4–8 Hz, α = 8–13 Hz, β = 13–30 Hz, γ = 30–45 Hz).
All connectivity and graph-theoretical computations were executed on the UNSW Katana high-performance computing cluster, using PBS batch scripts and the `compute_graphmetrics.py` pipeline.

For each subject × sedation level × band, the following graph-theoretical features were extracted from the weighted connectivity matrices:

| Metric | Description |
|---------|-------------|
| **mean_degree** | Average node degree (mean connection strength). |
| **clustering** | Weighted clustering coefficient (local interconnectedness). |
| **path_length** | Mean shortest path between all node pairs (integration). |
| **global_efficiency** | Inverse of average shortest path (global integration). |
| **local_efficiency** | Efficiency within neighborhoods (segregation). |
| **modularity (Q)** | Strength of community structure (Louvain algorithm). |
| **participation_coefficient** | Extent of cross-module connectivity. |
| **small_worldness** | Ratio of normalized clustering to normalized path length vs. random graphs. |

These metrics quantify large-scale network segregation, integration, and modular organization.

### Feature Transformation

After downloading the HPC-generated outputs, features were curated and transformed in a dedicated feature-processing notebook. Graph metrics were merged with propofol plasma concentration and behavioral performance (reaction time and accuracy). Metrics were then normalized within-subject and Δ-transformed relative to each subject’s own baseline, removing inter-individual differences in absolute connectivity while isolating sedation-related changes:

$\delta f = \frac{(f_{level} − f_{baseline})}{f_{baseline}}$


### Normalization and Data Curation

The final dataset contained both raw graph metrics and Δ-normalized features, aggregated across frequency bands, and restricted to the sedation levels Baseline (1), Mild (2), and Moderate (3). The Recovery state was excluded due to its transitional physiological profile.

### Model Training

Machine-learning models were trained to classify the three sedation levels. A Random Forest (RF) classifier was trained using a preprocessing pipeline that included standard scaling and PCA (retaining 90–99% variance). Hyperparameters (e.g., max_depth, n_estimators, max_features, min_samples_leaf) were tuned using group-aware GridSearchCV, ensuring that subjects were strictly segregated across folds.

In parallel, a 1D Convolutional Neural Network (CNN) was trained on reshaped feature vectors (N × F × 1), with convolutional layers, batch normalization, dropout, and dense layers. A structured hyperparameter search explored convolutional filter counts, kernel sizes, dropout probabilities, dense layer widths, and learning rates.

Both model families were evaluated using 5-fold GroupKFold cross-validation, preventing subject leakage and ensuring subject-independent generalization. Model performance was compared using fold-wise accuracy and confusion matrices, and statistical differences were assessed using a Wilcoxon signed-rank test.
Although the CNN (M ≈ .72) numerically outperformed the RF (M ≈ .68), the difference was not statistically significant (p = .50). 

>[!important]
>Importantly, both models achieved accuracies well above the 3-class chance level (.33), demonstrating that EEG-derived graph metrics contain meaningful information about sedation depth.

Cross-validation scores for both models were saved (rf_cv_scores.npy, cnn_cv_scores.npy) to support reproducible visualization and analysis.

<p align="center">
  <img src="figures/pipeline.png" alt="Pipeline flowchart" width="50%">
</p>

## Citation

This repository builds upon the dataset and framework described in:

> Chennu S, O’Connor S, Adapa R, Menon DK, Bekinschtein TA (2016) Brain Connectivity Dissociates Responsiveness from Drug Exposure during Propofol-Induced Transitions of Consciousness. PLoS Comput Biol 12(1): e1004669. https://doi.org/10.1371/journal.pcbi.1004669

If you use this repository, please cite the original study above and this repository as:

> Schätzle, H. (2025). *Sedation State Prediction using EEG Graph-Theoretical Features* [GitHub repository].  
https://github.com/05d762de69/sedation-state-prediction

## License

This repository is distributed under the MIT License.  
See `LICENSE` for details.

## Contact

For questions or collaboration:
**Hannes Schätzle**  
Erasmus University Rotterdam / UNSW Sydney  
📧 h.schaetzle@student.eur.nl