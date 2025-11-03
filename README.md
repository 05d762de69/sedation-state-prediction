# Sedation State Prediction using EEG Graph-Theoretical Features

Machine-learning pipeline for predicting sedation states (responsiveness in earlier versions) during propofol sedation based on EEG-derived graph-theoretical measures (e.g., small-worldness, clustering, mean degree).

## Overview

This project explores how network topology metrics extracted from EEG connectivity (dwPLI) relate to consciousness under propofol sedation. By leveraging graph-theoretical measures, we train machine-learning models to infer sedation states beyond descriptive analysis.
This project reuses the open-access EEG dataset from [Chennu (2016)](https://doi.org/10.1371/journal.pcbi.1004669) and extends their graph-theoretical analysis of sedation-related connectivity changes using additional machine learning approaches.




## Data

The dataset originates from [Chennu et al. (2016)](https://doi.org/10.1371/journal.pcbi.1004669), an open-access EEG study on propofol-induced changes in functional connectivity.  
Raw EEG data are available through the PLOS repository and are **not redistributed here** in compliance with data-sharing policies.

Derived features (dwPLI matrices, graph metrics) are generated via scripts in `src/connectivity.py` and `src/graph_metrics`.

## Methods

1. **Preprocessing:** EEG data filtered and segmented into epochs (already provided by [Chennu et al. (2016)](https://doi.org/10.1371/journal.pcbi.1004669))  
2. **Connectivity Analysis:** dwPLI computed for standard frequency bands.  
3. **Graph Construction:** Adjacency matrices thresholded and metrics extracted (e.g. small-worldness, clustering, modularity).  
4. **Model Training:** Machine learning classifiers trained to predict sedation responsiveness.  
5. **Evaluation:** Performance assessed via cross-validation and feature importance analysis.
![flowchart](<Screenshot 2025-11-04 at 09.38.39.png>)

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