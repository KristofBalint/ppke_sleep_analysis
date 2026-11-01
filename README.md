# PPKE Sleep Analysis

**Multimodal polysomnography (PSG) analysis for cognitive impairment detection.**

This repository contains the analysis and machine-learning pipelines developed by the AIMS-lab team at Pázmány Péter Catholic University for the **George B. Moody PhysioNet Challenge 2026: Screening for Cognitive Impairment During Sleep Studies**.

The project investigates whether physiological signals and sleep-related information collected during overnight polysomnography can be used to identify cognitive impairment. It focuses on extracting informative features from multiple signal modalities, evaluating their predictive value, and combining handcrafted features with pretrained sleep representations.

## Project overview

The analysis explores three modelling approaches:

1. **Single-modality analysis:** Handcrafted features are extracted from individual physiological modalities and used to train XGBoost classifiers. This provides a basis for comparing the predictive information available in different signal types.
2. **Stratified feature selection:** Features are evaluated across individual modalities, sleep stages, and modality–stage combinations. Several feature-reduction strategies are investigated to identify informative features while reducing redundancy.
3. **Hybrid SleepFM approach:** Handcrafted features are combined with representations extracted by the pretrained SleepFM model. The frozen SleepFM backbone provides learned physiological representations, which are fused with the handcrafted feature branch using a neural classifier.

## Data

The project uses the PSG recordings and associated annotations provided for the 2026 PhysioNet Challenge.

The recordings contain several physiological signal modalities:

- **EEG:** Electroencephalography
- **EOG:** Electrooculography
- **EMG:** Electromyography, including chin and leg channels
- **ECG:** Electrocardiography
- **Respiration:** Respiratory effort, airflow and oxygen saturation signals

Additional information includes sleep-stage annotations, respiratory events, arousals, and limb movements. Demographic variables are also used in the prediction pipeline.

The prediction target is `Cognitive_Impairment`.

The dataset is not included in this repository. Access and use are subject to the PhysioNet Challenge data-access requirements.

## Methodology

### Signal processing and feature extraction

PSG recordings are processed on a channel-specific basis. Signals are grouped according to their physiological modality, and features are extracted separately for individual channels and sleep stages where applicable.

The extracted features include:

- **General signal characteristics:** Statistical moments, signal amplitude, energy, variability, and temporal patterns.
- **Signal quality:** Flatline, clipping, amplitude, and other quality-related measurements.
- **EEG features:** Spectral characteristics, relative frequency-band power, and sleep spindle metrics.
- **ECG features:** Heart-rate variability and nonlinear HRV measurements.
- **Respiratory features:** Respiratory rate, variability, and temporal characteristics.
- **EMG features:** Signal and spectral characteristics.
- **Synchrony features:** Correlation, cross-correlation, lag, amplitude ratio, and phase-locking measurements between signals.

Feature extraction uses tools including [tsfresh](https://tsfresh.com/), [YASA](https://yasa-sleep.org/), and other signal-processing utilities.

### Stratified feature selection

The feature-selection pipeline evaluates several levels of signal grouping:

- Individual physiological modalities.
- Individual sleep stages.
- Combinations of modalities and sleep stages.

Within each stratum, features are evaluated using different reduction strategies:

1. **NaN-count filtering:** Features are progressively removed according to the number of missing values.
2. **Dimensionality reduction:** PCA or recursive feature elimination (RFE) is applied to selected feature subsets.
3. **Model evaluation:** XGBoost classifiers are evaluated using subject-grouped cross-validation.
4. **Feature weighting:** Model feature importances are weighted by the corresponding AUROC and accumulated across the evaluated strata and reduction configurations.

The resulting global feature weights are used to rank features and investigate how much of the original feature set can be retained without substantially reducing predictive performance.

### Hybrid SleepFM approach

The hybrid approach combines handcrafted physiological features with learned representations from the pretrained [SleepFM](https://github.com/zou-group/sleepfm-clinical) model.

Signals are mapped to the SleepFM modality groups (BAS, RESP, EKG and EMG), resampled to 128 Hz, normalized, and divided into five-minute segments. The pretrained SleepFM SetTransformer produces 128-dimensional segment representations, which are aggregated into a patient-level embedding.

In parallel, selected handcrafted features are imputed, standardized, and projected into a 128-dimensional representation. The two representations are concatenated and passed to a neural classification head.

The pretrained SleepFM backbone remains frozen during training; only the handcrafted projection and fusion classifier are optimized.

## Evaluation

Model performance is primarily assessed using the area under the receiver operating characteristic curve (AUROC). Subject-level grouping is used during data splitting and cross-validation to avoid placing recordings from the same subject in both training and validation folds.

The analysis also considers other classification metrics, including sensitivity, precision, balanced accuracy, and area under the precision–recall curve (AUPRC).

The project distinguishes between internal model-development experiments and evaluation performed as part of the official PhysioNet Challenge.

## Tools and libraries

The project uses Python and a range of scientific-computing and machine-learning libraries, including:

- [NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/) – numerical computing and data handling
- [MNE](https://mne.tools/) – physiological signal processing
- [scikit-learn](https://scikit-learn.org/) – preprocessing, feature selection, and evaluation
- [XGBoost](https://xgboost.readthedocs.io/) – gradient-boosted classification
- [tsfresh](https://tsfresh.com/) – time-series feature extraction
- [YASA](https://yasa-sleep.org/) – sleep analysis
- [SleepFM](https://github.com/zou-group/sleepfm-clinical) – pretrained sleep representations
- PyTorch – neural-network training and hybrid modelling

## Reproducibility and data considerations

- The PSG dataset is not distributed with this repository.
- Model training and evaluation require access to the corresponding Challenge data.
- Subject identifiers are used for grouping recordings during evaluation.
- Results from internal experiments should not be interpreted as official Challenge results unless explicitly identified as such.
- Exact results may depend on preprocessing choices, feature availability, and the selected model configuration.

## Acknowledgements

This work was developed at **Pázmány Péter Catholic University**, within the AIMS-lab research group, as part of the George B. Moody PhysioNet Challenge 2026.

We acknowledge the PhysioNet Challenge organizers and the researchers responsible for collecting and preparing the PSG datasets.

## References

- George B. Moody PhysioNet Challenge 2026: *Screening for Cognitive Impairment During Sleep Studies*. [PhysioNet](https://physionet.org/)
- SleepFM: [zou-group/sleepfm-clinical](https://github.com/zou-group/sleepfm-clinical)
- MNE: [mne.tools](https://mne.tools/)
- YASA: [yasa-sleep.org](https://yasa-sleep.org/)
- tsfresh: [tsfresh.com](https://tsfresh.com/)
