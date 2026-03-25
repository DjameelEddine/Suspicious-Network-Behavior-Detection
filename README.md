# Approach in `Suspicious Network Behavior Detection.ipynb`

## 1) Problem Definition

- Goal: build a multiclass network intrusion detection pipeline that classifies network flows into `BENIGN` or specific attack categories.
- Challenge: the dataset is highly imbalanced, where majority benign traffic can dominate learning and hide minority attack patterns.
- Objective: improve detection quality for rare attack classes while maintaining strong overall performance.
- Scope: train supervised models using structured flow-based features and produce an inference-ready artifact for deployment/testing.

## 2) Data Quality Preparation

- Replaces infinite values (`+inf`, `-inf`) with `NaN`.
- Drops rows with high missingness (more than 30% missing values).
- Fills remaining numeric missing values using median imputation.
- Removes duplicate records.

## 3) Exploratory Analysis for Modeling Decisions

- Examines class distribution and imbalance ratio.
- Reviews numeric feature statistics.
- Visualizes high-variance feature distributions.
- Uses correlation heatmap to inspect redundancy.
- Uses boxplots to inspect extreme values and spread.

## 4) Feature Engineering and Preprocessing

- Encodes target labels using `LabelEncoder`.
- Removes low-variance features via `VarianceThreshold`.
- Selects top features using `SelectKBest(f_classif)` with CV-tuned `k`.
- Applies `log1p` to highly skewed non-negative features.
- Splits data with stratified train/test split.
- Scales features using `RobustScaler`.

## 5) Class Imbalance Handling

- Builds class-specific oversampling targets based on attack family names.
- Applies `SMOTETomek` with `SMOTE` for eligible minority classes.
- Uses safe neighbor selection (`k_neighbors`) based on class size.
- Optionally undersamples `BENIGN` class to cap majority dominance.

## 6) Model Training and Selection

- Trains two models:
  - `RandomForestClassifier`
  - `XGBClassifier`
- Computes per-model:
  - train score,
  - test score,
  - weighted F1 score.
- Selects the best model using weighted F1.

## 7) Evaluation Protocol

- Generates multiclass `classification_report`.
- Reports weighted accuracy, precision, recall, and F1.
- Computes ROC-AUC (multiclass OVR when applicable).
- Plots confusion matrix.
- Shows feature importance for tree-based models.

## 8) Inference Packaging

- Saves artifacts (`model`, `scaler`, `label_encoder`, `selected_features`) into `network_intrusion_model.pkl`.
- Provides `predict_flow(flow_data_dict)` helper for single-sample inference.

## 9) Methodological Notes

- The approach is **closed-set classification**: unseen attack families are mapped to the nearest known class.
- Weighted F1 drives model selection; rare-class recall should still be monitored from the class report and confusion matrix.

