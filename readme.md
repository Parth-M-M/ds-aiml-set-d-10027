```markdown
# Predict campaign responses and identify audience segments.
Parth mangroliya 10027

## Project Overview
This project focuses on analyzing customer data from `set_d.csv` to derive insights, build predictive models for customer response, and perform customer segmentation. It covers descriptive statistics, statistical inference, linear algebra for dimensionality reduction, data preprocessing, and the implementation and comparison of classification models (Logistic Regression and Artificial Neural Network).

## Data Dictionary
The dataset `set_d.csv` contains the following columns:
- `record_id`: Unique identifier for each customer record.
- `visits`: Number of visits (numeric, may contain NaNs).
- `recency`: Recency of last visit (numeric, may contain NaNs).
- `engagement`: Engagement score (numeric).
- `spend`: Spending amount (numeric).
- `group`: Customer group (categorical: 'G1', 'G2').
- `response`: Binary target variable (0 or 1), indicating customer response.

## Methodology Details

### Data Generation and Initial Setup
- **Generator Command:** The synthetic data was generated using `numpy` with `np.random.default_rng(404)`. Customer groups were assigned using `rng.choice(["G1", "G2"], size=n)`, and the binary response `y` was derived from a score calculation: `y = (score > 0).astype(int)`.
- **Cleaning Rules:**
    - Duplicates were removed using `df.drop_duplicates(keep='first')`.
    - Missing numerical values in `visits` and `recency` were imputed using `sklearn.impute.SimpleImputer(strategy='median')`.
- **Feature Formula:** An engineered feature, `visits_x_engagement`, was created by multiplying `visits` and `engagement`.
- **Split Seed and Partition Sizes:** Data was partitioned into fit, validation, and test sets using `sklearn.model_selection.train_test_split` with `random_state=42`. The partition sizes are:
    - Fit set: 180 records
    - Validation set: 60 records
    - Test set: 60 records

### Tools/Package Versions
The following key packages and their versions were used:
- `pandas==2.2.3`
- `numpy==2.1.3`
- `matplotlib==3.10.0`
- `seaborn==0.13.2`
- `scipy==1.16.3`
- `scikit-learn==1.6.1`
- `tensorflow==2.20.0`
- `joblib==1.6.0`

### Folder Map
- `/data/raw`: Stores the initial `set_d.csv` file.
- `/preprocessors`: Stores fitted preprocessing objects (`median_imputer.joblib`, `onehot_encoder.joblib`, `standard_scaler.joblib`).
- `/predictions`: Stores model predictions (`ann_test_predictions.csv`).
- `/models`: Stores trained machine learning models (`ann_model.h5`).
- `/`: Root directory, contains this `README.md` and other notebook-generated files (e.g., `test_predictions.csv`, `requirements.txt`).

### Exact Setup/Run Sequence from Repository Root
1.  Ensure Python 3.x is installed.
2.  Install required packages: `pip install -r requirements.txt` (This notebook generates a `requirements.txt`).
3.  Run the Jupyter/Colab notebook cells sequentially from top to bottom.

## Statistical Analysis

### Welch's t-test for Mean Engagement (G1 vs. G2)
- **Hypotheses:**
    - H0: There is no difference in the true mean engagement between group G1 and group G2. ($\\mu_{G1} = \\mu_{G2}$)
    - H1: There is a significant difference in the true mean engagement between group G1 and group G2. ($\\mu_{G1} \\neq \\mu_{G2}$)
- **Assumptions:** Welch's t-test assumes unequal variances between the two groups, which is a robust choice when this assumption is uncertain.
- **Key Output:** P-value: `0.859`. Since the p-value is much greater than the common significance level of 0.05, we fail to reject the null hypothesis. There is no statistically significant difference in mean engagement between groups G1 and G2.

### Confidence Interval for Overall Mean Engagement
- **Interpretation:** The 95% t-Confidence Interval for Overall Observed Mean Engagement is `(49.72, 52.08)`. This means we are 95% confident that the true population mean engagement falls within this range.

## Model Settings and Evaluation

### Model Settings
-   **Baseline Model (DummyClassifier):** `strategy='most_frequent'`, `random_state=42`. Predicts the most frequent class (class 1, as shown by `dummy_preds`).
-   **Main Classifier (LogisticRegression):** `max_iter=1000`, `random_state=42`. Uses default solver and C-value.
-   **ANN Model (Keras Sequential):**
    -   Input Layer: Matches the number of transformed features (7).
    -   Hidden Layer 1: 16 neurons, `relu` activation.
    -   Hidden Layer 2: 8 neurons, `relu` activation.
    -   Output Layer: 1 neuron, `sigmoid` activation (for binary classification).
    -   Compilation: `Adam` optimizer (learning_rate=0.001), `binary_crossentropy` loss, `accuracy` metric.
    -   Training: `batch_size=16`, `epochs=50`, with `EarlyStopping` (patience=5) monitoring `val_loss`. Trained for 34 epochs.

### Held-Out Comparison (Test Set F1-scores)
-   Logistic Regression F1-score (Class 1): `0.771`
-   ANN F1-score (Class 1): `0.743`

### Cluster Findings
-   The optimal number of clusters (`k`) chosen was `2`, based on the highest silhouette score of `0.219`.
-   **Cluster Profiles (Mean of Scaled Features):**
    -   **Cluster 0:** Generally higher scaled `visits`, `engagement`, `spend`, and `visits_x_engagement`; slightly lower `recency`.
    -   **Cluster 1:** Generally lower scaled `visits`, `engagement`, `spend`, and `visits_x_engagement`; slightly higher `recency`.
    -   This suggests Cluster 0 represents more active/engaged/spending customers, while Cluster 1 represents less active customers.

### Numerical Findings
1.  **Largest Eigenvalue Share:** The largest eigenvalue's share of the total variance for 'engagement' and 'visits' is `0.529`. This indicates that approximately 52.9% of the variability in these two features can be explained by a single principal component.
2.  **Best K-Means Silhouette Score:** The highest silhouette score observed during K-Means clustering was `0.219` for `k=2`.

### Recommendation with Limitation
-   **Model Preference:** Based on the F1-scores on the test set, the Logistic Regression model (`F1=0.771`) slightly outperforms the ANN model (`F1=0.743`) for predicting customer response. Given this marginal difference and Logistic Regression's inherent interpretability, it is recommended to use the Logistic Regression model for this prediction task.
-   **Limitation:** The dataset is synthetic and relatively small (300 unique records). This might limit the generalizability of the findings and the models' performance on real-world, larger, and more complex datasets. Further, the observed silhouette score for clustering is relatively low, suggesting the clusters might not be very distinct.

## Reconciliation
-   The test metrics reported for both Logistic Regression and ANN models are calculated based on predictions made on the exact same `test_ids`. The saved prediction files (`test_predictions.csv` and `predictions/ann_test_predictions.csv`) contain these predictions for the corresponding `record_id`s, ensuring consistency.

## Accessible Video URL and Duration
-  not  enough time 

## References
-   joblib 

## Declaration
All work is my own except where cited.
```
