# Fraud Transaction Classifier (TensorFlow + scikit‑learn)

A compact, end‑to‑end pipeline to detect fraudulent transactions using a feed‑forward neural network. It covers data loading, feature engineering, scaling, class imbalance handling, training with early stopping, evaluation, and threshold tuning via the precision–recall curve.

---

## ✨ Highlights

* **Feature engineering**: log‑transform of amounts, per‑customer ratio features, and first‑seen indicators for category/merchant.
* **Robust preprocessing**: string cleanup + one‑hot encoding + `StandardScaler`.
* **Class imbalance aware**: `compute_class_weight('balanced', ...)`.
* **Neural network**: 128 → 64 → 32 → 1 (sigmoid), Adam (1e‑3).
* **Model selection**: early stopping on validation recall (high‑recall focus).
* **Threshold tuning**: choose decision thresholds from the precision–recall curve (e.g., \~0.9485 for \~70% precision in the provided run).

---

## 📦 Expected Columns in `Dataset.csv`

The script expects (at minimum) the following columns:

* **Identifiers & categorical**: `customer`, `merchant`, `age`, `gender`, `category`
* **Numeric**: `amount`
* **Optional / dropped**: `zipcodeOri`, `zipMerchant`, `step` (time index; engineered time features are *disabled* in this version)
* **Target**: `fraud` (binary: 0 = legitimate, 1 = fraud)

> The code cleans stray apostrophes in `age`, `gender`, `category` (e.g., `M'` → `M`).

---

## 🛠️ Setup

```bash
# Python 3.10+ recommended
pip install -U tensorflow scikit-learn pandas numpy matplotlib
```

Check TensorFlow version (printed at runtime). The sample run used **TensorFlow 2.18.0**.

---

## 🚀 How to Run

1. Place your data at `/content/Dataset.csv` (default path in the script). If running locally, update `file_path` accordingly.
2. Execute the script (e.g., in Colab or locally):

   ```bash
   python train_fraud_classifier.py
   ```
3. Training prints metrics each epoch and finishes with a classification report. A second block plots the precision–recall curve and finds a threshold that meets a precision target (70% by default in the example).

---

## 🧪 Train/Val Split & Scaling

* Stratified split: **80/20** (`random_state=42`).
* Features are scaled with `StandardScaler` fitted on **train** and applied to **val**.

> Note: The current script doesn’t persist the scaler or model—see **Production Tips** for saving.

---

## 🧱 Feature Engineering

* `amount_log = log1p(amount)` reduces skew.
* `amount_vs_avg_ratio`: for each transaction, ratio to the customer’s mean log‑amount.
* `is_new_category`: 1 if this is the first time the customer uses a given category.
* `is_new_merchant`: 1 if this is the first time the customer uses a given merchant.
* One‑hot encode: `age`, `gender`, `merchant`, `category`.
* Drop non‑predictive/leaky columns: `customer`, `amount`, `zipcodeOri`, `zipMerchant`.
* **Exclude** `step` from final features (kept in dataframe but dropped before modeling).

---

## 🧠 Model

```text
Input → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
Optimizer: Adam(lr=1e-3)
Loss: Binary Cross‑Entropy
Metrics: Accuracy, Precision, Recall
Class Weights: from sklearn.compute_class_weight("balanced")
```

Early stopping monitors validation recall to protect minority‑class performance:

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_recall',  # see note below
    patience=5,
    restore_best_weights=True,
    mode='max'
)
```

> **Keras naming note:** In the sample logs, metrics appeared as `precision_1` / `recall_1`, leading to the warning:
> `Early stopping conditioned on metric 'val_recall' which is not available ...`
>
> **Fix:** Name the metrics explicitly so the monitor exists:
>
> ```python
> metrics=[
>   'accuracy',
>   tf.keras.metrics.Precision(name='precision'),
>   tf.keras.metrics.Recall(name='recall')
> ]
> # and monitor='val_recall'
> ```

---

## 📈 Results (from the included run)

**Validation set (107,036 samples):**

* Overall accuracy: **\~0.99**
* Fraud class: precision **0.68–0.70**, recall **0.85–0.90** (depending on threshold)

At a tuned threshold **\~0.9485** (picked to reach \~70% precision):

* Legitimate: P=1.00, R=1.00
* Fraud: **P=0.70, R=0.85**, F1=0.77

> Thresholds are business‑dependent. Use the precision–recall curve to settle on an operating point that fits costs and SLAs.

---

## 🎯 Threshold Tuning

The script computes:

```python
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
```

Then selects the first index where `precision >= 0.70` and reports the corresponding threshold (≈ **0.9485** in the sample). Apply at inference via:

```python
y_pred = (y_pred_proba > chosen_threshold).astype(int)
```

---

## 🧯 Class Imbalance Handling

We compute class weights with:

```python
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weights[i] for i in range(2)}
```

This up‑weights the minority (fraud) class during training. You can also try:

* Focal loss
* Oversampling (SMOTE) or undersampling
* Threshold moving (already included)

---

## 🧪 Reproducibility Tips

* Set seeds for NumPy, TensorFlow, and Python `random` if exact repeatability is required.
* Log the `train_test_split` indices.
* Persist artifacts (model + scaler) for consistent inference.

---

## 🏭 Production Tips

* **Save model & scaler:**

  ```python
  import joblib
  classifier.save('fraud_model.keras')
  joblib.dump(scaler, 'scaler.joblib')
  ```
* **Inference pipeline:** apply the **same** preprocessing and `StandardScaler` as training.
* **Monitoring:** track drift, class prevalence, precision/recall at the chosen threshold.
* **Calibration:** consider Platt scaling / isotonic if you rely on probabilities.

---

## 📂 Suggested Repo Structure

```
.
├── train_fraud_classifier.py
├── README.md
├── requirements.txt
└── data/
    └── Dataset.csv
```

`requirements.txt` (minimal):

```
tensorflow
scikit-learn
pandas
numpy
matplotlib
```

---

## 🔧 Common Gotchas

* **EarlyStopping warning**: name metrics (see note above) or change `monitor` to the actual metric name printed in logs (e.g., `val_recall_1`).
* **Memory usage** with many one‑hot columns (e.g., `merchant`): consider hashing trick, target encoding, or embeddings.
* **Colab path**: update `file_path` if you’re not in `/content/`.

---

## 🗺️ Roadmap / Nice‑to‑Haves

* Entity embeddings for high‑cardinality categorical features
* Temporal features (re‑enable and refine `step`‑based features)
* Hyperparameter tuning (KerasTuner/Optuna)
* Calibration and cost‑sensitive evaluation
* Model explainability (SHAP) & feature drift monitoring

---

## 📜 License

MIT .

---

## 🙌 Acknowledgments

* TensorFlow & scikit‑learn teams
* The many public fraud‑detection datasets and papers that inspired common feature pattern
