# PaySim Fraud Detection

Detecting fraudulent mobile-money transactions in the PaySim dataset (6.3M transactions, 0.13% fraud). The project explores how fraud behaves, builds features from those findings, checks them for leakage, and compares Logistic Regression, Random Forest and XGBoost on imbalanced data. It ends with a threshold chosen by cost and a time-based test.

## Summary (for non-technical readers)

- **The problem:** only about 1 in 775 transactions is fraud. The bank's existing rule never raises a false alarm, but it catches only **0.2%** of fraud (16 of 8,213).
- **The model:** XGBoost uses only information the bank has *before* the money moves (transaction type, amount, the sender's balance, and the hour). It catches **99% of fraud** in the test set (1,626 of 1,643).
- **The trade-off:** at the default setting it raises about 3 false alarms for every fraud caught. That still affects less than 1% of normal customers.
- **Setting the alarm level is a business decision.** If a false alarm costs 50 (a staff check and an annoyed customer), the cheapest setting is 0.35. If a false alarm costs 500 the best setting moves to about 0.65, and at 5,000 it moves to about 0.9.
- **It works on future transactions:** trained on the first 80% of the month and tested on the last 20%, it still catches **97.5%** of fraud.
- **Lesson learned:** the first models scored a perfect 1.00. That turned out to be *leakage*, because some features used balances that are only known after the payment. I removed them, so the results above are realistic.

## Data

The dataset is not included in this repo because it is too large (about 470 MB).

1. Download it from Kaggle: [Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
2. Rename the CSV to `paysim.csv`
3. Put it in the same folder as the notebooks

## Notebooks

| Notebook | What it covers |
|---|---|
| [01_data_and_eda.ipynb](01_data_and_eda.ipynb) | Data quality, exploration (each section a question), the bank's baseline rule |
| [02_features_and_models.ipynb](02_features_and_models.ipynb) | Features, leakage check, Logistic Regression vs Random Forest vs XGBoost, confusion matrix, PR curve, cost-based threshold, feature importance, time-based split |

Steps across the two notebooks:

1. **Data quality**: size, types, missing values, duplicates, negative amounts (the data is clean)
2. **Exploration: how does fraud behave?**
   - How rare is fraud? (0.13%, so accuracy is the wrong metric)
   - Which transaction types have fraud? (only TRANSFER and CASH_OUT)
   - Are fraud amounts different? (median 441k vs 75k, but a lot of overlap)
   - Is there a 10M cap on fraud amounts? (287 frauds are exactly 10,000,000)
   - Do frauds come in pairs? (a TRANSFER, then a CASH_OUT of the same amount in the same hour)
   - Does fraud empty the account? (98% of frauds)
   - When does fraud happen? (highest rate at night, hours 0 to 6)
   - Which numeric columns are linked to fraud? (`oldbalanceOrig` is the strongest)
3. **Baseline**: the bank's `isFlaggedFraud` rule (precision 100%, recall 0.2%)
4. **Features**: split into "safe" (known before the payment: `is_transfer`, `log_amount`, `oldbalanceOrig`, `hour`) and after-transaction (`emptied`, `mismatch`, `zero_balance`); stratified split; RobustScaler fit on train only
5. **First models (all features)**: every model scores about 1.00, which is too good to be true
6. **Leakage check**: one feature at a time, then safe features only
7. **Realistic models**: Logistic Regression, Random Forest and XGBoost on the safe features, with class weights
8. **Evaluation**: confusion matrix, precision-recall curves, a threshold chosen by cost, feature importance
9. **Time-based split**: train on earlier steps, test on later ones

## Key findings

- Fraud only happens in TRANSFER and CASH_OUT transactions.
- Almost all frauds empty the origin account (98%). Most of the rest hit the 10M limit.
- Frauds come in pairs: the money is transferred out, then cashed out.
- Fraud is most common at night, when few normal customers are active.
- The built-in `isFlaggedFraud` rule is almost useless: it catches 0.2% of fraud.
- **Leakage:** with all 7 features, every model scores about 1.00. No single feature does this on its own: the best single feature, `zero_balance`, has an average precision of only 0.35. The three after-transaction features *together* give the answer away. With only the safe features, Logistic Regression drops to precision 0.71 / recall 0.20, so I use only the safe features from then on.

## Results

### Model comparison (safe features, class weights, random stratified split, threshold 0.5)

| Model | Precision | Recall | F1 | Avg precision |
|---|---|---|---|---|
| Logistic Regression (balanced) | 0.028 | 0.851 | 0.054 | 0.282 |
| Random Forest (balanced) | 0.762 | 0.777 | 0.770 | 0.854 |
| **XGBoost (weighted)** | 0.246 | **0.990** | 0.395 | **0.861** |
| *Bank rule (`isFlaggedFraud`)* | *1.000* | *0.002* | | |

**I chose XGBoost.** It has the highest average precision, so it ranks fraud best whatever the cut-off, and it catches 99% of fraud. Random Forest is close on average precision and more precise, but it misses more fraud (recall 0.78). Logistic Regression is far behind, probably because the fraud pattern is not a straight line.

### Confusion matrix (XGBoost, threshold 0.5)

- Frauds caught: **1,626 of 1,643** (17 missed)
- False alarms: **4,971**, about 3 for every fraud caught, out of about 552k normal transactions (less than 1%)

### Cost-based threshold

0.5 is only a default. I chose the threshold with the lowest total cost:
- **Missed fraud** = the bank loses the transaction amount
- **False alarm** = a fixed cost of 50

| Threshold | Precision | Recall | Total cost |
|---|---|---|---|
| 0.50 (default) | 0.246 | 0.990 | ≈2.54M |
| **0.35 (cheapest)** | 0.187 | 0.992 | **≈2.17M** |

**Sensitivity:** the cost of a false alarm is an assumption, so I tried other values. The cheapest threshold moves from **≈0.35** to **≈0.65** to **≈0.9** when a false alarm costs **50 / 500 / 5,000**. The more expensive an alarm is (for example staff time or customers leaving), the higher the threshold should be, so this is a business decision.

### Feature importance (XGBoost)

| Feature | Importance |
|---|---|
| `oldbalanceOrig` | 0.41 |
| `log_amount` | 0.29 |
| `is_transfer` | 0.16 |
| `hour` | 0.14 |

This matches the exploration: fraud targets accounts with money in them, fraud amounts are bigger, transfers are riskier, and fraud peaks at night. Importance shows how much the model *uses* a feature, not whether it raises or lowers the risk. SHAP would show that.

### Time-based split

A random split mixes the whole month, so the model partly "sees the future" and paired frauds can land on both sides of the split. Here I trained on the first 80% of steps and tested on the last 20% (same XGBoost model, same safe features, threshold 0.5).

| Split | Precision | Recall | Avg precision |
|---|---|---|---|
| Random (stratified) | 0.246 | 0.990 | 0.861 |
| Time-based | 0.430 | 0.975 | 0.886 |

Recall stays very high, so the model still catches almost all fraud in later transactions it has never seen. Precision and average precision are higher, probably because the later steps have a bigger share of fraud, so the two results are not directly comparable. The time split is the more realistic test, because a bank always predicts future transactions.

## Future work

- **Cross-validation and tuning:** 5-fold `TimeSeriesSplit` (each fold trains on the past and tests on the future) with `RandomizedSearchCV` and `scoring="average_precision"`
- **Resampling:** SMOTE on the training set only, compared with class weights
- **Probability calibration:** so predicted probabilities match real fraud rates and the cost-based threshold is more reliable
- **SHAP:** explain single predictions, not just overall importance
- **MLflow:** log each model's settings and scores to compare experiments

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
jupyter notebook
```
