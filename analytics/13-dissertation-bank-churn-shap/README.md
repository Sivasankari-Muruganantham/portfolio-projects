# Predicting and Explaining Customer Churn in Retail Banking

**A Machine Learning Approach with SHAP-Driven Retention Insights**

MSc Business Analytics dissertation · Queen's University Belfast · 2025–2026

---

## Overview

Retail banks usually see a customer's decision to leave only after it has been made. This project builds and compares five machine learning models to predict churn, explains the best model with SHAP, and turns the explanations into retention recommendations checked against the UK FCA Consumer Duty outcomes.

It follows a **predict → explain → recommend** sequence, with predictor choice and hypotheses grounded in **signalling theory** (Spence, 1973) and stated before the analysis.

## Research questions

1. **RQ1:** How accurately can churn be predicted, and which of five algorithms performs best?
2. **RQ2:** Which characteristics and behavioural signals drive predictions, and do the SHAP patterns support the theory-derived hypotheses?
3. **RQ3:** How can SHAP explanations be translated into retention recommendations that are actionable and Consumer-Duty-consistent?

## Data

- **Bank Customer Churn Records** (Kaggle, user Radheshyamkollipara): 10,000 customers, 18 variables, no missing values, customers from France, Germany and Spain
- Target `Exited`: 20.4% churned
- The dataset's own documentation describes the records as fictional/anonymised, and the collecting institution is undocumented. It is treated as data of unverified provenance, and the UK regulatory framing is **illustrative, not validated for UK customers**.

## Method

- CRISP-DM workflow, positivist quantitative design
- Stratified 80/20 train/test split with a fixed random seed
- SMOTE applied **inside each cross-validation fold, on training data only**; the test set is never resampled and is used once
- Five models: Logistic Regression, Decision Tree, Random Forest, SVM (radial kernel), XGBoost
- XGBoost tuned over a 27-configuration grid (`max_depth` × `eta` × `nrounds`) with 5-fold CV
- Recall-weighted classification threshold chosen by F-beta (β = 2) on out-of-fold training predictions
- SHAP (exact TreeSHAP) for global and local explanation
- Subgroup fairness audit by gender and age band

## Key results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.735 | 0.394 | 0.560 | 0.463 | 0.759 |
| Decision Tree | 0.842 | 0.615 | 0.604 | 0.610 | 0.828 |
| Random Forest | 0.849 | 0.672 | 0.509 | 0.579 | 0.851 |
| SVM (radial) | 0.824 | 0.570 | 0.560 | 0.565 | 0.814 |
| XGBoost (0.5 cutoff) | 0.856 | 0.698 | 0.516 | 0.593 | 0.851 |

*Test set, default cutoff, primary (Complain-excluded) data.*

- **XGBoost selected** (AUC 0.851, tied with Random Forest) for its exact compatibility with TreeSHAP.
- At the recall-weighted threshold of **0.16**, XGBoost recovers **83.8% of churners** (recall 0.838) at 39.1% precision.
- **Target leakage found:** including the `Complain` field pushes every model to AUC ≥ 0.998, including plain logistic regression. This is evidence of leakage rather than genuine predictive strength, so the field is excluded from the primary analysis and the sensitivity analysis makes the cost visible.

### What drives churn (SHAP)

| Hypothesis | Outcome |
|---|---|
| **H1:** inactive members carry higher churn risk | Supported |
| **H2:** product holding is non-linear | Strongly supported: two products is the safest holding, with sharp risk escalation from three upward |
| **H3:** age is non-linear | Supported with nuance: risk rises through most of the age range and softens in the oldest decile |
| **H4:** behavioural signals outrank demographics | Partially supported: behavioural signals outrank financial demographics, but not Age or Gender |

The top three features (Age, NumOfProducts, IsActiveMember) account for 53.2% of total SHAP importance. Partial support for H4 suggests the characteristics-versus-signals distinction is too coarse and that financial and identity/life-stage demographics should be separated.

### Recommendations

- Review product suitability for customers holding **three or more products** instead of treating a large holding as loyalty.
- Run **targeted re-engagement** for members who become inactive.
- No complaint-triggered recommendation is made, because of the leakage concern. Age is diagnostic only and is not an intervention target.

### Fairness audit

At the 0.16 threshold, recall is higher for female customers (0.878 vs 0.787) and for customers aged 60+ (0.968 vs 0.827), but false-positive rates are higher for the same groups. The 60+ band has only 116 test customers, so its estimates are indicative only.

## Limitations

- Non-UK, fictional/anonymised data of uncertain provenance
- AUC differences between models are point estimates from a single 5-fold CV
- The threshold is chosen from out-of-fold training predictions rather than a separate validation set
- Static snapshot, so it cannot show where churned customers go

## Tools

R (xgboost, SHAPforxgboost, SMOTE-based resampling, tidyverse) <!-- confirm and edit: add Python if used for any SHAP plots -->

## Repository contents

<!-- Edit to match the files you upload, for example: -->
- `code/`: analysis scripts
- `figures/`: ROC curve, SHAP summary and dependence plots, churn-rate charts
- `report/`: dissertation PDF (only if publication is permitted)

## Reference

Muruganantham, S. (2026) *Predicting and Explaining Customer Churn in Retail Banking: A Machine Learning Approach with SHAP-Driven Retention Insights*. MSc Research Report, Queen's University Belfast.
