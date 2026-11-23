# ==================================================================
# Predicting and Explaining Customer Churn in Retail Banking:
# A Machine Learning Approach with SHAP-Driven Retention Insights
# MSc Business Analytics dissertation, Queen's University Belfast
#
# Full pipeline: EDA, data preparation, fold-wise SMOTE, five-algorithm
# training, XGBoost hyperparameter grid search, out-of-fold threshold
# selection, final test-set evaluation, subgroup fairness audit and
# SHAP explanation, run for two data variants:
#   - primary     (Complain excluded)
#   - sensitivity (Complain included)
#
# Input:  Customer-Churn-Records.csv (Kaggle "Bank Customer Churn Records",
#         user Radheshyamkollipara). Download it and place it in the
#         working directory. The dataset is not redistributed here.
# Output: outputs/plots, outputs/tables, outputs/models,
#         outputs/session_info.txt
#
# Note: section numbers in comments refer to the Research Report.
# ==================================================================

library(tidyverse)
library(ggplot2)
library(dplyr)
library(caret)
library(corrplot)
library(gridExtra)
library(scales)
library(pROC)
library(smotefamily)
library(xgboost)
library(randomForest)
library(e1071)
library(rpart)
library(SHAPforxgboost)
library(data.table)
library(fastDummies)

set.seed(40485762) # fixed seed, reported for reproducibility (Section 3.6)

BETA <- 2 # F-beta weighting for threshold selection (Section 3.8):
          # beta > 1 weights recall more heavily than precision,
          # reflecting the argument that a missed churner is
          # roughly twice as costly as an unnecessary retention
          # contact.

dir.create("outputs/plots", showWarnings = FALSE, recursive = TRUE)
dir.create("outputs/tables", showWarnings = FALSE, recursive = TRUE)
dir.create("outputs/models", showWarnings = FALSE, recursive = TRUE)
stopifnot(dir.exists("outputs/plots"), dir.exists("outputs/tables"), dir.exists("outputs/models"))

# ---- reproducibility record: seed and full package/version environment ----
sink("outputs/session_info.txt")
cat("Random seed:", 40485762, "\n\n")
print(sessionInfo())
sink()

# ==================================================================
# PART A: EXPLORATORY DATA ANALYSIS AND PRELIMINARY DATA QUALITY
# ==================================================================
df <- read.csv("Customer-Churn-Records.csv")
dim(df)
head(df)
str(df)
summary(df)

# Missing values
colSums(is.na(df))

# Duplicate check
sum(duplicated(df))
sum(duplicated(df$CustomerId))

# Target variable distribution
table(df$Exited)
prop.table(table(df$Exited)) * 100

ggplot(df, aes(x = factor(Exited), fill = factor(Exited))) +
  geom_bar() +
  scale_fill_manual(values = c("#2E75B6", "#C00000"), labels = c("Retained", "Churned")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Customer Churn Distribution", subtitle = "Retail Banking Dataset - 10,000 customers",
       x = "Customer Status", y = "Count", fill = "Status") +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5, size = 5) +
  theme_minimal() +
  theme(plot.title = element_text(size = 15, face = "bold"), plot.subtitle = element_text(size = 11))

ggplot(df, aes(x = Geography, fill = factor(Exited))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("#2E75B6", "#C00000"), labels = c("Retained", "Churned")) +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Churn Rate by Geography", x = "Country", y = "Percentage", fill = "Status") +
  theme_minimal() + theme(plot.title = element_text(size = 16, face = "bold"))

# Age-group chart is descriptive only; AgeGroup is not passed to any model.
df_eda <- df %>% mutate(AgeGroup = cut(Age, breaks = c(0, 30, 40, 50, 60, 100),
                                       labels = c("Under 30", "30-40", "40-50", "50-60", "Over 60")))
ggplot(df_eda, aes(x = AgeGroup, fill = factor(Exited))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("#2E75B6", "#C00000"), labels = c("Retained", "Churned")) +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Churn Rate by Age Group", x = "Age Group", y = "Percentage", fill = "Status") +
  theme_minimal() + theme(plot.title = element_text(size = 16, face = "bold"))

ggplot(df, aes(x = factor(NumOfProducts), fill = factor(Exited))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("#2E75B6", "#C00000"), labels = c("Retained", "Churned")) +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Churn Rate by Number of Products", x = "Number of Products", y = "Percentage", fill = "Status") +
  theme_minimal() + theme(plot.title = element_text(size = 16, face = "bold"))

ggplot(df, aes(x = factor(Complain), fill = factor(Exited))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("#2E75B6", "#C00000"), labels = c("Retained", "Churned")) +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Churn Rate by Complaint History", x = "Complaint Filed (0 = No, 1 = Yes)",
       y = "Percentage", fill = "Status") +
  theme_minimal() + theme(plot.title = element_text(size = 16, face = "bold"))

ggplot(df, aes(x = Gender, fill = factor(Exited))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("#2E75B6", "#C00000"), labels = c("Retained", "Churned")) +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Churn Rate by Gender", x = "Gender", y = "Percentage", fill = "Status") +
  theme_minimal() + theme(plot.title = element_text(size = 16, face = "bold"))

numeric_cols <- df %>% select(CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard,
                              IsActiveMember, EstimatedSalary, Complain, Satisfaction.Score,
                              Point.Earned, Exited)
cor_matrix <- cor(numeric_cols)
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8,
         title = "Correlation Matrix - Bank Churn Dataset", mar = c(0, 0, 1, 0),
         col = colorRampPalette(c("#C00000", "White", "#2E75B6"))(200))

# Outlier / anomaly check - zero-balance proportion
zero_balance <- df %>% filter(Balance == 0) %>% nrow()
zero_balance
round(zero_balance / nrow(df) * 100, 2)

p1 <- ggplot(df, aes(x = factor(Exited), y = Age, fill = factor(Exited))) + geom_boxplot() +
  scale_fill_manual(values = c("#2E75B6", "#C00000"), labels = c("Retained", "Churned")) +
  labs(title = "Age by Churn Status", x = "Customer Status", y = "Age", fill = "Status") + theme_minimal()
p2 <- ggplot(df, aes(x = factor(Exited), y = Balance, fill = factor(Exited))) + geom_boxplot() +
  scale_fill_manual(values = c("#2E75B6", "#C00000"), labels = c("Retained", "Churned")) +
  labs(title = "Balance by Churn Status", x = "Customer Status", y = "Balance", fill = "Status") + theme_minimal()
p3 <- ggplot(df, aes(x = factor(Exited), y = CreditScore, fill = factor(Exited))) + geom_boxplot() +
  scale_fill_manual(values = c("#2E75B6", "#C00000"), labels = c("Retained", "Churned")) +
  labs(title = "Credit Score by Churn Status", x = "Customer Status", y = "Credit Score", fill = "Status") +
  theme_minimal()
grid.arrange(p1, p2, p3, ncol = 3)

# Chi-square tests and Cramer's V effect sizes for Geography and Gender
# against Exited (reported in Section 3.4/3.5).
chisq_geo <- chisq.test(table(df$Geography, df$Exited)); chisq_geo
chisq_gender <- chisq.test(table(df$Gender, df$Exited)); chisq_gender
cramers_v <- function(tbl) {
  chi2 <- chisq.test(tbl)$statistic
  n <- sum(tbl)
  k <- min(nrow(tbl), ncol(tbl))
  sqrt(chi2 / (n * (k - 1)))
}
cramers_v(table(df$Geography, df$Exited))
cramers_v(table(df$Gender, df$Exited))

# Class imbalance - explicit framing for SMOTE justification (Section 3.6)
imbalance_table <- prop.table(table(df$Exited)) * 100
imbalance_table
cat(sprintf("Class imbalance: %.1f%% retained vs %.1f%% churned - supports use of SMOTE for model training.\n",
            imbalance_table[1], imbalance_table[2]))

mean(df$Age); sd(df$Age) # reported for unscaling Age in SHAP dependence plots

# ==================================================================
# PART B: DATA PREPARATION (Section 3.5)
# Explicit allow-list of modelling variables, so no exploratory or
# derived column from Part A (e.g. AgeGroup) is carried into modelling.
# ==================================================================
model_df <- df %>%
  select(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts,
         HasCrCard, IsActiveMember, EstimatedSalary, Exited,
         Complain, Satisfaction.Score, Card.Type, Point.Earned)

prep_dataset <- function(data, drop_complain = TRUE) {
  d <- data
  if (drop_complain) d <- d %>% select(-Complain)
  d <- d %>% mutate(Gender = ifelse(Gender == "Male", 1, 0))
  d <- dummy_cols(d,
                  select_columns = c("Geography", "Card.Type"),
                  remove_first_dummy = TRUE,
                  remove_selected_columns = TRUE)
  d$Exited <- factor(d$Exited, levels = c(0, 1), labels = c("Retained", "Churned"))
  d
}

# Reference categories for the one-hot-encoded factors (reported in Section 3.5):
levels(factor(df$Geography)); levels(factor(df$Card.Type))

run_pipeline <- function(data_variant_name, drop_complain) {

  # Re-seed on every call so each variant reproduces identically regardless
  # of call order (see Technical Report Section 2.10).
  set.seed(40485762)

  cat("\n\n==================== RUNNING VARIANT:", data_variant_name, "====================\n")
  primary_df <- prep_dataset(model_df, drop_complain = drop_complain)

  ## ---- Stratified train/test split (Section 3.5) ----
  train_idx <- createDataPartition(primary_df$Exited, p = 0.8, list = FALSE)
  train_raw <- primary_df[train_idx, ]
  test_raw  <- primary_df[-train_idx, ]
  prop.table(table(train_raw$Exited)); prop.table(table(test_raw$Exited))

  ## ---- Scaling - fitted on the training partition only, then applied
  ## to both partitions, so no test-set statistic informs the scaling ----
  numeric_vars <- intersect(
    c("CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
      "EstimatedSalary", "Satisfaction.Score", "Point.Earned"),
    names(train_raw)
  )
  preproc <- preProcess(train_raw[, numeric_vars], method = c("center", "scale"))
  train_scaled <- train_raw
  test_scaled  <- test_raw
  train_scaled[, numeric_vars] <- predict(preproc, train_raw[, numeric_vars])
  test_scaled[, numeric_vars]  <- predict(preproc, test_raw[, numeric_vars])

  ## ================================================================
  ## FOLD-WISE SMOTE (Section 3.5)
  ## caret's `sampling` argument in trainControl applies the supplied
  ## function to each fold's own training portion only, during
  ## resampling, and to the full training set for the final refit -
  ## never to the assessment/held-out fold. This is what keeps every
  ## synthetic observation derived only from data the model is not
  ## subsequently validated against.
  ## ================================================================
  smote_sampling_func <- function(x, y) {
    dtemp <- as.data.frame(x)
    dtemp[] <- lapply(dtemp, as.numeric)
    sm <- smotefamily::SMOTE(X = dtemp, target = y, K = 5, dup_size = 0)
    out_x <- sm$data[, -ncol(sm$data), drop = FALSE]
    out_y <- factor(sm$data[[ncol(sm$data)]], levels = levels(y))
    list(x = out_x, y = out_y)
  }

  ctrl <- trainControl(
    method = "cv", number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary,
    savePredictions = "final",
    sampling = list(name = "SMOTE (smotefamily, fold-wise)",
                    func = smote_sampling_func, first = TRUE)
  )

  ## ---- hyperparameter candidate grids (Section 3.6 / Table 4.1) ----
  grid_dt  <- expand.grid(cp = c(0.001, 0.005, 0.01, 0.02, 0.05))
  grid_rf  <- expand.grid(mtry = c(2, 4, 6, 8))
  grid_svm <- expand.grid(sigma = c(0.01, 0.05, 0.1), C = c(0.5, 1, 2))

  cat("Training Logistic Regression...\n")
  model_lr <- train(Exited ~ ., data = train_scaled, method = "glm",
                    family = "binomial", trControl = ctrl, metric = "ROC")
  cat("Training Decision Tree...\n")
  model_dt <- train(Exited ~ ., data = train_scaled, method = "rpart",
                    trControl = ctrl, metric = "ROC", tuneGrid = grid_dt)
  cat("Training Random Forest...\n")
  model_rf <- train(Exited ~ ., data = train_scaled, method = "rf",
                    trControl = ctrl, metric = "ROC", tuneGrid = grid_rf)
  cat("Training SVM (radial)...\n")
  model_svm <- train(Exited ~ ., data = train_scaled, method = "svmRadial",
                     trControl = ctrl, metric = "ROC", tuneGrid = grid_svm)
  # train_scaled here is the unbalanced training data throughout - caret
  # applies SMOTE internally, per fold and for the final refit, via `ctrl`.

  ## ================================================================
  ## XGBOOST - manual fold-wise cross-validation (native training, so
  ## caret's `sampling` mechanism does not apply and is replicated by
  ## hand below). This loop also produces the out-of-fold (OOF)
  ## predictions used for threshold selection below.
  ##
  ## Maximum tree depth, learning rate and number of boosting rounds are
  ## tuned jointly as three dimensions of a single grid search, evaluated
  ## on identical terms across the same five folds - see Section 3.6 for
  ## the methodological rationale for treating boosting rounds this way
  ## rather than via an early-stopping rule.
  ## ================================================================
  xgb_grid <- expand.grid(max_depth = c(3, 4, 6), eta = c(0.05, 0.1, 0.2),
                          nrounds = c(100, 250, 500))
  folds <- createFolds(train_scaled$Exited, k = 5, list = TRUE)
  to_numeric_matrix <- function(d) as.matrix(d %>% select(-Exited) %>% mutate(across(everything(), as.numeric)))

  xgb_cv_search <- data.frame()
  oof_probs_by_config <- list()

  for (g in seq_len(nrow(xgb_grid))) {
    params <- list(
      objective = "binary:logistic", eval_metric = "auc",
      max_depth = xgb_grid$max_depth[g], eta = xgb_grid$eta[g],
      subsample = 0.8, colsample_bytree = 0.8
    )
    nrounds_g <- xgb_grid$nrounds[g]
    fold_aucs <- c()
    oof_probs <- rep(NA_real_, nrow(train_scaled))

    for (f in seq_along(folds)) {
      val_idx <- folds[[f]]
      fold_train <- train_scaled[-val_idx, ]
      fold_val   <- train_scaled[val_idx, ]

      sm <- smote_sampling_func(fold_train %>% select(-Exited), fold_train$Exited)
      fold_train_bal <- sm$x %>% mutate(Exited = sm$y)

      dtr <- xgb.DMatrix(data = to_numeric_matrix(fold_train_bal),
                         label = ifelse(fold_train_bal$Exited == "Churned", 1, 0))
      dval <- xgb.DMatrix(data = to_numeric_matrix(fold_val),
                          label = ifelse(fold_val$Exited == "Churned", 1, 0))

      fit <- xgb.train(params = params, data = dtr, nrounds = nrounds_g, verbose = 0)
      val_pred <- predict(fit, dval)
      oof_probs[val_idx] <- val_pred
      fold_aucs <- c(fold_aucs, as.numeric(auc(roc(fold_val$Exited, val_pred,
                                                   levels = c("Retained", "Churned"),
                                                   quiet = TRUE))))
    }
    xgb_cv_search <- bind_rows(xgb_cv_search, data.frame(
      max_depth = xgb_grid$max_depth[g], eta = xgb_grid$eta[g], nrounds = nrounds_g,
      mean_cv_auc = mean(fold_aucs)
    ))
    oof_probs_by_config[[g]] <- oof_probs
  }
  print(xgb_cv_search)

  best_config_idx <- which.max(xgb_cv_search$mean_cv_auc)
  best_params <- list(
    objective = "binary:logistic", eval_metric = "auc",
    max_depth = xgb_cv_search$max_depth[best_config_idx],
    eta = xgb_cv_search$eta[best_config_idx],
    subsample = 0.8, colsample_bytree = 0.8
  )
  best_nrounds <- xgb_cv_search$nrounds[best_config_idx]
  oof_probs_final <- oof_probs_by_config[[best_config_idx]]
  cat("Best XGBoost configuration (by mean 5-fold CV AUC):\n")
  print(xgb_cv_search[best_config_idx, ])

  ## ---- Final refit: SMOTE applied once to the full training set (no
  ## cross-validation happens at this step, so a single pass here
  ## introduces no leakage) ----
  sm_full <- smote_sampling_func(train_scaled %>% select(-Exited), train_scaled$Exited)
  train_balanced_full <- sm_full$x %>% mutate(Exited = sm_full$y)
  dtrain_full <- xgb.DMatrix(data = to_numeric_matrix(train_balanced_full),
                             label = ifelse(train_balanced_full$Exited == "Churned", 1, 0))
  model_xgb <- xgb.train(params = best_params, data = dtrain_full, nrounds = best_nrounds)

  ## ================================================================
  ## THRESHOLD SELECTION FROM OUT-OF-FOLD PREDICTIONS ONLY (Section 3.7)
  ## Uses oof_probs_final (training-partition, out-of-fold predictions
  ## only) - the test set is not touched until the evaluation step below.
  ## ================================================================
  y_train_num <- ifelse(train_scaled$Exited == "Churned", 1, 0)
  thresholds <- seq(0.05, 0.95, by = 0.01)
  fbeta_at_threshold <- sapply(thresholds, function(t) {
    pred_t <- ifelse(oof_probs_final > t, 1, 0)
    tp <- sum(pred_t == 1 & y_train_num == 1)
    fp <- sum(pred_t == 1 & y_train_num == 0)
    fn <- sum(pred_t == 0 & y_train_num == 1)
    precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
    recall    <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
    if (precision == 0 && recall == 0) return(0)
    (1 + BETA^2) * precision * recall / ((BETA^2 * precision) + recall)
  })
  best_threshold <- thresholds[which.max(fbeta_at_threshold)]
  cat("F-beta (beta =", BETA, ") optimal threshold, chosen from training-partition OOF predictions:",
      best_threshold, "\n")

  ## ================================================================
  ## FINAL EVALUATION - the test set is touched exactly once, here
  ## ================================================================
  evaluate_model <- function(model, test_data, model_name) {
    preds <- predict(model, test_data)
    probs <- predict(model, test_data, type = "prob")[, "Churned"]
    cm <- confusionMatrix(preds, test_data$Exited, positive = "Churned", mode = "everything")
    roc_obj <- roc(response = test_data$Exited, predictor = probs,
                   levels = c("Retained", "Churned"), quiet = TRUE)
    data.frame(Model = model_name, Accuracy = cm$overall["Accuracy"],
               Precision = cm$byClass["Precision"], Recall = cm$byClass["Recall"],
               F1 = cm$byClass["F1"], AUC = as.numeric(auc(roc_obj)))
  }

  dtest <- xgb.DMatrix(data = to_numeric_matrix(test_scaled),
                       label = ifelse(test_scaled$Exited == "Churned", 1, 0))
  xgb_probs_test <- predict(model_xgb, dtest)
  xgb_preds_default <- factor(ifelse(xgb_probs_test > 0.5, "Churned", "Retained"),
                              levels = c("Retained", "Churned"))
  cm_default <- confusionMatrix(xgb_preds_default, test_scaled$Exited,
                                positive = "Churned", mode = "everything")
  roc_xgb <- roc(test_scaled$Exited, xgb_probs_test, levels = c("Retained", "Churned"), quiet = TRUE)
  xgb_preds_opt <- factor(ifelse(xgb_probs_test > best_threshold, "Churned", "Retained"),
                          levels = c("Retained", "Churned"))
  cm_opt <- confusionMatrix(xgb_preds_opt, test_scaled$Exited, positive = "Churned", mode = "everything")

  results <- bind_rows(
    evaluate_model(model_lr,  test_scaled, "Logistic Regression"),
    evaluate_model(model_dt,  test_scaled, "Decision Tree"),
    evaluate_model(model_rf,  test_scaled, "Random Forest"),
    evaluate_model(model_svm, test_scaled, "SVM"),
    data.frame(Model = "XGBoost (default cutoff 0.5)", Accuracy = cm_default$overall["Accuracy"],
               Precision = cm_default$byClass["Precision"], Recall = cm_default$byClass["Recall"],
               F1 = cm_default$byClass["F1"], AUC = as.numeric(auc(roc_xgb))),
    data.frame(Model = paste0("XGBoost (threshold-optimised, cutoff = ", best_threshold, ")"),
               Accuracy = cm_opt$overall["Accuracy"], Precision = cm_opt$byClass["Precision"],
               Recall = cm_opt$byClass["Recall"], F1 = cm_opt$byClass["F1"],
               AUC = as.numeric(auc(roc_xgb)))
  )
  print(results)

  ## ================================================================
  ## SUBGROUP FAIRNESS AUDIT (Section 3.8) - final model and frozen
  ## threshold, test set only, split by gender and by age band.
  ## ================================================================
  audit_df <- test_scaled %>%
    mutate(pred = xgb_preds_opt, truth = Exited,
           GenderLabel = ifelse(Gender == max(Gender), "Male", "Female"))
  age_mean <- preproc$mean["Age"]; age_sd <- preproc$std["Age"]
  audit_df$AgeBand <- ifelse((audit_df$Age * age_sd + age_mean) >= 60, "60+", "Under 60")

  subgroup_metrics <- function(d, group_col) {
    d %>% group_by(.data[[group_col]]) %>% summarise(
      n = n(),
      recall = sum(pred == "Churned" & truth == "Churned") / sum(truth == "Churned"),
      precision = ifelse(sum(pred == "Churned") == 0, NA, sum(pred == "Churned" & truth == "Churned") / sum(pred == "Churned")),
      false_positive_rate = sum(pred == "Churned" & truth == "Retained") / sum(truth == "Retained"),
      .groups = "drop"
    )
  }
  fairness_gender <- subgroup_metrics(audit_df, "GenderLabel")
  fairness_age    <- subgroup_metrics(audit_df, "AgeBand")
  print(fairness_gender); print(fairness_age)

  ## ---- SHAP explanation (Section 3.6): TreeSHAP via SHAPforxgboost,
  ## applied to the final threshold-optimised XGBoost model; the
  ## background/explanation dataset is the SMOTE-balanced training
  ## partition used to fit that final model. ----
  xgb_train_matrix <- to_numeric_matrix(train_balanced_full)
  shap_values <- shap.values(xgb_model = model_xgb, X_train = xgb_train_matrix)
  shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train = xgb_train_matrix)

  ## ---- diagnostic plots ----
  png(sprintf("outputs/plots/shap_summary_%s.png", data_variant_name), width = 8, height = 6, units = "in", res = 300)
  print(shap.plot.summary(shap_long))
  dev.off()

  png(sprintf("outputs/plots/shap_dependence_numproducts_%s.png", data_variant_name), width = 7, height = 5, units = "in", res = 300)
  print(shap.plot.dependence(shap_long, x = "NumOfProducts"))
  dev.off()

  png(sprintf("outputs/plots/shap_dependence_age_%s.png", data_variant_name), width = 7, height = 5, units = "in", res = 300)
  print(shap.plot.dependence(shap_long, x = "Age"))
  dev.off()

  png(sprintf("outputs/plots/roc_xgboost_%s.png", data_variant_name), width = 6, height = 6, units = "in", res = 300)
  plot(roc_xgb, main = paste("ROC Curve - XGBoost -", data_variant_name))
  dev.off()

  ## ---- save all outputs, namespaced by data variant ----
  suffix <- data_variant_name
  write.csv(results, sprintf("outputs/tables/model_results_%s.csv", suffix), row.names = FALSE)
  write.csv(xgb_cv_search, sprintf("outputs/tables/xgb_hyperparameter_search_%s.csv", suffix), row.names = FALSE)
  write.csv(data.frame(threshold = thresholds, Fbeta = fbeta_at_threshold),
            sprintf("outputs/tables/threshold_sweep_OOF_%s.csv", suffix), row.names = FALSE)
  write.csv(data.frame(best_threshold = best_threshold, beta = BETA),
            sprintf("outputs/tables/chosen_threshold_%s.csv", suffix), row.names = FALSE)
  write.csv(shap_values$mean_shap_score, sprintf("outputs/tables/shap_mean_importance_%s.csv", suffix))
  write.csv(shap_long, sprintf("outputs/tables/shap_long_values_%s.csv", suffix), row.names = FALSE)
  write.csv(fairness_gender, sprintf("outputs/tables/fairness_gender_%s.csv", suffix), row.names = FALSE)
  write.csv(fairness_age, sprintf("outputs/tables/fairness_age_%s.csv", suffix), row.names = FALSE)
  saveRDS(model_xgb, sprintf("outputs/models/xgb_model_%s.rds", suffix))

  list(results = results, best_threshold = best_threshold, shap_long = shap_long,
       fairness_gender = fairness_gender, fairness_age = fairness_age,
       xgb_cv_search = xgb_cv_search)
}

# ==================================================================
# Run both data variants: primary (Complain excluded) and sensitivity
# (Complain included) - Section 3.5.
# ==================================================================
primary_run     <- run_pipeline("primary_no_complain", drop_complain = TRUE)
sensitivity_run <- run_pipeline("sensitivity_with_complain", drop_complain = FALSE)

cat("\n\n==================== COMPARISON: primary vs Complain-included ====================\n")
print(bind_rows(primary_run$results, sensitivity_run$results))
