library(randomForest)
library(caret)
library(ROCR)
library(C50)
library(gmodels)

set.seed(300)

# Load data

fbs <- read.csv("predict_shares_data.csv")

# Retain cases where current pvs > 0
library("dplyr")

fbs <- filter(fbs, cur_pv_count > 0)


# Randomize data
fbs <- fbs[sample(nrow(fbs)),]

# Drop ID column
fbs <- fbs[-c(1:2)]

# Factorize outcome variable

fbs$shared <- factor(fbs$shared, levels = c("no_shared", "shared"), labels = c("no_shared", "shared"))

# Factorize dummy variables

fbs$dummy_mobile <- factor(fbs$dummy_mobile, levels = c(0,1), labels = c("Not Mobile", "Mobile"))
fbs$dummy_desktop <- factor(fbs$dummy_desktop, levels = c(0,1), labels = c("Not Desktop", "Desktop"))


# Unfactorize shareability variable

fbs$shareability <- as.numeric(as.character(fbs$shareability))
  
# Check split
table(fbs$shared)

# Remove incomplete rows
fbs <- fbs[complete.cases(fbs),]

# Create training set with first 80% of rows
fbs_train <- fbs[1:round(nrow(fbs) * 0.8, 0), ]

# Create test set with remaining 20% of rows
fbs_test <- fbs[(round(nrow(fbs) * 0.8, 0)+1):nrow(fbs),]

# Create vectors of outcome variables
fbs_train_result <- fbs_train[1]
fbs_test_result <- fbs_test[1]

# Remove outcome variables from datasets
fbs_train <- fbs_train[-1]
fbs_test <- fbs_test[-1]

# Turn outcome variables into vectors
fbs_train_result <- unlist(fbs_train_result)
fbs_test_result <- unlist(fbs_test_result)

# Train model
rf <- randomForest(fbs_train_result ~ ., data=fbs_train)

# Check error rate in model
rf

# Predict values for test data
fbs_test_predict <- predict(rf, fbs_test, type="prob")


ctrl <- trainControl(method = "repeatedcv",
                     number = 10, repeats = 10)
grid_rf <- expand.grid(.mtry = c(2, 4, 5, 10))
set.seed(300)
m_rf <- train(fbs_train_result ~ ., data = fbs_train, method = "rf",
              metric = "Kappa", trControl = ctrl,
              tuneGrid = grid_rf)

m_rf

# Create prediction object
pred <- prediction(predictions = fbs_test_predict[,2],
                   labels = fbs_test_result)

perf <- performance(pred, measure="tpr", x.measure="fpr")

plot(perf, main = "ROC curve for FB Share Prediction", col= "blue", lwd = 3)
abline(a=0, b=1, lwd=2, lty=2)


# Try Decision Tree instead

set.seed(12345)

fbs_dt_model <- C5.0(fbs_train, fbs_train_result)
summary(fbs_dt_model)

fbs_dt_pred <- predict(fbs_dt_model, fbs_test)

CrossTable(fbs_test_result, fbs_dt_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn= c('actual share', 'predicted share'))


# Create data.frame of only returning visitors
fbs_r <- subset(fbs, fbs$prev_sessions > 0)

# Create training set with first 80% of rows
fbs_train <- fbs[1:round(nrow(fbs) * 0.8, 0), ]

# Create test set with remaining 20% of rows
fbs_test <- fbs[(round(nrow(fbs) * 0.8, 0)+1):nrow(fbs),]

# Create vectors of outcome variables
fbs_train_result <- fbs_train[1]
fbs_test_result <- fbs_test[1]

# Remove outcome variables from datasets
fbs_train <- fbs_train[-1]
fbs_test <- fbs_test[-1]

# Turn outcome variables into vectors
fbs_train_result <- unlist(fbs_train_result)
fbs_test_result <- unlist(fbs_test_result)

# Train model
rf_r <- randomForest(fbs_train_result ~ ., data=fbs_train)

# Check error rate in model
rf_r

# Predict results
fbs_test_predict <- predict(rf_r, fbs_test, type="prob")

# Create prediction object
pred_r <- prediction(predictions = fbs_test_predict[,2],
                     labels = fbs_test_result)

perf <- performance(pred_r, measure="tpr", x.measure="fpr")

plot(perf, main = "ROC curve for FB Share Prediction", col= "blue", lwd = 3)
abline(a=0, b=1, lwd=2, lty=2)
abline(a=.8, b=0)
abline(a=.9, b=0)
abline(a=.95, b=0)
