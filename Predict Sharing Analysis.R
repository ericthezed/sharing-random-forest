library(randomForest)
library(caret)
library(ROCR)
library(C50)
library(gmodels)

set.seed(300)

# Load data

fbs <- read.csv("predict_shares_data.csv")

# Randomize data
fbs <- fbs[sample(nrow(fbs)),]

# Drop ID column
fbs <- fbs[-c(1:2)]

# Factorize outcome variable

fbs$shared <- factor(fbs$shared, levels = c("no_shared", "shared"), labels = c("no_shared", "shared"))

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

plot(perf, main = "ROC curve for FB Subscription Prediction", col= "blue", lwd = 3)
abline(a=0, b=1, lwd=2, lty=2)


# Try Decision Tree instead

set.seed(12345)

fbs_dt_model <- C5.0(fbs_train, fbs_train_result)
summary(fbs_dt_model)

fbs_dt_pred <- predict(fbs_dt_model, fbs_test)

CrossTable(fbs_test_result, fbs_dt_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn= c('actual subscribe', 'predicted subscribe'))

# Create data.frame of only returning visitors
fbs_r <- subset(fbs, fbs$prev_sessions > 0)

# Create training set with first 80% of rows
fbsr_train <- fbs_r[1:round(nrow(fbs_r) * 0.8, 0), ]

# Create test set with remaining 20% of rows
fbsr_test <- fbs_r[(round(nrow(fbs_r) * 0.8, 0)+1):nrow(fbs_r),]

# Create vectors of outcome variables
fbsr_train_result <- fbsr_train[1]
fbsr_test_result <- fbsr_test[1]

# Remove outcome variables from datasets
fbsr_train <- fbsr_train[-1]
fbsr_test <- fbsr_test[-1]

# Turn outcome variables into vectors
fbsr_train_result <- unlist(fbsr_train_result)
fbsr_test_result <- unlist(fbsr_test_result)

# Train model
rf_r <- randomForest(fbsr_train_result ~ ., data=fbsr_train)

# Check error rate in model
rf_r

# Predict results
fbsr_test_predict <- predict(rf_r, fbsr_test, type="prob")

# Create prediction object
pred_r <- prediction(predictions = fbsr_test_predict[,2],
                     labels = fbsr_test_result)

perf <- performance(pred_r, measure="tpr", x.measure="fpr")

plot(perf, main = "ROC curve for FB Subscription Prediction", col= "blue", lwd = 3)
abline(a=0, b=1, lwd=2, lty=2)
abline(a=.8, b=0)
abline(a=.9, b=0)
abline(a=.95, b=0)


# Logit model

# Load data
sharing_data <- read.csv("sharing_data.csv")
sharing <- data.frame(sharing_data)


# Assign names to variables
names(sharing) <- c("nugget_id", "visitor_id", "session_id", "device", "shares", "pageviews", "likes", "video_views_thirty", "comments", "subscribes", "new_visitor", "avg_ams")

# Remove unneeded columns
# sharing <- subset(sharing, select = -c(visitor_id, session_id))

# Recode variables

sharing$shares <- ifelse(sharing$shares > 0,1,0)
sharing$shares <- factor(sharing$shares, levels = c(0,1), labels = c("No", "Yes"))

sharing$new_visitor <- revalue(sharing$new_visitor, c("Yes"="New", "No"="Returning"))
sharing$new_visitor <- factor(sharing$new_visitor, levels = c("New","Returning"), labels = c("New","Returning"))

sharing$device <- factor(sharing$device, levels = c("Mobile","Desktop", "Tablet"), labels = c("Mobile","Desktop", "Tablet"))

sharing$log_avg_ams <- log10(sharing$avg_ams)


# Histogram of AMs

hist_avg_ams <- hist(avg_ams)
hist_log_avg_ams <- hist(log_avg_ams)

hist_avg_ams
hist_log_avg_ams


# Logit models predicting shares

logit_share <- glm(shares ~ device + pageviews + likes + video_views_thirty + comments + subscribes + new_visitor + log_avg_ams, family = "binomial", data = sharing)
summary(logit_share)


# Logit models predicting shares by device type 
logit_share <- glm(shares ~ pageviews + likes + video_views_thirty + comments + subscribes + new_visitor + log_avg_ams, family = "binomial", data = subset(sharing, device == "Desktop"))
summary(logit_share_desktop)

logit_share <- glm(shares ~ pageviews + likes + video_views_thirty + comments + subscribes + new_visitor + log_avg_ams, family = "binomial", data = subset(sharing, device == "Mobile"))
summary(logit_share_mobile)

logit_share <- glm(shares ~ pageviews + likes + video_views_thirty + comments + subscribes + new_visitor + log_avg_ams, family = "binomial", data = subset(sharing, device == "Tablet"))
summary(logit_share_tablet)








# Crosstabs for Shares
library("gmodels")

CrossTable(sharing$shares, sharing$device, chisq = TRUE)
CrossTable(sharing$shares, sharing$new_visitor, chisq = TRUE)




# Plot logit models
library("ggplot2")

sharing$shares_num <- as.numeric(sharing$shares=="Yes")

ggplot(sharing, aes(x= , y=shares_num)) + geom_point() + stat_smooth(method="glm", family="binomial", se=F, aes(y=shares_num)) + facet_grid(new_visitor ~ device_type) + xlab(" ") + ylab("Share Probability") + ggtitle(" vs. Share Probability")

ggplot(sharing, aes(x= , y=shares_num)) + geom_point() + stat_smooth(method="glm", family="binomial", se=F, aes(y=shares_num)) + facet_grid(new_visitor ~ device_type) + xlab(" ") + ylab("Share Probability") + ggtitle(" vs. Share Probability")

ggplot(sharing, aes(x= , y=shares_num)) + geom_point() + stat_smooth(method="glm", family="binomial", se=F, aes(y=shares_num)) + facet_grid(new_visitor ~ device_type) + xlab(" ") + ylab("Share Probability") + ggtitle(" vs. Share Probability")

# ggplot(page_weight_data, aes(x=log_max_load_time, y=bounced)) + geom_point() + stat_smooth(geom = "smooth", method="glm", family="binomial", formula = page_weight_data$bounced ~ page_weight_data$log_max_load_time, se=FALSE, na.rm = TRUE) + facet_grid(new_visitor ~ device_type)
# ggplot(page_weight_data, aes(x=log_max_load_time, y=quick_exit_10)) + geom_point() + stat_smooth(geom = "smooth", method="glm", family="binomial", formula = page_weight_data$bounced ~ page_weight_data$log_max_load_time, se=FALSE, na.rm = TRUE) + facet_grid(new_visitor ~ device_type)
# ggplot(page_weight_data, aes(x=log_max_load_time, y=quick_exit_30)) + geom_point() + stat_smooth(geom = "smooth", method="glm", family="binomial", formula = page_weight_data$bounced ~ page_weight_data$log_max_load_time, se=FALSE, na.rm = TRUE) + facet_grid(new_visitor ~ device_type)
