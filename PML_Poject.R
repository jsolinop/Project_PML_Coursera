library(readr)
library(dplyr)
library(caret)
library(recipes)
library(MASS)

# Create DATA folder
wd <- getwd()
if(!dir.exists(paste0(wd,"/DATA"))) dir.create(paste0(wd,"/DATA"))

# Download training and test sets
if(!file.exists(paste0("DATA/pml-training.csv"))) {
        download.file(
                "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                "DATA/pml-training.csv"
        )
}
if(!file.exists(paste0("DATA/pml-testing.csv.csv"))) {
        download.file(
                "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                "DATA/pml-testing.csv"
        )
}

pml_training <- read.csv("DATA/pml-training.csv")
pml_test <- read.csv("DATA/pml-testing.csv")

str(pml_training)
count_nas <- apply(pml_training,2,function(x) sum(is.na(x)))
count_str0 <- apply(pml_training,2,function(x) sum(x=="", na.rm = T))

col_nas <-
        names(which(apply(pml_training[,-c(1:5)], 2, function(x)
                any(is.na(x) | (x=="")))))

# Delete nearZerovar - variables with near-zero variance
pml_data <- pml_training[,-c(1:5)] %>% dplyr::select(-all_of(col_nas)) %>% 
        mutate_if(is.character, as.factor) %>% dplyr::select(-nearZeroVar(.))

# Select variables with recursive feature elimination - rfe

library(doParallel)
registerDoParallel(cores = 3)

# Size of the predictors to analyze
subsets <- c(4:10)

# Number of resamples - bootstrapping
reps <- 5

set.seed(1234)
seeds <- vector(mode = "list", length = reps + 1)
for(i in 1:reps) {
        seeds[[i]] <- sample.int(1000,length(subsets)+1)
}
seeds[[reps+1]] <- sample.int(1000,1)

ctrl_rfe <- rfeControl(functions = rfFuncs, method = "boot", number = reps,
                       returnResamp = "all", allowParallel = TRUE,verbose = FALSE,
                       seeds = seeds)

set.seed(4321)
data_rfe <- rfe(classe ~., data = pml_data,
                size = subsets,
                metric = "Accuracy",
                rfeControl = ctrl_rfe,
                ntree = 500)

head(data_rfe$results,12)
data_rfe$optVariables
plot(data_rfe$perfNames)

data_rfe$fit$confusion

ggplot(data = data_rfe$results, aes(x = Variables, y = Accuracy)) +
        geom_line() +
        scale_x_continuous(breaks = unique(data_rfe$results$Variables)) +
        geom_point() +
        geom_errorbar(aes(ymin = Accuracy - AccuracySD, ymax = Accuracy + AccuracySD),width = 0.2) +
        geom_point(data = data_rfe$results %>% slice(which.max(Accuracy)), color = "red")+
        theme_bw()

qplot(pml_data$classe, main = "Classe distribucion")

Train <- createDataPartition(pml_data$classe, p=.7,list = FALSE)

training <- pml_data[Train,]
testing <- pml_data[-Train,]


# modeling K-NN
set.seed(1234)
tr_ctrl_loocv <- trainControl(method = "LOOCV") # Leave one out

tr_ctrl_cv30 <- trainControl(method = "cv", # Cross Validation
             number = 30)
tr_ctrl_cv20 <- trainControl(method = "cv", # Cross Validation
                             number = 20)

tr_ctrl_cv10 <- trainControl(method = "cv", # Cross Validation
                             number = 10)

tr_ctrl_boot <- trainControl(method = "boot", # Bootstrapping
                             number = 25)

modelFit_KNN <- train(classe ~., data = training, method = "knn", trControl = tr_ctrl_loocv)
result_KNN <- predict(modelFit_KNN, newdata = testing)
confm_KNN <- confusionMatrix(result_KNN, testing$classe)
confm_KNN$overall[1]

modelFit_KNN2 <- train(classe ~., data = training, method = "knn", trControl = tr_ctrl_boot)
result_KNN2 <- predict(modelFit_KNN2, newdata = testing)
confusionMatrix(result_KNN2, testing$classe)$overall

plot(modelFit_KNN)
plot(modelFit_KNN2)


# modeling Decision Tree

# modeling Random Forest

# modeling Gboost

# modeling SVM