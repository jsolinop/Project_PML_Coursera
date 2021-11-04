# modeling LDA
set.seed(1234)

system.time(modelFit_LDA <-
                    train(
                            classe ~ ., 
                            data = training, 
                            method = 'lda'
                    ))

result_LDA <- predict(modelFit_LDA,newdata = testing)
confM_LDA <- confusionMatrix(result_LDA,testing$classe)
confM_LDA$overall


system.time(modelFit_LDA_prep <-
                    train(
                            classe ~ .,
                            data = training,
                            method = 'lda'
                            ))
result_LDA_prep <- predict(modelFit_LDA_prep,newdata = testing)
confM_LDA_prep <- confusionMatrix(result_LDA_prep,testing$classe)
confM_LDA_prep$overall


# modeling QDA
modelFit_QDA <- train(classe ~ ., data = training, method = 'qda' )
result_QDA <- predict(modelFit_QDA,newdata = testing)
confM_QDA <- confusionMatrix(result_QDA,testing$classe)
confM_QDA$overall
