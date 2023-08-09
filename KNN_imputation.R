
#Import Libraries
library(readr)
library(tidyverse)
library(mice)
library(caret)
library(bnstruct)

# Read the data
train<-read_csv('TrainingData.csv',na=c('na','#VALUE!','missing','N/A'))
test<-read_csv('testX.csv',na=c('na','#VALUE!','missing','N/A'))

# Function to check NAs column wise
na_check<-function(dataset){
  sapply(dataset,function(x) sum(is.na(x)))
}

# Missing Value count
na_check(train)
na_check(test)

# Missing values dataframe
df<-as.data.frame(na_check(train))

df$info<-rownames(df)
colnames(df)<-c("count","var")

df<-df%>%filter(count>0)

test_na<-as.data.frame(na_check(test))

test_na$info<-rownames(test_na)
colnames(test_na)<-c("count","var")

test_na<-test_na%>%filter(count>0)

# Plot for missing values
df%>%mutate(var=fct_reorder(var,count))%>%top_n(10)%>%ggplot()+geom_col(aes(var,count))+coord_flip()+
  labs(x='Count',y='Variable',title='Missing Values Frequency')+theme(plot.title = element_text(hjust = 0.5))

test_na%>%mutate(var=fct_reorder(var,count))%>%top_n(10)%>%ggplot()+geom_col(aes(var,count))+coord_flip()+
  labs(x='Count',y='Variable',title='Missing Values Frequency')+theme(plot.title = element_text(hjust = 0.5))

# KNN imputation

train_matrix<-as.matrix(train[,c(-1,-53,-48)])

test_matrix<-as.matrix(test[,c(-1,-48)])

train_complete<-knn.impute(train_matrix)

test_complete<-knn.impute(test_matrix)


train_complete<-as.data.frame(train_complete)
test_complete<-as.data.frame(test_complete)

test_complete$mvar47<-test$mvar47
train_complete$mvar47<-train$mvar47

train_complete$default_ind<-train$default_ind

# Save as CSV file
write.csv(train_complete,'train_complete.csv')
write.csv(test_complete,'test_complete.csv')


### Variable Importance Plot


train_complete$default_ind<-as.factor(train_complete$default_ind)

set.seed(123)

# Repeated Cross-Validation

repeat_cv <- trainControl(method='repeatedcv', number=5, repeats=3)

set.seed(123)

## Split the data so that 80% of it is for training

train_index <- createDataPartition(y=train_c$default_ind, p=0.8, list=FALSE)

## Subset the data (For training and validation set)
training_set <- train_c[train_index, ]
val_set <- train_c[-train_index, ]

set.seed(123)

## Train a random forest model
forest <- train(
  
  # Formula
  default_ind~., 
  data=training_set, 
  # `rf` method for random forest
  method='rf', 
  # Add repeated cross validation
  trControl=repeat_cv,
  # Accuracy to measure the performance of the model
  metric='Accuracy')

# Plot
var_imp <- varImp(forest, scale=FALSE)$importance
var_imp <- data.frame(variables=row.names(var_imp), importance=var_imp$Overall)
