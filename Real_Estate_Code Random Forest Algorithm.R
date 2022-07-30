# Price of a property prediction ---------------------------------------------------------------------

# Set working directory
setwd("C:/Users/staru/OneDrive/Desktop/Machine Learning/Edvancer Projects R/Real_Estate_Project/LR model 4 Final")

# Importing training and test dataset
hd_train <- read.csv("housing_train.csv", stringsAsFactors = FALSE)
hd_test <- read.csv("housing_test.csv", stringsAsFactors = FALSE)

# Library to use for this model
library(dplyr) # For Data Preparation
library(cvTools) # For cross validation
library(randomForest) # For building random forest

# Structure of the both Training and Test data set
glimpse(hd_train)
glimpse(hd_test)

# Column Names of Training and Test Data
names(hd_train)
names(hd_test)

# Difference which column name is not available on test data
setdiff(names(hd_train), names(hd_test))

# Check Missing Value in Imported Training and Test Data
sum(is.na(hd_train))
sum(is.na(hd_test))

# Response variable is 'Price'
# Creating a empty column for response variable in hd_test
hd_test$Price = NA

# Creating new column named 'data' in both hd_train & hd_test
# Which specifies if the data is from train or test set
hd_train$data = 'train'
hd_test$data = 'test'

# Combine both data sets into one for data cleaning purpose 
hd_all <- rbind(hd_train, hd_test)

glimpse(hd_all)

# Checking missing value of hd_all Dataset
sum(is.na(hd_all))
sapply(hd_all, function(x) sum(is.na(x)))

# Data Preparation -------------------------------------------------------------------------------

# Let's check number of unique values in each column
sort(sapply(hd_all, function(x) length(unique(x))))

# Name of all character columns
names(hd_all)[sapply(hd_all, function(x) is.character(x))]
# Name of all numerical columns
names(hd_all)[sapply(hd_all, function(x) is.numeric(x))]

# Create Dummy Function Formula
CreateDummies = function(data, var, freq_cutoff = 0){
  t = table(data[,var])
  t = t[t > freq_cutoff]
  t = sort(t)
  categories = names(t)[-1] # Excluding first name as we need one less dummy variable
  
  for(cat in categories){
    name=paste(var,cat,sep="_") # Column name is written as var_cat, e.g., State_FL
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    
    data[,name] = as.numeric(data[,var] == cat)
    # column gets values 1 & 0 for respective category
  }
  
  data[,var]= NULL # Removing 'var' column from data
  return(data) #return data
}

# Drop Variable - Address
hd_all <- hd_all %>% 
  select(-Address)

# Create Data Frame for character variables
char_logical = sapply(hd_all,is.character)
cat_cols = names(hd_all)[char_logical]
cat_cols

cat_cols = cat_cols[!(cat_cols %in% c('data','Price'))]
cat_cols

# Create Dummies for character variables
for(col in cat_cols){
  hd_all = CreateDummies(hd_all, col, 50)
}

glimpse(hd_all)

# Let's check number of unique values in each column
sort(sapply(hd_all, function(x) length(unique(x))))

# Name of all character columns
names(hd_all)[sapply(hd_all, function(x) is.character(x))]
                     
# Name of all numerical columns
names(hd_all)[sapply(hd_all, function(x) is.numeric(x))]

# Missing Value Treatment ---------------------------------------------------------------------------------------

# NA values in all the columns of hd_all
sum(is.na(hd_all))
sort(sapply(hd_all, function(x) sum(is.na(x))))

# We can go ahead and separate training and test data BUT first we check NA values
hd_all = hd_all[!((is.na(hd_all$Price)) & hd_all$data == 'train'), ]

# Imputing all missing values by mean function
for(col in names(hd_all)){
  
  if(sum(is.na(hd_all[,col])) > 0 & !(col %in% c("data","Price"))) {
    
    hd_all[is.na(hd_all[,col]),col] = mean(hd_all[hd_all$data =='train',col],na.rm = T)
  }
  
}

# Price have 1885 missing values that we have own create
sum(is.na(hd_all)) # 1885 - These are missing value of Price from Test Data

# Separate train and test
hd_train = hd_all %>% filter(data == 'train') %>% select(-data)
hd_test = hd_all %>% filter(data =='test') %>% select (-data,-Price)

# # Export Training and Test data set for future use
write.csv(hd_train, "hd_train_clean.csv", row.names = F)
write.csv(hd_test, "hd_test_clean.csv", row.names = F)

# -------------------------------------------------------------------------------------------------------------------------------------
# Model Building On Entire Training data ----------------------------------------------------------------
# Regression Random Forest with Parameter Tuning -------------------------------------------------------------------------------------

param = list(mtry = c(5,10,15,20,25,50),
             ntree = c(50,100,200,500,700,1000),
             maxnodes = c(5,10,15,20,30,50),
             nodesize = c(1,2,5,10))

# Function for getting all possible combinations : expand.grid()
all_comb = expand.grid(param) # Grid Search for all combination
#5*5*6*4 = 600 combinations of parameters,
# And for 10-fold CV, it would build 600*10 trees to find the best 
# performing parameters.

# Function for selecting random subset of Params
subset_paras = function(full_list_para, n = 10){
  
  all_comb = expand.grid(full_list_para)
  
  set.seed(1)
  
  s = sample(1:nrow(all_comb),n)
  
  subset_para = all_comb[s,]
  
  return(subset_para)
}

# Randomize Grid Search
num_trials = 55
my_params = subset_paras(param, num_trials)
# Note: A good value for num_trials is around 50-60

# CVTuning For Regression
myerror = 9999999

# Lets Start CVTuning For Regression
for(i in 1:num_trials){
  print(paste0('starting iteration:',i))
  # Uncomment the line above to keep track of progress
  params = my_params[i,]
  
  k = cvTuning(randomForest, Price~., data = hd_train,
               tuning = params,
               folds = cvFolds(nrow(hd_train), K = 10, type = "random"),
               seed = 2
  )
  score.this = k$cv[,2] # Cross Validation error
  print(paste0('CV Score: ', score.this))
  
  if(score.this < myerror){
    print(params)
    # Uncomment the line above to keep track of progress
    myerror = score.this
    print(myerror)
    # Uncomment the line above to keep track of progress
    best_params = params
  }
  
  print('DONE')
  # Uncomment the line above to keep track of progress
}

myerror
best_params
            
# mtry = Number of variables randomly sampled as candidates at each split. 
# ntree = Number of trees to grow
# maxnodes = Maximum number of terminal nodes trees in the forest can have
# nodesize = Minimum size of terminal nodes

# Final Model With Obtained Best Parameters ------------------------------------------------------------------------

hd.rf.final = randomForest(Price~.,
                           mtry = best_params$mtry,
                           ntree = best_params$ntree,
                           maxnodes = best_params$maxnodes,
                           nodesize = best_params$nodesize,
                           data = hd_train)

hd.rf.final
names(hd.rf.final)
hd.rf.final$terms

# Making Prediction on Entire Test Data Set ------------------------------------------------------------------

# Prediction on Test Data Set
test.pred = predict(hd.rf.final, newdata = hd_test)
test.pred[1:5] # Probability Score For Entire Training Data

# Prediction on Training Data Set
train.final.pred <- predict(hd.rf.final, newdata = hd_train)
train.final.pred[1:5] # Probability Score of Entire Training Data
hd_train$Price[1:5] # Price in Training Data

train.final.error <- hd_train$Price - train.final.pred
train.final.error[1:5] # All Final Training Errors

# Root Mean Square Error (RMSE Value)
train.final.error**2 %>% mean() %>% sqrt()
# Final Training Error 346297.7

# Export Prediction for submission
write.csv(test.pred,"mysubmissionRFR.csv",row.names = F)

