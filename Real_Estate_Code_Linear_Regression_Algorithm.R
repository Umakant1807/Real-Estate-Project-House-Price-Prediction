
# Price of a property prediction ---------------------------------------------------------------------

# Set working directory
setwd("C:/Users/staru/OneDrive/Desktop/Real Estate Project")

# Importing training and test dataset
hd_train <- read.csv("housing_train.csv", stringsAsFactors = FALSE)
hd_test <- read.csv("housing_test.csv", stringsAsFactors = FALSE)

# Library to use for this model
library(dplyr) # For Data Preparation
library(tidyr) # For Data Preparation

# Structure of the both training and test data set
glimpse(hd_train)
glimpse(hd_test)

# Column names of training and test data
names(hd_train)
names(hd_test)

# Difference which column name is not available on test data
setdiff(names(hd_train), names(hd_test))

# Check Missing Value in imported training and test data
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

# To check the combined data set which names as hd_all
glimpse(hd_all)

# Checking missing value of hd_all dataset
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

# Drop Variable - Address which have 9324 unique values
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

sum(is.na(hd_all)) # 1885 - These are missing value of Price from test Data

# Separate train and test
hd_train = hd_all %>% filter(data == 'train') %>% select(-data)
hd_test = hd_all %>% filter(data =='test') %>% select (-data,-Price)

# # Export Training and Test data set for future use
write.csv(hd_train, "hd_train_clean.csv", row.names = F)
write.csv(hd_test, "hd_test_clean.csv", row.names = F)

# -------------------------------------------------------------------------------------------------------------------------------------
# Model building for entire training data ----------------------------------------------------------------------------------------------
# Remove Multicoliniarity of training data -------------------------------------------------------------------------------------------

# Fitting a final linear model on the data using all variables
fit.final <- lm(Price~., data = hd_train)
summary(fit.final)
formula(fit.final)

library(car)

# we'll take vif cutoff as 5
vif(fit.final)
sort(vif(fit.final), decreasing = T)[1:5] # CouncilArea_ have high Multicoliniarity

# Remove CouncilArea_ variable as it has highest vif (49.74370)
fit.final <- lm(Price~. -CouncilArea_, data = hd_train)
sort(vif(fit.final),decreasing = T)[1:5]

# Remove Postcode variable as it has highest vif (10.467797)
fit.final <- lm(Price~. -CouncilArea_ -Postcode, data = hd_train)
sort(vif(fit.final),decreasing = T)[1:5]

# Remove Distance variable as it has highest vif (8.115671)
fit.final <- lm(Price~. -CouncilArea_ -Postcode -Distance, data = hd_train)
sort(vif(fit.final),decreasing = T)[1:5] # All Multicolinarity will removed 
# and all values are less than 5

# Set Formula for Linear Model
# Now let's create linear regression model
fit.final <- lm(Price~. -CouncilArea_ -Postcode -Distance, data = hd_train)
summary(fit.final)
formula(fit.final)
names(summary(fit.final))

# Step-wise regression for variable selection based on AIC score -------------------------------------
fit.final <- step(fit.final)

# Based on AIC score _ Lower the better
summary(fit.final) # Not all variables are significant
names(fit.final$coefficients)
formula(fit.final)

# Based on summary of fit.final remove variables step by step which have highest p-values
fit.final = lm(Price ~ Rooms + Bedroom2 + Bathroom + Car + Landsize + BuildingArea + 
                 YearBuilt + Suburb_Hadfield + Suburb_Abbotsford + Suburb_HeidelbergWest + 
                 Suburb_OakleighSouth + Suburb_CoburgNorth + Suburb_HeidelbergHeights + 
                 Suburb_Malvern + Suburb_Moorabbin + Suburb_Rosanna + Suburb_Niddrie + 
                 Suburb_Maidstone + Suburb_AirportWest + Suburb_Bulleen + 
                 Suburb_Ormond + Suburb_Strathmore + Suburb_SunshineNorth + 
                 Suburb_WestFootscray + Suburb_AvondaleHeights + Suburb_Fawkner + 
                 Suburb_AltonaNorth + Suburb_Armadale + Suburb_Burwood + Suburb_Williamstown + 
                 Suburb_Melbourne + Suburb_SunshineWest + Suburb_Ivanhoe + 
                 Suburb_TemplestoweLower + Suburb_KeilorEast + Suburb_HawthornEast + 
                 Suburb_Prahran + Suburb_Kensington + Suburb_Sunshine + Suburb_Toorak + 
                 Suburb_Maribyrnong + Suburb_Doncaster + Suburb_AscotVale + 
                 Suburb_Hampton + Suburb_Balwyn + Suburb_MalvernEast + Suburb_Camberwell + 
                 Suburb_PascoeVale + Suburb_BrightonEast + Suburb_Hawthorn + 
                 Suburb_BalwynNorth + Suburb_Coburg + Suburb_Northcote + Suburb_Kew + 
                 Suburb_Brighton + Suburb_Glenroy + Suburb_GlenIris + Suburb_Essendon + 
                 Suburb_SouthYarra + Suburb_Preston + Suburb_BentleighEast + 
                 Suburb_Reservoir + Type_u + Type_h + Method_PI + Method_S + 
                 SellerG_Raine + SellerG_Douglas + SellerG_Kay + SellerG_Miles + 
                 SellerG_Greg + SellerG_Sweeney + SellerG_RT + SellerG_Ray + 
                 SellerG_Marshall + SellerG_Barry + SellerG_hockingstuart + 
                 SellerG_Jellis + CouncilArea_Monash + CouncilArea_Whitehorse + 
                 CouncilArea_Brimbank + CouncilArea_HobsonsBay + CouncilArea_Bayside + 
                 CouncilArea_Melbourne + CouncilArea_Banyule + CouncilArea_PortPhillip + 
                 CouncilArea_Yarra + CouncilArea_Maribyrnong + CouncilArea_GlenEira + 
                 CouncilArea_MooneeValley + CouncilArea_Moreland + CouncilArea_Boroondara, data = hd_train)

summary(fit.final)
sort((summary(fit.final)$coefficients)[,4], decreasing = T)[1:5]

# Final Fit Formula
fit.final = lm(Price ~ Rooms + Bedroom2 + Bathroom + Car + Landsize + BuildingArea + 
                 YearBuilt + Suburb_Hadfield + Suburb_HeidelbergWest + Suburb_OakleighSouth + 
                 Suburb_CoburgNorth + Suburb_HeidelbergHeights + Suburb_Malvern + 
                 Suburb_Moorabbin + Suburb_Rosanna + Suburb_Niddrie + Suburb_Maidstone + 
                 Suburb_AirportWest + Suburb_Bulleen + 
                 Suburb_SunshineNorth + Suburb_WestFootscray + Suburb_AvondaleHeights + 
                 Suburb_Fawkner + Suburb_AltonaNorth + Suburb_Armadale + 
                 Suburb_Williamstown + Suburb_SunshineWest + Suburb_Ivanhoe + 
                 Suburb_TemplestoweLower + Suburb_KeilorEast + Suburb_HawthornEast + 
                 Suburb_Prahran + Suburb_Kensington + Suburb_Sunshine + Suburb_Toorak + 
                 Suburb_Maribyrnong + Suburb_Hampton + 
                 Suburb_Balwyn + Suburb_MalvernEast + Suburb_Camberwell + 
                 Suburb_PascoeVale + Suburb_BrightonEast + Suburb_Hawthorn + 
                 Suburb_BalwynNorth + Suburb_Coburg + Suburb_Northcote + Suburb_Kew + 
                 Suburb_Brighton + Suburb_Glenroy + Suburb_GlenIris + Suburb_Essendon + 
                 Suburb_SouthYarra + Suburb_Preston + Suburb_BentleighEast + 
                 Suburb_Reservoir + Type_u + Type_h + Method_PI + Method_S + SellerG_Kay + SellerG_Miles + SellerG_Greg + SellerG_RT + SellerG_Marshall + SellerG_Jellis + CouncilArea_Whitehorse + 
                 CouncilArea_Brimbank + CouncilArea_HobsonsBay + CouncilArea_Bayside + 
                 CouncilArea_Melbourne + CouncilArea_Banyule + CouncilArea_PortPhillip + 
                 CouncilArea_Yarra + CouncilArea_Maribyrnong + CouncilArea_GlenEira + 
                 CouncilArea_MooneeValley + CouncilArea_Moreland + CouncilArea_Boroondara, data = hd_train)

summary(fit.final)
sort((summary(fit.final)$coefficients)[,4], decreasing = T)[1:5]

# Making prediction on the Entire training data -------------------------------------------

# Prediction on Test Data Set
test.pred <- predict(fit.final, newdata = hd_test)
test.pred[1:5] # Probability Score For Entire Test Data

# Prediction on Training Data Set
train.final.pred <- predict(fit.final, newdata = hd_train)
train.final.pred[1:5] # Probability Score for Entire Training Data
hd_train$Price[1:5] # Interest Rate in Training Data

train.final.error <- hd_train$Price - train.final.pred
train.final.error[1:5] # All Final Training Errors

# Root Mean Square Error (RMSE Value)
train.final.error**2 %>% mean() %>% sqrt()
# Final Training Error  380141.7

# Export Prediction for submission
write.csv(test.pred, "submisionLR.csv", row.names = F)

