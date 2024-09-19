# pm25_models.R


### LOAD LIBRARIES

library(tidymodels)
library(tidyverse)
library(ISLR)
library(ISLR2)
library(tidyverse)
library(glmnet)
library(modeldata)
library(ggthemes)
library(janitor)
library(xgboost)
library(ranger)
library(vip)
library(corrplot)
library(kableExtra)
library(rsample)
library(modeldata)
library(janitor)
library(naniar) # to assess missing data patterns
library(themis) # for upsampling
library(maps)
library(knitr)
library(parsnip)
tidymodels_prefer()


pm25_data <- read.csv("Data/PM25_data.csv") 

set.seed(111) # for reproducibility

pm25_split <- initial_split(pm25_data, prop=0.7, strata = "site_id")
pm25_train <- training(pm25_split)
pm25_test <- testing(pm25_split)

# verify
nrow(pm25_train)/nrow(pm25_data)
nrow(pm25_test)/nrow(pm25_data)



### CREATE RECIPE
pm25_recipe <- recipe(pm25 ~ county_id + latitude + longitude 
                      + tavg + tmin + tmax + prcp + wdir + wspd + 
                        pres + date, data=pm25_data) %>%
  step_mutate(date = as.Date(date, format = "%m/%d/%Y")) %>%
  step_impute_bag(wdir) %>%
  step_date(date, features = "doy") %>%
  step_rm(date) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())
# prep(pm25_recipe) %>% bake(pm25_train)

pm25_folds <- group_vfold_cv(pm25_train, v = 10, group = "site_id")


### CREATE MODELS

# model 1: Linear Regression
linear_model = linear_reg() %>%
  set_engine("lm")

# model 2: K-Nearest Neighbors
knn_model = nearest_neighbor(neighbors = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("kknn")

# model 3: Elastic Net
elastic_model = linear_reg(penalty = tune(),
                           mixture = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("glmnet")

# model 4: Random Forest
rf_model= rand_forest(mtry = tune(),
                      trees = tune(),
                      min_n = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("ranger", importance = "impurity")

# model 5: Gradient Boosted Trees (XGBoost)
xgb_model = boost_tree(mtry = tune(),
                       trees = tune(),
                       learn_rate = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

# model 6: Neural Network
nn_model = mlp(hidden_units = tune(),
               penalty = tune()) %>%
  set_mode("regression") %>%
  set_engine("nnet")


# Linear Regression
lm_workflow <- workflow() %>% 
  add_model(linear_model) %>% 
  add_recipe(pm25_recipe)

# K-Nearest Neighbors
knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(pm25_recipe)

# Elastic Net
elastic_workflow <- workflow() %>% 
  add_model(elastic_model) %>%
  add_recipe(pm25_recipe)

# Random Forest
rf_workflow <- workflow() %>% 
  add_recipe(pm25_recipe) %>% 
  add_model(rf_model)

# Gradient Boosted Trees (XGBoost)
xgb_workflow <- workflow() %>% 
  add_recipe(pm25_recipe) %>% 
  add_model(xgb_model)

# Neural Network
nn_workflow <- workflow() %>%
  add_recipe(pm25_recipe) %>% 
  add_model(nn_model)



### TUNE HYPERPARAMETERS

# K-Nearest Neighbors
knn_grid <- grid_regular(
  neighbors(range = c(1, 20)),
  levels = 5
)

# Elastic Net
elastic_grid <- grid_regular(
  penalty(range = c(-5, 5)), 
  mixture(range = c(0, 1)),
  levels = 10
)

# Random Forest
rf_grid <- grid_regular(
  mtry(range = c(1, 11)),  # number of predictors at each split
  trees(range = c(200, 1000)),  # trees
  min_n(range = c(2, 10)),  # min data points in a node
  levels = 5  #  levels for each parameter
)

# Gradient Boosted Trees (XGBoost)
xgb_grid <- grid_regular(
  mtry(range = c(1, 11)),  # number of predictors at each split
  trees(range = c(200, 1000)),  # trees
  learn_rate(range = c(0.01, 0.3)),  # learning rate
  levels = 5
)

# Neural Network
nn_grid <- grid_regular(
  hidden_units(range = c(32, 64)),  # neurons in the hidden layer
  penalty(range = c(0.001, 0.1)),  # regularization penalty
  levels = 10
)


### SAVE TUNED MODELS

# KNN
write_rds(knn_results, file = "Models/KNN.rds")

# Elastic Net
write_rds(elastic_results, file = "Models/ElasticNet.rds")

# Random Forest
write_rds(rf_results, file = "Models/RF.rds")

# Gradient Boosted Trees (XGBoost)
write_rds(xgb_results, file = "Models/XGB.rds")

# Neural Network
write_rds(nn_results, file = "Models/NeuralNet.rds")



