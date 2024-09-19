# pm25_main.R

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



### DATA CLEANING

pm25_data <- read.csv("Data/PM25_data.csv") 
head(pm25_data)

dim(pm25_data)
names(pm25_data)

# rename some columns to make them easier to work with
pm25_data <- pm25_data %>%
  rename(
    site_id = SiteID,
    pm25 = MeanPM25Concentration,
    site_name = SITE,
    latitude = SITE_LATITUDE,
    longitude = SITE_LONGITUDE,
    state = STATE,
    county_id = COUNTY_CODE,
    county_name = COUNTY
  )

pm25_data %>% 
  summary()

missing_columns <- pm25_data %>%
  summarise_all(~ any(is.na(.))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "has_missing") %>%
  filter(has_missing) %>%
  pull(column)

missing_columns


### EDA

vis_miss(pm25_data)

pm25_data %>% ggplot(aes(x = pm25)) +
  geom_histogram(binwidth = 1, color = "blue")


# visualize pm25 from an arbitrary date
pm25_july2020 <- pm25_data %>%
  filter(date == "7/1/20")
california_map <- map_data("state") %>%
  filter(region == "california")
ggplot() +
  geom_polygon(data = california_map, aes(x = long, y = lat, group = group), fill = "lightgray", color = "black") +
  geom_point(data = pm25_july2020, aes(x = longitude, y = latitude, color = pm25, size = pm25), alpha = 0.7) +
  scale_color_gradient(low = "green", high = "red", name = "PM2.5 Levels") +
  labs(x = "Longitude", y = "Latitude") + theme_bw()


numeric_data <- pm25_data %>%
  select_if(is.numeric) %>%
  select(-site_id, -county_id, -wdir, -snow)

pm25_cor <- cor(numeric_data)

pm25_corrplot <- corrplot(pm25_cor, method = "circle", addCoef.col = 1)


ggplot(pm25_data, aes(x = tavg, y = pm25)) +
  geom_point(alpha = 0.6, color = "blue") +
  labs(x = "Average Temperature (Celsius)", 
       y = "PM2.5 Concentration") +
  theme_bw()

ggplot(pm25_data, aes(x = wspd, y = pm25)) +
  geom_point(alpha = 0.6, color = "lightblue") +
  labs(x = "Wind Speed", 
       y = "PM2.5 Concentration") +
  theme_bw()


### SPLIT DATA

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


### READ IN TUNED MODELS

# Linear Regression
lm_fit <- fit_resamples(lm_workflow, resamples = pm25_folds)

# KNN
knn_tuned <- read_rds(file = "Models/KNN.rds")

# Elastic Net
elastic_tuned <- read_rds(file = "Models/ElasticNet.rds")

# Random Forest
rf_tuned <- read_rds(file = "Models/RF.rds")

# Gradient Boosted Trees (XGBoost)
xgb_tuned <- read_rds(file = "Models/XGB.rds")

# Neural Network
nn_tuned <- read_rds(file = "Models/NeuralNet.rds")


### GET BEST MODELS

# Linear Regression
lm_rmse <- lm_fit %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  select(mean)

# best KNN
best_knn <- knn_tuned %>%
  select_best(metric = "rmse")

best_knn_rmse <- knn_tuned %>%
  collect_metrics() %>%
  filter(.config == best_knn$.config, .metric == "rmse") %>%
  select(mean)

# best Elastic Net 
best_elastic <- elastic_tuned %>%
  select_best(metric = "rmse")

best_elastic_rmse <- elastic_tuned %>%
  collect_metrics() %>%
  filter(.config == best_elastic$.config, .metric == "rmse") %>%
  select(mean)

# best Random Forest
best_rf <- rf_tuned %>%
  select_best(metric = "rmse")

best_rf_rmse <- rf_tuned %>%
  collect_metrics() %>%
  filter(.config == best_rf$.config, .metric == "rmse") %>%
  select(mean)

# best XGBoost
best_xgb <- xgb_tuned %>%
  select_best(metric = "rmse")

best_xgb_rmse <- xgb_tuned %>%
  collect_metrics() %>%
  filter(.config == best_xgb$.config, .metric == "rmse") %>%
  select(mean)

# best Neural Network
best_nn <- nn_tuned %>%
  select_best(metric = "rmse")

best_nn_rmse <- nn_tuned %>%
  collect_metrics() %>%
  filter(.config == best_nn$.config, .metric == "rmse") %>%
  select(mean)

training_results <- tibble(
  Model = c("Linear Regression","KNN", "Elastic Net", "Random Forest", 
            "XGBoost","Neural Network"),
  RMSE = c(lm_rmse$mean, best_knn_rmse$mean, best_elastic_rmse$mean, 
           best_rf_rmse$mean, best_xgb_rmse$mean, best_nn_rmse$mean))
training_results


ggplot(training_results, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_bw() +
  labs(y = "RMSE", x = "Model") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

autoplot(knn_tuned, metric="rmse")

autoplot(elastic_tuned, metric="rmse")

autoplot(rf_tuned, metric="rmse")

autoplot(xgb_tuned, metric = 'rmse')

autoplot(nn_tuned, metric="rmse")

best_rf <- rf_tuned %>%
  select_best(metric = "rmse")

best_rf

final_rf_workflow <- rf_workflow %>%
  finalize_workflow(best_rf)

final_rf_model <- final_rf_workflow %>%
  fit(data = pm25_train)

rf_train_pred <- final_rf_model %>%
  predict(new_data = pm25_train)

rf_training_results <- pm25_train %>%
  bind_cols(rf_train_pred %>%
              rename(predicted_pm25 = .pred))

rf_training_results %>% metrics(truth = pm25, estimate = predicted_pm25)


rf_predictions <- final_rf_model %>%
  predict(new_data = pm25_test)

rf_results <- pm25_test %>%
  bind_cols(rf_predictions %>%
              rename(predicted_pm25 = .pred))

rf_results %>% metrics(truth = pm25, estimate = predicted_pm25)

final_rf_model %>% 
  extract_fit_engine() %>%
  vip()


