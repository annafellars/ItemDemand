library(vroom)
library(forecast)
library(dplyr)
library(ggplot2)
library(patchwork)
library(tidymodels)
library(tidyverse)
library(embed)
library(discrim)
library(themis)
library(poissonreg)
library(bestglm)
library(yardstick)

train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")

store1 <- train_data |>
  filter(store == 1, item == 1)

test_store1 <- test_data |>
  filter(store == 1, item == 1)

store2 <- train_data |>
  filter(store == 2, item == 1)

plot1 <- store1 |>
  ggplot(mapping=aes(x = date, y = sales)) +
  geom_line() +
  geom_smooth(se = FALSE)

plot2 <- store1 |>
  pull(sales) |>
  forecast::ggAcf()


plot3 <- store1 |>
  pull(sales) |>
  forecast::ggAcf(lag.max = 2*365)


plot4 <- store2 |>
  ggplot(mapping=aes(x = date, y = sales)) +
  geom_line() +
  geom_smooth(se = FALSE)

plot5 <- store2 |>
  pull(sales) |>
  forecast::ggAcf()


plot6 <- store2 |>
  pull(sales) |>
  forecast::ggAcf(lag.max = 2*365)

((plot1 | plot2 | plot3) / (plot4 | plot5 | plot6))

my_recipe <- recipe(sales~., data = store1) |>
  step_rm(c("store", "item")) |>
  step_date(date, features = c("dow", "month", "year", "doy", "decimal")) |>
  step_mutate(
    date_dow = factor(date_dow),
    date_month = factor(date_month),
    date_doy = factor(date_doy),
    date_year = as.numeric(date_year),
    date_decimal = as.numeric(date_decimal)) |>
  step_range(date_decimal, min = 0, max = pi) |>
  step_mutate(sinDEC = sin(date_decimal), cosDEC=cos(date_decimal)) |>
  step_lag(date_decimal, lag = 3, default = 0) |>
  step_lencode_glm(all_nominal_predictors(), outcome = vars(sales)) |>
  step_normalize(all_nominal_predictors())

prepped_recipe <- prep(my_recipe)
show <- bake(prepped_recipe, new_data = store1)


#########################################################
#BART

bart_mod <- bart(trees=tune()) |>
  set_engine("dbarts") |>
  set_mode("regression")

bart_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(bart_mod)

## Set up grid and tuning values
bart_tuning_params <- grid_regular(trees(),
                                   levels = 5)

##Split data for CV
bart_folds <- vfold_cv(store1, v = 5, repeats = 1)

##Run the CV
bart_CV_results <- bart_wf |>
  tune_grid(resamples = bart_folds,
            grid = bart_tuning_params,
            metrics = metric_set(rmse,rsq, smape))

#Find best tuning parameters
bart_best_tune <- bart_CV_results |>
  show_best(metric = "smape", n = 1)

