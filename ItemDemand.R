library(vroom)
library(forecast)
library(dplyr)
library(ggplot2)

train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")

store1 <- train_data |>
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
