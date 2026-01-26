### SPQRX
SPQRX is keras base R library for quantile regression and density, introduce in the paper "Semi-parametric bulk and tail regression using spline-based neural networks" By Reetam, Jordan PhD(https://arxiv.org/abs/2504.19994)
```r

library(SPQRX)



dataset <- read.csv('/home/dalton/Downloads/housing (1).csv')

coords <- strsplit(gsub("POINT \\(|\\)", "", dataset$geometry), " ")

# Convert to numeric matrix
coords <- do.call(rbind, lapply(coords, function(x) as.numeric(x)))

# Assign to new columns
dataset$lon <- coords[,1]
dataset$lat <- coords[,2]


y <- matrix ( dataset$price, ncol = 1)
y <- log(y)

dataset <- dataset[, names(dataset) != "price"]

x <- as.matrix(dataset[, c('bedrooms', 'batchrooms', 'squareFeet', 'lon', 'lat')])

data <- preprocessing.data(x, y, n.knots = 25)

x_training <- data$x_training
x_validation <- data$x_validation
x_testing <- data$x_testing

y_training <- data$y_training
y_validation <- data$y_validation
y_testing <- data$y_testing



p_a = 0.95
p_b = 0.999
c1 = 35
c2 = 5


#source('SPQRX.R')
model.heavy <- fit.spqrx(input_dim = 5, hidden_dim = c(30, 30 ), n.knots = 25, x_training = x_training, x_validation = x_validation,
                         y_training = y_training, y_validation = y_validation,p_a = p_a, p_b = p_b, c1 = c1, c2 = c2)



eval.plot.qexp(model.heavy, x_testing, y_testing)

eval.plot.qexp(model.heavy, x_training, y_training)

eval.plot.qexp(model.heavy, x_validation, y_validation)


x_explain <- x_testing[1:2, ]
y_explain <- y_testing[1:2, ]

shapley_values <- eval.explain.shapr(model.heavy, x_testing, x_explain, y_testing, y_explain,
                                     type = 'QF', tau = 0.99)

```{r}
