
# load in monomial'd csv data ...
data.file.name = 'tensile-strength-177c.extendedcsv'
dm <- data.matrix(read.table(data.file.name, sep = ",", header = TRUE, row.names = 1))

test.fraction <- 1.0 / 3.0
test.size <- round(nrow(dm) * test.fraction)
test.indices <- sample(nrow(dm), test.size)

dm.test <- dm[test.indices, ]
x.test <- dm.test[, 1:(ncol(dm) - 1)]
y.test <- dm.test[, ncol(dm)]

dm <- dm[-test.indices, ]

x <- dm[, 1:(ncol(dm) - 1)]
y <- dm[, ncol(dm)]

library(lars)

do.cv <- TRUE

if(do.cv) {
    cv.lars(
        x,
        y,
        type = 'forward.stagewise',
        use.Gram = FALSE,
        trace = TRUE,
        plot.it = TRUE,
        fraction = seq(from = 0, to = 0.01, length = 100)
    )
}
model <- lars(
    x,
    y,
    type = 'lasso',
    trace = TRUE,
)

s.value <- 0.65
y.predicted <- predict(model, x.test, s = s.value, type = 'fit', mode = 'fraction')

# out of curiosity what are the coefficients?
c <- predict(model, s = s.value, type = 'coefficients', mode = 'fraction')
c.sparse <- c$coefficients[c$coefficients != 0.0]
print(c.sparse)

r <- y.test - y.predicted$fit
mse <- (t(r) %*% r) / length(r)

r.squared <- 1.0 - (mse / var(y.test))

print(paste('r squared:', r.squared))
