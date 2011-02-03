
data.file.name <- 'vectors.csv'
mydata <- data.matrix(read.table(data.file.name, sep = ",", header = TRUE, row.names = 1))
mydata <- scale(mydata)
# Determine number of clusters
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:15) {
    wss[i] <- sum(kmeans(mydata, centers=i)$withinss)
}
x11()
plot(1:15, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

k <- 5
fit <- kmeans(mydata, 5) # 5 cluster solution
x11()
library(cluster)
clusplot(mydata, fit$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
