
data.file.name <- 'distance.csv'
dm <- data.matrix(read.table(data.file.name, sep = ",", header = TRUE, row.names = 1))
distances <- as.dist(dm)

mode <- 'mds'

if(mode == 'hierarchical') {
# Ward Hierarchical Clustering
    fit <- hclust(distances, method="ward")
    x11()
    plot(fit) # display dendogram
} else if(mode == 'mds') {
    fit <- cmdscale(distances, eig = TRUE, k = 2)
    print(fit)
    x11()
    x <- fit$points[, 1]
    y <- fit$points[, 2]
    plot(x, y)
    text(x, y, labels = row.names(dm))
}
