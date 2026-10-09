# Faithful classroom formulas. No classroom data is embedded.
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]; input <- args[2]; output <- args[3]
d <- read.csv(input, check.names = FALSE)
if (mode == 'train') {
  d$subscribe <- ifelse(d$subscribe == 'yes', 1, 0)
  d$recency <- d$last
  d$frequency <- d$home + d$sports + d$clothes + d$health + d$books + d$digital + d$toys
  d$monetary_value <- d$electronics + d$nonelectronics
  set.seed(12345)
  training_index <- sample(x = 1:nrow(d), size = .75 * nrow(d), replace = FALSE)
  training <- d[training_index, ]; test <- d[-training_index, ]
  tree1 <- rpart::rpart(subscribe ~ recency + frequency + monetary_value, data = training, method = 'class')
  set.seed(888)
  forest1 <- ranger::ranger(subscribe ~ recency + frequency + monetary_value,
    data = training, num.trees = 5000, seed = 888, probability = TRUE)
  saveRDS(tree1, file.path(output, 'tree.rds'))
  saveRDS(forest1, file.path(output, 'forest.rds'))
  for (name in c('training','test')) {
    frame <- get(name)
    frame$tree_probability <- predict(tree1, newdata = frame, type = 'prob')[, '1']
    frame$forest_probability <- predict(forest1, frame)$predictions[, 2]
    write.csv(frame, file.path(output, paste0(name, '.csv')), row.names = FALSE)
  }
  capture.output(print(tree1), file = file.path(output, 'tree_rules.txt'))
} else if (mode == 'predict') {
  model <- readRDS(args[4])
  if (inherits(model, 'rpart')) p <- predict(model, newdata = d, type = 'prob')[, '1']
  else p <- predict(model, d)$predictions[, 2]
  write.csv(data.frame(probability = p), output, row.names = FALSE)
} else if (mode == 'cluster') {
  k <- as.integer(args[4]); maximum <- as.integer(args[5]); set.seed(as.integer(args[6]))
  X <- as.matrix(d)
  centroids <- X[sample.int(nrow(X), k), , drop = FALSE]
  history <- list()
  for (iteration in seq_len(maximum)) {
    clusters <- apply(X, 1, function(row) which.min(rowSums((centroids - matrix(row, nrow=k, ncol=ncol(X), byrow=TRUE))^2)))
    history[[length(history)+1]] <- data.frame(step=2*iteration-1, row=seq_len(nrow(X)), cluster=clusters)
    updated <- centroids
    for (id in seq_len(k)) {
      points <- X[clusters==id, , drop=FALSE]
      updated[id, ] <- if (nrow(points)==0) X[sample.int(nrow(X),1), ] else colMeans(points)
    }
    history[[length(history)+1]] <- data.frame(step=2*iteration, row=seq_len(nrow(X)), cluster=clusters)
    if (max(abs(updated-centroids)) < 1e-6) break
    centroids <- updated
  }
  write.csv(do.call(rbind,history),output,row.names=FALSE)
}
