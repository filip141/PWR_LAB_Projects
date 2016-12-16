# Setup Project folder
setwd('/home/filip/PWR_LAB_Projects/RKMeans')

# Libraries
library(matrixStats)
library(cluster)
library(caret)

zeros <- function(width, height){
	mat <- c()
	for(i in 1:width){
		mat <- rbind(mat, rep(0, height))
	}
	return(mat)
}

# Prepare data
prepareData <- function(path, crossValidation=TRUE, nfolds=10, classify=TRUE){
	# Read CSV
	data = read.csv(path)
	# Labels Mapping
	if(classify){
		classLabels = data$label
		if(typeof(classLabels) == "integer"){
			classLabels = paste("class:", as.character(data$label), sep=" ")
		}
		strlevels = unique(classLabels)
		str2id = 1:length(strlevels)
		names(str2id) <- strlevels
		new_labels = c()
		for(lid in 1:length(classLabels)){
			new_labels = c(new_labels, str2id[[classLabels[lid]]])
		}
		data$label <- new_labels
	}
	else{
		str2id = NULL
	}

	# Set Random Generator Seed
	if(crossValidation){
		set.seed(10)
		csFolds = createFolds(data[, 1], k=nfolds, list=TRUE, returnTrain=FALSE)
		
		# Return tuple
		retVal = list(data, csFolds, str2id)
		names(retVal) <- c("data", "folds", "str2id")
		return(retVal)
	}
	retVal = list(data, NULL, str2id)
	names(retVal) <- c("data", 'folds', "str2id")
	return(retVal)
}

# K-Means Clustering
clusterData <- function(data, classify=TRUE, kmeans=TRUE, clusterNumber=7){
	# Variables
	purnityAll = c()
	dbiAll = c()
	silhouetteAll = c()
	nClasses = length(data$str2id)
	csFolds = data$folds
	data = data$data
	nfolds = length(csFolds)
	# Create confusion matrix
	confusionMatrix <- zeros(nClasses, nClasses)
	# Iterate over Crossvalidation folds
	for(cv in 1:nfolds){
		print("---------------------------------------------------------------")
		# Create Training and test set
		test_set = data[as.numeric(unlist(csFolds[names(csFolds)[cv]])),]
		print(sprintf("Test set created. Length: %d",length(test_set[, 1])))
		training_set = data[as.numeric(unlist(csFolds[names(csFolds)[-cv]])),]
		print(sprintf("Training set created. Length: %d",length(training_set[, 1])))

		# Remove known labels for classification
		if(classify){
			labels = training_set[length(training_set)][, 1]
			training_set = training_set[-length(training_set)]
		}
		# Scale data
		trainingRMean = colMeans(training_set)
		trainingRStd = colSds(as.matrix(training_set))
		trainingNMean = sweep(training_set, 2, trainingRMean, "-")
		training_set = sweep(trainingNMean, 2, trainingRStd, "/")

		# Cluster
		if(kmeans){

			rCluster <- kmeans(training_set, clusterNumber, nstart=20)
			print("K-Means Clustering Completed!")
		}
		else{
			rCluster <- pam(training_set, clusterNumber)
			print("PAM Clustering Completed!")
		}

		# Verify
		print("=========================================================")
		print(sprintf("Veryfing Test data for Fold: %d", cv))

		# Only for classification
		classNames <- vector(mode="character", length=clusterNumber)
		if(classify){
			# Names mapping
			namesMapping = table(rCluster$cluster, labels)
			# Iterate over clusters
			for(clust_id in 1:clusterNumber){
				max_idx = which.max(namesMapping[clust_id,])
				classNm = colnames(namesMapping)[max_idx]
				classNames[clust_id] = classNm
			}
			result = list(rCluster, trainingRStd, trainingRMean, classNames, confusionMatrix)
			names(result) <- c("clustResult", "trainingStd", "trainingMean", "mapping", "confusionMatrix")	

			# Confusion Matrix and Purity
			confusionMatrix = verifyClassifier(result, test_set, kmeans=kmeans)
			purityCluster = purity(rCluster$cluster, labels)
			purnityAll = c(purnityAll, purityCluster)
		}

		if(kmeans){
			db = daviesBouldin(training_set, rCluster$cluster, centers = rCluster$center, 
				clust_size = rCluster$size)
		}
		else{
			db = daviesBouldin(training_set, rCluster$cluster, centers = rCluster$medoids, 
				clust_size = rCluster$clusinfo[,1])
		}
		
		sh = silhouette(rCluster$cluster, daisy(training_set))
		dbiAll = c(dbiAll, db)
		silhouetteAll = c(silhouetteAll, sh[,3])
		print("=========================================================")
	}

	# Metrics
	if(classify){
		acc = accuracy(confusionMatrix)
		prec = precision(confusionMatrix)
		rec = recall(confusionMatrix)
		fs = fscore(confusionMatrix)
		purnityEval = sum(purnityAll) / length(purnityAll)
		dbiEval = sum(dbiAll) / length(dbiAll)
		silhouetteEval = sum(silhouetteAll) / length(silhouetteAll)
		print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print("Algoritm Results")
		print("Confusion Matrix: ")
		print(confusionMatrix)
		print("Evalutaing Algoritm: ")
		print(sprintf("Recall: %f", rec))
		print(sprintf("Accuracy: %f", acc))
		print(sprintf("Precision: %f", prec))
		print(sprintf("F1-Score: %f", fs))
		print(sprintf("Purnity: %f", purnityEval))
		print(sprintf("Davies–Bouldin: %f", dbiEval))
		print(sprintf("Silhouette: %f", silhouetteEval))
	}
	else{
		dbiEval = sum(dbiAll) / length(dbiAll)
		silhouetteEval = sum(silhouetteAll) / length(silhouetteAll)
		print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print(sprintf("Davies–Bouldin: %f", dbiEval))
		print(sprintf("Silhouette: %f", silhouetteEval))	
	}

}

verifyClassifier <- function(rCLuster, test_set, kmeans=TRUE){
	testLabels = test_set$label
	# Centroids and Medoids uncertainity
	if(kmeans){
		centers = rCLuster$clustResult$centers
	}
	else{
		centers = rCLuster$clustResult$medoids
	}
	numCenters = length(centers[, 1])
	confusionMatrix = rCLuster$confusionMatrix
	for(rid in 1:nrow(test_set)){
		obs = test_set[rid, ][-length(test_set[rid, ])]
		label = test_set[rid, ]$label

		# Normalize record
		obsNMean = obs - rCLuster$trainingMean
		obs = obsNMean / rCLuster$trainingStd
		min_dist = 100000
  		min_index = -1
  		for(cent in 1:numCenters){
  			distance = dist(rbind(centers[cent,], obs))	
  			if(distance < min_dist){
  				min_dist = distance 
  				min_index = cent
  			}		
  		}
  		# Verify result
  		predicted = as.numeric(rCLuster$mapping[min_index])
  		confusionMatrix[label, predicted] = confusionMatrix[label, predicted] + 1 
	}
	return(confusionMatrix)
}

daviesBouldin <- function(data, clusters, centers, clust_size) {
	nCLusters = length(clust_size)
	db_vec = rep(0, nCLusters)
	for(nm_id in 1:length(clusters))
	{
		cent_id = clusters[[nm_id]]
		obs = data[nm_id,]
		cent = unname(centers[cent_id, ])
		db_vec[cent_id] = db_vec[cent_id] + (sqrt((sum((obs - cent)^2))) / clust_size[cent_id])
	}
	# Iterate over clusters
	ssd = zeros(nCLusters, nCLusters) 
	for(cl_o in 1:nCLusters){
		for(cl_t in 1:nCLusters){
			cent_o = unname(centers[cl_o, ])
			cent_t = unname(centers[cl_t, ])
			dist = sqrt(sum((cent_o - cent_t)^2))
			ssd[cl_o, cl_t] = (db_vec[cl_o] + db_vec[cl_t]) / dist
		}		
	}
	ssd[is.nan(ssd)] <- 0
	ssd[is.infinite(ssd)] <- 0
	maxRow = apply(ssd, 1, max)
	db = sum(maxRow) / length(maxRow)
	return(db)
}

purity <- function(clusters, classes) {
  purCluster = sum(apply(table(classes, clusters), 2, max)) / length(clusters)
  return(purCluster)
}

fscore <- function(confusionMatrix){
	diagFs = diag(confusionMatrix)

	## Precision Calculate
	cols = colSums(confusionMatrix)
	cols[cols==0] <- 1
	prec = (diagFs / cols)

	## RECALL
	rows = rowSums(confusionMatrix)
	rows[rows==0] <- 1
	rec = (diagFs / rows)
	f1score = 2 * ((prec * rec) / (prec + rec))
	f1score[is.nan(f1score)] <- 0
	f1score = sum(f1score) / length(f1score)
	return(f1score)
}

accuracy <- function(confusionMatrix){
	total = sum(colSums(confusionMatrix))
	diagSum = sum(diag(confusionMatrix))
	return(diagSum / total)
}

precision <- function(confusionMatrix){
	cols = colSums(confusionMatrix)
	cols[cols==0] <- 1
	diagPrec = diag(confusionMatrix)
	precision = sum(diagPrec / cols) / length(diagPrec)
	return(precision)
}

recall <- function(confusionMatrix){
	rows = rowSums(confusionMatrix)
	rows[rows==0] <- 1
	diagRec = diag(confusionMatrix)
	recall = sum(diagRec / rows) / length(diagRec)
	return(recall)
}

# Main function
main <- function(){
	print("Reading data")
	data = prepareData("RData/iris.csv", crossValidation=TRUE, classify=TRUE, nfolds=10)
	clusterData(data, kmeans=FALSE, classify=TRUE, clusterNumber=5)

}
main()
