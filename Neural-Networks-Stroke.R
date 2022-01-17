# Testing neural networks for prediction of Stroke, 01/10/2021, Pawar
setwd("/Users/Bio-user/Desktop/K Award Proposal")
data <- read.csv("Final.csv", header=TRUE)

# Total subjects 1559, column 2-12 are training features, columns 13-25 are predictor outcomes

# Scale the data frame automatically using the scale function in R
# Transform the data using a max-min normalization technique
#scaleddata<-scale(data[,2:12])

# Choose max-min normalization technique as follows
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

maxmindf <- as.data.frame(lapply(data[,2:13], normalize))

maxmindf$Stroke <- data[,13]
#maxmindf <- subset(maxmindf, select = -c(12))

# Training and Test Data, 3 fold split for all outcomes
trainset <- maxmindf[1:1000, ]
testset <- maxmindf[1001:1559, ]

# Training and Test Data, 5 fold split for only stroke
#trainset <- maxmindf[1:900, ]
#testset <- maxmindf[901:1559, ]

# Training and Test Data, 10 fold split for only stroke
#trainset <- maxmindf[1:1044, ]
#testset <- maxmindf[1045:1559, ]


#Training Neural Network
library(neuralnet)
nn <- neuralnet(Stroke ~ Infarct_lateral + Infarct_Site+Male+age+Seizure_Absent+Ini_glu+SBP+DBP+PR+BT+SaO2, data=trainset, hidden=c(2,1), linear.output=TRUE, learningrate.limit = NULL, learningrate = 0.19, threshold=0.01)
nn$result.matrix
plot(nn)

#Test the resulting output
temp_test <- subset(testset, select = c("Infarct_lateral", "Infarct_Site", "Male","age","Seizure_Absent","Ini_glu","SBP","DBP","PR","BT","SaO2"))
head(temp_test)
nn.results <- compute(nn, temp_test)
results <- data.frame(actual = testset$Stroke, prediction = nn.results$net.result)
#options(digits=8)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
cm <- table(prediction, actual)

# Evaluation Metrics
accuracy <- sum(cm[1], cm[4]) / sum(cm[1:4])
precision <- cm[4] / sum(cm[4], cm[3])
sensitivity <- cm[4] / sum(cm[4], cm[2])
fscore <- (2 * (sensitivity * precision))/(sensitivity + precision)
specificity <- cm[1] / sum(cm[1], cm[3])

table(accuracy, precision, sensitivity, specificity, fscore)

###############FOllowing are 3-fold all outcomes metrices#####################
# Stroke outcome, all variables
sensitivity = 0.954326923076923, 
specificity = 0.706293706293706, 
fscore = 0.928654970760234
precision = 0.904328018223235
accuracy = 0.89087656529517 

# Hemorrhagic outcome, all stroke variables plus stroke 
sensitivity = 0.322033898305085 
specificity = 0.897905759162304 
fscore = 0.417582417582418
precision = 0.59375
accuracy = 0.715563506261181

# Ischemic outcome, all stroke variables plus stroke 
sensitivity = 0.516393442622951 
specificity = 0.311111111111111 
fscore = 0.429301533219762
precision = 0.36734693877551
accuracy = 0.400715563506261 

# One sided face outcome, all stroke variables plus stroke 
sensitivity = 0.428571428571429
specificity = 0.711926605504587 
fscore = 0.0677966101694915
precision = 0.0368098159509202
accuracy = 0.704830053667263

# One sided arm outcome, all stroke variables plus stroke 
sensitivity = 0.702917771883289
specificity = 0.543956043956044
fscore = 0.731034482758621
precision = 0.761494252873563
accuracy = 0.651162790697674 

# One sided leg outcome, all stroke variables plus stroke 
sensitivity = 0.612732095490716
specificity = 0.598901098901099
fscore = 0.6784140969163
precision = 0.759868421052632
accuracy = 0.608228980322004  

# Asymmetry outcome, all stroke variables plus stroke 
sensitivity = 0.830238726790451
specificity = 0.412087912087912
fscore = 0.785445420326223
precision = 0.745238095238095
accuracy = 0.694096601073345  

# Not ambulatory outcome, all stroke variables plus stroke 
sensitivity = 0.458563535911602
specificity = 0.666666666666667
fscore = 0.425641025641026
precision = 0.397129186602871
accuracy = 0.599284436493739 

# Not able to speak outcome, all stroke variables plus stroke 
sensitivity = 0.5
specificity = 0.552278820375335
fscore = 0.417040358744
precision = 0.357692307692308
accuracy = 0.534883720930233

# Visual disturbances outcome, all stroke variables plus stroke 
sensitivity = 0.5
specificity = 0.998168498168498
fscore = 0.5
precision = 0.5
accuracy = 0.996350364963504 

# Abnormal sensation outcome, all stroke variables plus stroke 
sensitivity = 0.3
specificity = 0.859744990892532
fscore = 0.066666666666
precision = 0.0375
accuracy = 0.849731663685152 

# Mental change outcome, all stroke variables plus stroke 
sensitivity = 0.573333333333333
specificity = 0.861570247933884
fscore = 0.464864864864865
precision = 0.390909090909091
accuracy = 0.822898032200358

# Not able to grasp outcome, all stroke variables plus stroke 
sensitivity = 0.04
specificity = 0.847107438016529
fscore = 0.03947368421
precision = 0.038961038961039
accuracy = 0.738819320214669

###############FOllowing are 5-fold stroke outcomes metrices#####################
sensitivity = 0.948
specificity = 0.716981132075472
fscore = 0.9303238469
precision = 0.913294797687861
accuracy = 0.892261001517451 

###############FOllowing are 10-fold stroke outcomes metrices#####################
sensitivity = 0.946564885496183
specificity = 0.745901639344262
fscore = 0.934673366834171
precision = 0.923076923076923
accuracy = 0.899029126213592 

###################Check Stroke 3-fold accuracy plot on different lr's################

lr 0.19 Accuracy				0.885509838998211
lr 0.15 Accuracy				0.887298747763864
lr 0.1 Accuracy				0.899821109123435
lr 0.01 Accuracy				0.898032200357782 
lr 0.001 Accuracy				0.894454382826476
lr 0.0001 Accuracy				0.894454382826476 
lr 0.00001 Accuracy				0.899821109123435

t <- c(0.00001, 0.0001, 0.001, 0.01, 0.1, 0.15, 0.19)
z <- c(0.899821109123435, 0.894454382826476, 0.894454382826476, 0.898032200357782,0.899821109123435,0.887298747763864, 0.885509838998211)  
plot(t,z, type="l", col="green", lwd=5, pch=15, xlab="Learning rate", ylab="Accuracy")





