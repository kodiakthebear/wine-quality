#Loading the requisite packages:
library(ggplot2)
library(gridExtra)
library(naniar)
library(dplyr)
library(tibble)
library(GGally)
library(rpart)
library(ggparty)
library(caret)
library(randomForest)
library(corrplot)
install.packages("readxl")

#Loading the datasets into R:
rohil <- read.csv("/Users/mukundranjan/Documents/Academics/Easter/Rohil Assignment/BA2.csv")

#Merging the datasets to create a unified data set for simultaneous analysis, with an added wine_type column 
#with 0 indicating a red wine and 1 indicating a white wine:
redwine$wine_type <- 0
whitewine$wine_type <- 1
wine <- rbind(redwine, whitewine)
head(wine)
#EDA (various):
wine
#Structural analysis:
str(wine)

#Checking for missing values:
missVal <- sum(is.na(wine))
print(missVal)
plot(colSums(is.na(wine)))
plot(rowSums(is.na(wine)))

wine_summary <- summary(wine)
print(wine_summary)

#Variable Distribution Study via Histograms: 
par(mfrow=c(1,1))
hist(wine$fixed.acidity, xlab = "fixed.acidity", main = "Histogram of fixed.acidity")
hist(wine$volatile.acidity, xlab = "volatile.acidity", main = "Histogram of volatile.acidity")
hist(wine$citric.acid, xlab = "citric.acid", main = "Histogram of citric.acid")
hist(wine$residual.sugar, xlab = "residual.sugar", main = "Histogram of residual.sugar")
hist(wine$chlorides, xlab = "chlorides", main = "Histogram of chlorides")
hist(wine$free.sulfur.dioxide, xlab = "free.sulfur.dioxide", main = "Histogram of free.sulfur.dioxide")
hist(wine$total.sulfur.dioxide, xlab = "total.sulfur.dioxide", main = "Histogram of total.sulfur.dioxide")
hist(wine$density, xlab = "density", main = "Histogram of density")
hist(wine$pH, xlab = "pH", main = "Histogram of pH")
hist(wine$sulphates, xlab = "sulphates", main = "Histogram of Sulphates")
hist(wine$alcohol, xlab="alcohol", main = "Histogram of Alcohol")
hist(wine$quality, xlab = "quality", main = "Histogram of Quality")
hist(wine$wine_type)

#Correlation heatmap:
cormatrix <- cor(wine[ , !(names(wine) %in% c("wine_type", "quality_label"))], use = "complete.obs")
corrplot(cormatrix, method="color", type="upper", tl.col="black", tl.srt=45, addCoef.col="black")

#Boxplot of quality:
boxplot(wine$quality, col = "dodgerblue", main="Boxplot of Wine Quality Ratings")

#Overall Distribution of Wine by Quality:

ggplot(wine, aes(x = quality, color = as.factor(wine_type), fill = as.factor(wine_type))) +
  geom_density(alpha = 0.3) +
  scale_fill_manual(values = c("red", "dodgerblue"), labels = c("Red Wine", "White Wine")) +
  scale_color_manual(values = c("red", "dodgerblue"), labels = c("Red Wine", "White Wine")) + theme_minimal() + labs(title = "Density Plot of Wine Quality Distribution", x = "Quality Score", y = "Density", fill = "Wine Type", color = "Wine Type")



#Checking all data types:
sapply(wine, class) 

#Checking for duplicates:
sum(duplicated(wine))

#Scatterplot matrices for red and white wine respectively: (Replace w workshop scatterplots)
ggpairs(wine, columns = c("quality", "fixed.acidity", "volatile.acidity", "citric.acid", "residual.sugar", "chlorides", "free.sulfur.dioxide", "total.sulfur.dioxide", "density", "pH", "sulphates", "alcohol"), title = "Scatterplot Matrix of Wine Data") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8))

ggpairs(redwine, columns = c("quality", "fixed.acidity", "volatile.acidity", "citric.acid", "residual.sugar", "chlorides", "free.sulfur.dioxide", "total.sulfur.dioxide", "density", "pH", "sulphates", "alcohol"), title = "Scatterplot Matrix of Red Wine Data") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8))

ggpairs(whitewine, columns = c("quality", "fixed.acidity", "volatile.acidity", "citric.acid", "residual.sugar", "chlorides", "free.sulfur.dioxide", "total.sulfur.dioxide", "density", "pH", "sulphates", "alcohol"), title = "Scatterplot Matrix of White Wine Data") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8))


#From the scatterplot matrices it is clear that alcohol and volatile.acidity have a high correlation with wine quality across both 
#Red and White wine. We will visualise this relationship using ggplot:

ggplot(wine, aes(x=alcohol,y=quality))+
  geom_point(aes(colour=factor(wine_type)))+
  xlab("Alcohol Content")+
  ylab("Wine Quality")+
  labs(colour="")+
  scale_color_manual(values=c("coral4","goldenrod"),labels=c("Red","White"))

ggplot(wine, aes(x=volatile.acidity,y=quality))+
  geom_point(aes(colour=factor(wine_type)))+
  xlab("Volatile Acidity")+
  ylab("Wine Quality")+
  labs(colour="")+
  scale_color_manual(values=c("coral4","goldenrod"),labels=c("Red","White"))

#Preparing the data for modelling:
wine$quality_label <- ifelse(wine$quality>=6, "high","low")
wine$quality_label <- as.factor(wine$quality_label)

#Model 1 - Decision Tree for wine classification

#Building and evaluating the classification tree:
winetree <- rpart(quality_label ~ . - quality - wine_type, data = wine, control = rpart.control(minsplit = 10, minbucket = 3, cp = 0.01, maxdepth = 10), method = "class")
plot(as.party(winetree))
printcp(winetree)

#Evaluating the tree before any pruning:
predict1 <- predict(winetree, type = "class")
cmatrix1 <- confusionMatrix(predict1, wine$quality_label)
print(cmatrix1)

accuracy1 <- cmatrix1$overall["Accuracy"]
precision1 <- cmatrix1$byClass["Precision"]
recall1 <- cmatrix1$byClass["Recall"]
f1_score1 <- 2 * ((precision1 * recall1) / (precision1 + recall1))

print(paste("Accuracy:", accuracy1))
print(paste("Precision:", precision1))
print(paste("Recall:", recall1))
print(paste("F1-score:", f1_score1))



#Pruning our tree using the Best CP:
opt = which.min(winetree$cptable[,"xerror"])
cp = winetree$cptable[opt, "CP"]
pruned_winetree = prune(winetree,cp)
plot(as.party(pruned_winetree))
autoplot(as.party(pruned_winetree))

#Evaluating Tree Performance:
predictions <- predict(pruned_winetree, type = "class")
cmatrix <- confusionMatrix(predictions, wine$quality_label)
print(cmatrix)

accuracy <- cmatrix$overall["Accuracy"]
precision <- cmatrix$byClass["Precision"]
recall <- cmatrix$byClass["Recall"]
f1_score <- 2 * ((precision * recall) / (precision + recall))

print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1-score:", f1_score))




#Model 2 - Random Forest
#We will now begin the procedure to build a Random Forest using the wine dataset. 

set.seed(42)
ind = sample(1:nrow(wine), size=2000)
test_wine = wine[ind,]
train_wine = wine[-ind,]
winerf <- randomForest(factor(quality_label) ~ .-quality, data = train_wine, ntree=10000, mtry=3, proximity=TRUE)
winerf

#Testing the predictive capabilities of our model on the test set:
winerf_pred <- predict(winerf, newdata = test_wine)
confusionMatrix(winerf_pred, factor(test_wine$quality_label))


#Visualising the RF outputs:
plot(winerf)

varImpPlot(winerf, sort = TRUE, pch=16)

MDSplot(winerf, train_wine$quality_label)



ggplot(train_wine, aes(x = alcohol, y = volatile.acidity, color = factor(quality_label))) +
  geom_point(alpha = 0.7) +
  theme_bw()               #Visualising the relationship between volatile.acidity and alcohol as
                                #they are the two most important variables. 


ggplot(train_wine, aes(x = alcohol, y = density, color = factor(quality_label))) +
  geom_point(alpha = 0.7) +
  theme_bw()                #Visualising the relationship between density and alcohol as
                                 #they are amongst the three most important variables.

ggplot(train_wine, aes(x = volatile.acidity, y = density, color = factor(quality_label))) +
  geom_point(alpha = 0.7) +
  theme_bw()                #Visualising the relationship between density and alcohol as
                                 #they are amongst the three most important variables.



#We had initially started with a simple binary classification problem. We will now try predicting wine quality
#As a whole, on just quality labels. For this, we will use a regression model with quality as the target varible. 

winerf_reg <- randomForest(quality ~., data=train_wine, ntree=1000, mtry=3)
winerf_reg

plot(winerf_reg)



winerf_reg_pred <- test_wine %>% mutate(pred_rf = predict(winerf_reg, newdata = test_wine))
RMSE(winerf_reg_pred$pred_rf, winerf_reg_pred$quality)
ggplot(winerf_reg_pred) +
  geom_point(aes(x=quality, y=pred_rf)) +
  geom_abline(intercept = 0,slope = 1, colour = "red") +
  theme_bw()


