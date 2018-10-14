#######################################################
# Project : Employeee Turnover Prediction in HR Analytics #
# Team    : Gautam HArinarayanan                      #
#           Mayank Gulati                             #
#           Carol Sun                                 #
#           Clifton Rains                             #
#           Tugce Sahan                               #
#######################################################

# Online version: https://tugcesahan.shinyapps.io/HR-Analytics/
# Loading necessary libraries

library(dplyr)
library(caret)
library(leaps)
library(data.table)
library(ROCR)
library(rpart)
library(rattle)
library(randomForest)
library(e1071)
library(pROC)
library(DMwR)
library(rpart)

library(rsconnect)
library(shiny)
library(ggplot2)
library(tidyr)
library(shinythemes)
library(scales)
library(wordcloud2)
library(shinydashboard)
#library(modeest)

# plot-ref:https://www.kaggle.com/esmaeil391/ibm-hr-analysis-with-90-3-acc-and-89-auc

################################################################################
# Data loading
################################################################################

# set memory limits
options(java.parameters = "-Xmx64048m") # 64048 is 64 GB

myUrl <- "https://raw.githubusercontent.com/tsahan/HR-Analytics-Shiny-App/master/HRA_Attrition.csv?token=AMbVxN89_S4pbI_12ci05DwddVPyULoAks5byi2GwA%3D%3D"
d<-read.csv(myUrl, sep="," ,header=T)

str(d)
dim(d)
head(d)
# coerce feature types to their correct types for analysis
d$Education <- factor(d$Education)
d$EnvironmentSatisfaction <- factor(d$EnvironmentSatisfaction)
d$JobInvolvement <- factor(d$JobInvolvement)
d$JobLevel <- factor(d$JobLevel)
d$JobSatisfaction <- factor(d$JobSatisfaction)
d$PerformanceRating <- factor(d$PerformanceRating)
d$RelationshipSatisfaction <- factor(d$RelationshipSatisfaction)
d$StockOptionLevel <- factor(d$StockOptionLevel)
d$WorkLifeBalance <- factor(d$WorkLifeBalance)

#let us first check for missing values
dim(d[!complete.cases(d),])[[1]]/nrow(d)*100

#source("DataQualityReport.R")
#DataQualityReport(d)

################################################################################
## Data clean up
################################################################################

# making Attrition the 'y' variable
names(d)[2] <- "y"

# moving 'y' to the front
d <-d %>%
  select(y, everything())

# let's do a summary of the table
summary(d)

# We can see that EmployeeCount, Over18 and StandardHours have repeated values throughout. We shall remove these columns.
# Also, EmployeeNumber is insignificant here as we cannot derive any details from it. We shall remove that too.
d <- subset(d, select = -c(EmployeeCount,Over18,StandardHours,EmployeeNumber))

# Let's convert Attrition to binary, Yes <- 1 and No <- 0
d$y <- ifelse(d$y == "Yes",1,0)
d$y <- as.factor(d$y)

shiny_d<-d

# ################################################################################
# ## Creating Dummy Variables
# ################################################################################
# 
# dummies <- dummyVars(y ~ ., data = d)            # create dummyes for Xs
# ex <- data.frame(predict(dummies, newdata = d))  # actually creates the dummies
# names(ex) <- gsub("\\.", "", names(ex))          # removes dots from col names
# d <- cbind(d$y, ex)                              # combine your target variable with Xs
# names(d)[1] <- "y"                               # make target variable called 'y'
# rm(dummies, ex)                                  # delete temporary things we no longer need
# 
# # We will remove one dummy variable from each factor variable
# d <- d[,-c(3,7,11,16,25,26,32,37,39,50,51,58,61,65,69,75)]
# 
# ################################################################################
# # Remove Zero- and Near Zero-Variance Predictors
# ################################################################################
# 
# dim(d) # dimension of dataset
# nzv <- nearZeroVar(d[,2:ncol(d)], uniqueCut=10) # identify columns that are "near zero"
# d_filtered <- d[,2:ncol(d)][, -nzv]            # remove those columns from your dataset
# dim(d_filtered)                                # dimension of your filtered dataset
# 
# # we can investigate some of the features that the nearZeroVar() function has
# # identified for removal
# setdiff(names(d), names(d_filtered))
# table(d$Education5)
# table(d$y)
# 
# # We can see that Education5 has mostly 0s (1422) compared to 1s(48). We shall remove this column as well so as to
# # not get the model biased.
# d <- cbind(d$y, d_filtered)   # combine y with the Xs
# names(d)[1] <- "y"            # fix the y variable name
# 
# rm(d_filtered, nzv)           # clean up  
# 
# ################################################################################
# # Identify Correlated Predictors and remove them
# ################################################################################
# 
# # calculate correlation matrix using Pearson's correlation formula
# descrCor <-  cor(d[,2:ncol(d)])                           # correlation matrix
# highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .85) # number of Xs having a corr > some value
# summary(descrCor[upper.tri(descrCor)])                    # summarize the correlations
# 
# highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.85)
# filteredDescr <- d[,2:ncol(d)][,-highlyCorDescr] # remove those specific columns from your dataset
# descrCor2 <- cor(filteredDescr)                  # calculate a new correlation matrix
# 
# # summarize those correlations to see if all features are now within our range
# summary(descrCor2[upper.tri(descrCor2)])
# 
# # update our d dataset by removing those filtered variables that were highly correlated
# d <- cbind(d$y, filteredDescr)
# names(d)[1] <- "y"
# 
# rm(filteredDescr, descrCor, descrCor2, highCorr, highlyCorDescr)  # clean up
# 
# ################################################################################
# # Identifying linear dependencies and remove them
# ################################################################################
# 
# # first save response
# y <- d$y
# 
# # create a column of 1s. This will help identify all the right linear combos
# d <- cbind(rep(1, nrow(d)), d[2:ncol(d)])
# names(d)[1] <- "ones"
# 
# # identify the columns that are linear combos
# comboInfo <- findLinearCombos(d)
# comboInfo
# 
# # There are no linear combos to remove
# # remove the "ones" column in the first column
# d <- d[, c(2:ncol(d))]
# 
# # Add the target variable back to our data.frame
# d <- cbind(y, d)
# 
# rm(y, comboInfo)  # clean up

################################################################################
# Standardize (and/ normalize) input features.
################################################################################

cat_data<-d[ ,sapply(d, is.factor)]
num_data<-d[ ,!sapply(d, is.factor)]

preProcValues <- preProcess(num_data, method = c("center","scale"))
d_Nums <- predict(preProcValues, num_data)
d <- cbind(d_Nums, cat_data)
str(d)
# moving 'y' to the front
d <-d %>%
  select(y, everything())

# ################################################################################
# 
# numcols <- apply(X=d, MARGIN=2, function(c) sum(c==0 | c==1)) != nrow(d)
# catcols <- apply(X=d, MARGIN=2, function(c) sum(c==0 | c==1)) == nrow(d)
# dNums <- d[,numcols]
# dCats <- d[,-numcols]
# 
# preProcValues <- preProcess(dNums[,2:ncol(dNums)], method = c("center","scale"))
# dNums <- predict(preProcValues, dNums)
# 
# # combine the standardized numeric features with the dummy vars
# d <- cbind(dNums, dCats)
# str(d)
# # moving 'y' to the front
# d <-d %>%
#   select(y, everything())
# 
# rm(preProcValues, numcols, catcols, dNums, dCats)  # clean up

################################################################################
# Data partitioning
################################################################################

set.seed(1234) # set a seed so you can replicate your results

inTrain <- createDataPartition(y = d$y,   # outcome variable
                               p = .80,   # % of training data you want
                               list = F)
# Creating partitions
train <- d[inTrain,]  # training data set
test <- d[-inTrain,]  # test data set

rm(inTrain)

################################################################################
# Automatic feature selection using using forward and backward selection
################################################################################

# automatic backward selection
mb <- regsubsets(y ~ ., data=train
                 , nbest=1
                 , intercept=T
                 , method='backward'
                 , really.big=T
)

vars2keep <- data.frame(summary(mb)$which[which.max(summary(mb)$adjr2),])
names(vars2keep) <- c("keep")  
head(vars2keep)

vars2keep <- setDT(vars2keep, keep.rownames=T)[]
vars2keep <- c(vars2keep[which(vars2keep$keep==T & vars2keep$rn!="(Intercept)"),"rn"])[[1]]

# here are the final features found to be statistically significant
vars2keep

# features to keep as per backward selection are Age, NumCompaniesWorked, Business Travel, Environment Satisfaction, 
# JobInvolvement, Job Level, OverTimeNo and StockOption

################################################################################
# Modelling - Logistic Regression
################################################################################

set.seed(1234)

attLog <- glm(y~.-1,data=train,family = binomial)

summary(attLog)

predGlm <- predict(attLog,type="response",newdata=test)

predGlm <- predicted<-ifelse(predGlm > 0.5,1,0)
predGlm <- as.factor(predGlm)

confusionMatrix(test$y,predGlm) # Accuracy 87.03% 


################################################################################
# Modelling - Decision Trees
################################################################################
set.seed(1234)

decisionTreeModel <- rpart(y~.,data=train,method="class",minbucket = 25)
#fancyRpartPlot(decisionTreeModel)

# Variables to keep as per Decision Tree are JobRole, Overtime, MaritalStatus and Monthly Income

predDT <- predict(decisionTreeModel,newdata = test,type = "class")

confusionMatrix(test$y,predDT) # Accuracy 85.32%

################################################################################
# Modelling - SVM
################################################################################

set.seed(1234)

tuned <- tune(svm,factor(y)~.,data = train)
svm.model <- svm(train$y~., data=train
                 ,type="C-classification", gamma=tuned$best.model$gamma
                 ,cost=tuned$best.model$cost
                 ,kernel="radial")

svm.prd <- predict(svm.model,newdata=test)
confusionMatrix(svm.prd,test$y) #Accuracy 84.3%

################################################################################
# Modelling - XGBoost
################################################################################

set.seed(123)
xgbData <- d

indexes <- createDataPartition(y = xgbData$y,   # outcome variable
                               p = .80,   # % of training data you want
                               list = F)

XGBtrain.Data <- xgbData[indexes,]
XGBtest.Data <- xgbData[-indexes,]

XGBtrain.Data$y <- ifelse(XGBtrain.Data$y == 1,"Yes","No")
XGBtest.Data$y <- ifelse(XGBtest.Data$y == 1,"Yes","No")

formula <- y~.
fitControl <- trainControl(method="cv", number = 3,classProbs = TRUE )
xgbGrid <- expand.grid(nrounds = 50,
                       max_depth = 12,
                       eta = .03,
                       gamma = 0.01,
                       colsample_bytree = .7,
                       min_child_weight = 1,
                       subsample = 0.9
)

XGB.model <- train(formula, data = XGBtrain.Data,
                   method = "xgbTree"
                   ,trControl = fitControl
                   , verbose=0
                   , maximize=FALSE
                   ,tuneGrid = xgbGrid
)


importance <- varImp(XGB.model)
xg.varImportance <- data.frame(Variables = row.names(importance[[1]]), 
                               Importance = round(importance[[1]]$Overall,2))

# Create a rank variable based on importance of variables
rankImportance <- xg.varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance)) +
  geom_bar(stat='identity',colour="white", fill = "lightgreen") +
  geom_text(aes(x = Variables, y = 1, label = Rank),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Variables', title = 'Relative Variable Importance') +
  coord_flip() + 
  theme_bw()

XGB.prd <- predict(XGB.model,XGBtest.Data)
XGBtest.Data$y <- as.factor(XGBtest.Data$y)

confusionMatrix(XGB.prd, XGBtest.Data$y) # Accuracy 85.03%

XGB.plot <- plot.roc(as.numeric(XGBtest.Data$y), as.numeric(XGB.prd),lwd=2, type="b", print.auc=TRUE,col ="blue")

Classcount <- table(xgbData$y)

# Before sampling output
d %>%
  group_by(y) %>%
  tally() %>%
  ggplot(aes(x = y, y = n,fill=y)) + guides(fill=guide_legend(title="Attrition")) +
  geom_bar(stat = "identity") +
  theme_minimal()+
  labs(x="Attrition", y="Count of Attriation")+
  ggtitle("Attrition (Before Balancing)")+
  geom_text(aes(label = n), vjust = -0.5, position = position_dodge(0.9))

#########################################################################
#Create Balanced sample
#########################################################################

# Over Sampling
over <- ( (0.6 * max(Classcount)) - min(Classcount) ) / min(Classcount)

# Under Sampling
under <- (0.4 * max(Classcount)) / (min(Classcount) * over)
over <- round(over, 1) * 100
under <- round(under, 1) * 100

#Generate the balanced data set
BalancedData <- SMOTE(y~., d, perc.over = over, k = 5, perc.under = under)

# let check the output of the Balancing
BalancedData %>%
  group_by(y) %>%
  tally() %>%
  ggplot(aes(x = y, y = n,fill=y)) +
  geom_bar(stat = "identity") + guides(fill=guide_legend(title="Attrition")) +
  theme_minimal()+
  labs(x="Attrition", y="Count of Attriation")+
  ggtitle("Attrition (After Balancing)")+
  geom_text(aes(label = n), vjust = -0.5, position = position_dodge(0.9))

#########################################################################
# XGBOOST with Balanced sample
#########################################################################

#With balanced data
set.seed(123)
xgbData <- BalancedData

indexes <- createDataPartition(y = xgbData$y,   # outcome variable
                               p = .80,   # % of training data you want
                               list = F)
BLtrain.Data <- xgbData[indexes,]
BLtest.Data <- xgbData[-indexes,]

formula <- y~.

fitControl <- trainControl(method="cv", number = 3,classProbs = TRUE )

xgbGrid <- expand.grid(nrounds = 500,
                       max_depth = 20,
                       eta = .03,
                       gamma = 0.01,
                       colsample_bytree = .7,
                       min_child_weight = 1,
                       subsample = 0.9
)

BLtrain.Data$y <- ifelse(BLtrain.Data$y == 1,"Yes","No")
BLtest.Data$y <- ifelse(BLtest.Data$y == 1,"Yes","No")

BLtrain.Data$y <- as.factor(BLtrain.Data$y)

Bal_XGB.model <- train(formula, data = BLtrain.Data,
                       method = "xgbTree"
                       ,trControl = fitControl
                       ,verbose=0
                       ,maximize=FALSE
                       ,tuneGrid = xgbGrid
)

NewXGB.prd <- predict(Bal_XGB.model,BLtest.Data)

BLtest.Data$y <- as.factor(BLtest.Data$y)

confusionMatrix(NewXGB.prd, BLtest.Data$y) # Accuracy 88.14%

# Create a rank variable based on importance
importance <- varImp(Bal_XGB.model)
varImportance <- data.frame(Variables = row.names(importance[[1]]), 
                            Importance = round(importance[[1]]$Overall,2))

rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance)) +
  geom_bar(stat='identity',colour="white", fill = "lightgreen") +
  geom_text(aes(x = Variables, y = 1, label = Rank),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Variables', title = 'Relative Variable Importance') +
  coord_flip() + 
  theme_bw()


XGB.plot <- plot.roc (as.numeric(BLtest.Data$y), as.numeric(NewXGB.prd),lwd=2, type="b", print.auc=TRUE, col ="blue")

par(mfrow=c(2,3))
plot.roc (as.numeric(XGBtest.Data$y), as.numeric(XGB.prd),main="XGBoost",lwd=2, type="b", print.auc=TRUE, col ="blue")
plot.roc (as.numeric(test$y), as.numeric(predDT), main="Decision Tree",lwd=2, type="b", print.auc=TRUE, col ="brown")
plot.roc (as.numeric(BLtest.Data$y), as.numeric(NewXGB.prd),main="New XGBoost",lwd=2, type="b", print.auc=TRUE, col ="green")
plot.roc (as.numeric(test$y), as.numeric(svm.prd),main="SVM",lwd=2, type="b", print.auc=TRUE, col ="red")
plot.roc (as.numeric(test$y), as.numeric(predGlm),main="Logistic Regression",lwd=2, type="b", print.auc=TRUE, col ="magenta")

# After performing Logistic Regression, Decision Trees, SVM and XGBoost, we notice that accuracy is highest in 
# XGBoost on a balanced dataset, with Accuracy of 90.06%.

write.csv(NewXGB.prd,'testpredictions_xgb.csv')

################################################################################
#                                     R Shiny                                      #
################################################################################

turnOverRate = dim(shiny_d[shiny_d$y == 1,])[1]/dim(d)[1]

#test_df<-d[1,]
#cat_data<-d[ ,sapply(d, is.factor)]
num_data<-d[ ,!sapply(d, is.factor)]
num_d<- as.data.frame(sapply(num_data,MARGIN = 2,FUN=mean))

test_df<-data.frame(t(num_d[,1]))
colnames(test_df) <- rownames(num_d)

test_df$y<-as.factor("Yes")
test_df$BusinessTravel<-as.factor("Travel_Rarely")
test_df$Department<-as.factor("Research & Development")
test_df$Education<-as.factor("3")
test_df$EducationField<-as.factor("Life Sciences")
test_df$EnvironmentSatisfaction<-as.factor("3")
test_df$Gender<-as.factor("Male")
test_df$JobInvolvement<-as.factor("3")
test_df$JobLevel<-as.factor("1")
test_df$JobRole<-as.factor("Sales Executive")
test_df$JobSatisfaction<-as.factor("4")
test_df$MaritalStatus<-as.factor("Married")
test_df$OverTime<-as.factor("No")
test_df$PerformanceRating<-as.factor("3")
test_df$RelationshipSatisfaction<-as.factor("3")
test_df$StockOptionLevel<-as.factor("0")
test_df$WorkLifeBalance<-as.factor("3")


ui <-  dashboardPage(skin = "red",
                     dashboardHeader(title = 'Employee Turnover Prediction', titleWidth = 400),
                     dashboardSidebar(disable = TRUE),
                     dashboardBody(
                       theme = shinytheme("simplex"),
                       # tags$h1("Employee Turnover Rate Prediction", align = "center"),
                       tags$head(
                         tags$style(HTML("
                                         h1 {
                                         font-weight: bold;
                                         line-height: 1.1;
                                         color: #35A3E2;
                                         }
                                         "))
                         ),
                       tabsetPanel(
                         tabPanel("Word Cloud", wordcloud2Output('wordcloud2',width = "100%")),
                         tabPanel("EDA", plotOutput("graphByFactor"), selectInput("var", tags$br(), "Choose a factor to explore:", 
                                                                                  choices=list('Business Travel'='BusinessTravel', 'Department', 'Education', 'Education Field'='EducationField',
                                                                                               'Environment Satisfaction'='EnvironmentSatisfaction','Gender','Job Involvement'='JobInvolvement',
                                                                                               'Job Level'='JobLevel', 'Job Role'='JobRole', 'Job Satisfaction'='JobSatisfaction',
                                                                                               'Marital Status'='MaritalStatus','Over Time'= 'OverTime', 'Performance Rating'='PerformanceRating',
                                                                                               'Relationship Satisfaction'='RelationshipSatisfaction','Stock Option Level'='StockOptionLevel',
                                                                                               'Work Life Balance'='WorkLifeBalance'))),
                         tabPanel("Predictive Analytics",
                                  sidebarPanel(
                                    
                                    selectInput("JobInv", "Job Involvement", 
                                                choices=list('Low'=1,'Medium'=2,'High'=3,'Very High'=4)),
                                    selectInput("JobLev", "Job Level", 
                                                choices=list('Entry'=1,'Intermediate'=2,'Experienced'=3,'Advanced'=4,'Expert'=5)),
                                    sliderInput(inputId = "MonthlyIncome", 
                                                label = "Monthly Income", 
                                                value = mean(shiny_d$MonthlyIncome), min = 2000, max = 20000, step = 1000),
                                    sliderInput(inputId = "DistanceFromHome", 
                                                label = "Distance from home", 
                                                value = mean(shiny_d$DistanceFromHome), min = min(shiny_d$DistanceFromHome), max = max(shiny_d$DistanceFromHome), step = 5),
                                    sliderInput(inputId = "age", 
                                                label = "Age", 
                                                value = mean(shiny_d$Age), min = min(shiny_d$Age), max = max(shiny_d$Age), step = 1),
                                    sliderInput(inputId = "YearsAtCompany", 
                                                label = "Years at Company", 
                                                value = mean(shiny_d$YearsAtCompany), min = min(shiny_d$YearsAtCompany), max = 20, step = 1),
                                    sliderInput(inputId = "TotalWorkingYears", 
                                                label = "Total Working Years", 
                                                value = mean(shiny_d$TotalWorkingYears), min = min(shiny_d$TotalWorkingYears), max = max(shiny_d$TotalWorkingYears), step = 1),
                                    
                                    radioButtons("Overtime", label = "Overtime",
                                                 choices = list("Yes", "No"), 
                                                 selected = "Yes"),
                                    actionButton(inputId = "submit",
                                                 label = "Update")
                                    
                                    ),
                                 
                                 mainPanel( plotOutput("plotPredict"), tags$br(),
                                            infoBox("Current Turnover Rate",verbatimTextOutput("attrate"),fill = TRUE , width = 6, color = "light-blue", icon = icon("info-circle")),  
                                            infoBox("The Probability of Employee Leaving",verbatimTextOutput("predrate"),fill = TRUE , width = 6, color = "light-blue", icon = icon("info-circle"))
                                            
                                  )
                         )
                       )
                         )
                       )

server <- function(input, output) {
  predicted <- eventReactive(input$submit, {
    
    #test_df$y <- ifelse(test_df$y == 1,"Yes","No")
    test_df$y<-as.factor(test_df$y)
    
    test_df$Age<-(input$age-mean(shiny_d$Age))/sd(shiny_d$Age)
    test_df$YearsAtCompany<-(input$YearsAtCompany-mean(shiny_d$YearsAtCompany))/sd(shiny_d$YearsAtCompany)
    test_df$JobInvolvement<-as.factor(input$JobInv)
    test_df$JobLevel<-as.factor(input$JobLev)
    test_df$Overtime<-as.factor(input$Overtime)
    test_df$MonthlyIncome<-(input$MonthlyIncome-mean(shiny_d$MonthlyIncome))/sd(shiny_d$MonthlyIncome)
    test_df$DistanceFromHome<-(input$DistanceFromHome-mean(shiny_d$DistanceFromHome))/sd(shiny_d$DistanceFromHome)
    test_df$TotalWorkingYears<-(input$TotalWorkingYears-mean(shiny_d$TotalWorkingYears))/sd(shiny_d$TotalWorkingYears)
    
    
    #the function to find value of the response variable
    #pre_t <- predict(Bal_XGB.model, test_df, type='prob')
    pre_t <- predict(Bal_XGB.model, test_df, type='prob')
    fun1 <- pre_t[[1]]
    fun2 <- pre_t[[2]]
    prob_val<-c(fun1,fun2)
    cate<-c("No","Yes")
    
    out_data<-data.frame(cate,prob_val)
    return(out_data)
  })
  pp <- eventReactive(input$submit,{
    ggplot(predicted(), aes(cate, prob_val)) +  
      geom_bar(stat = "identity",aes(fill = cate)) + 
      scale_y_continuous(labels = scales::percent,limits = c(0,1))+
      theme_minimal()+
      labs(x="Turnover", y="Percentage of Turnover")
    
  })
  output$plotPredict <- renderPlot({
    pp()
  })
  
  output$graphByFactor <- renderPlot({
    shiny_d %>%
      ggplot(aes_string(input$var, group = "y")) + 
      geom_bar(aes(y = ..prop.., fill = factor(..x..)), 
               stat="count", 
               alpha = 0.7) +
      geom_text(aes(label = scales::percent(..prop..), y = ..prop.. ), 
                stat= "count", 
                vjust = -.5) +
      labs(y = "Percentage", fill= input$var) +
      facet_grid(~y) +
      theme_minimal()+
      theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) + 
      ggtitle("Turnover")
  })
  
  output$wordcloud2 <- renderWordcloud2({
    dataWeights <- data.frame("word" = c('Age',	'Turnover',	'Business Travel',	'Daily Rate',	'Department',	'Distance From Home',	'Education',	'Education Field',	'Environment Satisfaction',	'Gender',	'Hourly Rate',	
                                         'Job Involvement',	'Job Level',	'Job Role',	'Job Satisfaction',	'Marital Status',	'Monthly Income',	'Monthly Rate', 'Num of Companies Worked',' Over Time',	'Percent Salary Hike',	
                                         'Performance Rating',	'Relationship Satisfaction',	'Standard Hours',	'Stock Option Level',	'Total Working Years',	'Training Times Last Year',	'Work Life Balance',	'Years At Company',	
                                         'Years In Current Role',	'Years Since Last Promotion',	'Years With Current Manager'),
                              "freq" = c(5,5,4,4,7,2,2,6,1,2,7,8,7,4,4,7,1,9,9,1,2,2,1,5,8,1,3,5,1,2,1))
    wordcloud2(dataWeights, color = "random-light", size=0.5)
  })
  output$attrate <- renderText({
    paste(percent(turnOverRate))
  })
  output$predrate <- renderText({
    paste(percent(predicted()[2,2]))
  })
}

shinyApp(ui = ui, server = server)
