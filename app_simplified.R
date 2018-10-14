#######################################################
# Project : Employee Turnover Prediction in HR Analytics #
# Team    : Gautam HArinarayanan                      #
#           Mayank Gulati                             #
#           Carol Sun                                 #
#           Clifton Rains                             #
#           Tugce Sahan                               #
#######################################################

# Loading necessary libraries
# Works with no errors in R Studio Server.

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


# plot-ref:https://www.kaggle.com/esmaeil391/ibm-hr-analysis-with-90-3-acc-and-89-auc

################################################################################
# Data loading
################################################################################

# set memory limits
options(java.parameters = "-Xmx64048m") # 64048 is 64 GB

myUrl <- "https://raw.githubusercontent.com/tsahan/HR-Analytics-Shiny-App/master/HRA_Attrition.csv?token=AMbVxN89_S4pbI_12ci05DwddVPyULoAks5byi2GwA%3D%3D"
d<-read.csv(myUrl, sep="," ,header=T)

str(d)

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

predGlm_res <- predicted<-ifelse(predGlm > 0.5,1,0)
predGlm_res <- as.factor(predGlm_res)

confusionMatrix(test$y,predGlm_res) # Accuracy 86.69% 


################################################################################
#                                     R Shiny                                      #
################################################################################

turnOverRate = dim(shiny_d[shiny_d$y == 1,])[1]/dim(d)[1]

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
                                                value = mean(shiny_d$DistanceFromHome), min = min(shiny_d$DistanceFromHome), max = 30, step = 5),
                                    sliderInput(inputId = "age", 
                                                label = "Age", 
                                                value = mean(shiny_d$Age), min = min(shiny_d$Age), max = max(shiny_d$Age), step = 2),
                                    sliderInput(inputId = "YearsAtCompany", 
                                                label = "Years at Company", 
                                                value = mean(shiny_d$YearsAtCompany), min = min(shiny_d$YearsAtCompany), max = 20, step = 2),
                                    sliderInput(inputId = "NumCompaniesWorked", 
                                                label = "Number of Companies Worked", 
                                                value = mean(shiny_d$NumCompaniesWorked), min = min(shiny_d$NumCompaniesWorked), max = max(shiny_d$NumCompaniesWorked), step = 2),
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
    test_df$OverTime<-as.factor(input$Overtime)
    test_df$MonthlyIncome<-(input$MonthlyIncome-mean(shiny_d$MonthlyIncome))/sd(shiny_d$MonthlyIncome)
    test_df$DistanceFromHome<-(input$DistanceFromHome-mean(shiny_d$DistanceFromHome))/sd(shiny_d$DistanceFromHome)
    test_df$NumCompaniesWorked<-(input$NumCompaniesWorked-mean(shiny_d$NumCompaniesWorked))/sd(shiny_d$NumCompaniesWorked)
    
    
    #the function to find value of the response variable
    
    pre_t <- predict(attLog, test_df, type='response')
    fun1 <- 1-pre_t[[1]]
    fun2 <- pre_t[[1]]
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
    dataWeights <- data.frame("word" = c('Age',	'Business Travel',	'Daily Rate',	'Department',	'Distance From Home',	'Education',	'Education Field',	'Environment Satisfaction',	'Gender',	'Hourly Rate',	
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