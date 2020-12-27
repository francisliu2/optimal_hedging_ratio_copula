install.packages("xtable")
install.packages("xlsx", dependencies = TRUE)
install.packages("dplyr")
install.packages("Rcpp")

library(xtable)
library(dplyr)
#Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre-9.0.1')
Sys.setenv(JAVA_HOME='/System/Library/Frameworks/JavaVM.framework/Versions/Current')
#Sys.setenv(JAVA_HOME="/Library/Java/JavaVirtualMarchines/jdk-11.jdk/Contents/Home/")

library(xlsx)
library(readxl)

library(gofCopula)
library(dplyr)



# SP and SPF
#SP = read_excel("../data/SP500_2020.xlsx", range = "C7046:C7827")#C3916:C5219
#SPF = read_excel("../data/SP500FUT_2020.xlsx", range = "C7046:C7827")  #C3:C7827

#data=read.csv("../data/sp500.csv")
data = read.csv("../simulated_data.csv")

#colnames(SP)="SP"
#colnames(SPF)="SPF"

#SP    = as.numeric(unlist(SP))
#SP    = data.frame(SP)
#SPF   = as.numeric(unlist(SPF))
#SPF    = data.frame(SPF)

#data = cbind(SP, SPF)
data = as.matrix(data)

SP_copula = gof(data, priority = "tests", copula = c("normal", "t", "clayton", "gumbel", "frank"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 1000)

#gof(data, priority = "tests", copula = c("normal", "t"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 10)
SP_copula_3y = unlist(SP_copula)
write.xlsx(SP_copula_3y, "D:/Git_copula/optimal_hedging_ratio_copula/parameter_estimation/SP_copula_3y.xlsx")
