install.packages("xtable")
install.packages("xlsx", dependencies = TRUE)
install.packages("dplyr")
install.packages("Rcpp")

library(xtable)
library(dplyr)
#Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre-9.0.1')

library(xlsx)
library(readxl)

library(gofCopula)
library(dplyr)



# SP and SPF
SP = read_excel("D:/Project/project/project/project/SRM/SRM data/2020/2020/SP500_2020.xlsx", range = "C3:C7827")#C3916:C5219
SPF = read_excel("D:/Project/project/project/project/SRM/SRM data/2020/2020/SP500FUT_2020.xlsx", range = "C3:C7827")  #C3:C7827

colnames(SP)="SP"
colnames(SPF)="SPF"

SP    = as.numeric(unlist(SP))
SP    = data.frame(SP)
SPF   = as.numeric(unlist(SPF))
SPF    = data.frame(SPF)

data = cbind(SP, SPF)
data = as.matrix(data)

SP_copula = gof(data, priority = "tests", copula = c("normal", "t", "clayton", "gumbel", "frank"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 1000)

gof(data, priority = "tests", copula = c("normal", "t"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 10)
