install.packages("xtable")
install.packages("xlsx", dependencies = TRUE)
install.packages("Rcpp")
install.packages("dplyr")

install.packages("gofCopula")

library(xtable)
library(dplyr)
#Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre-9.0.1')

library(xlsx)
library(readxl)

library(gofCopula)

# SP and SPF
SP = read_excel("D:/Project/project/project/project/SRM/SRM data/2020/2020/SP500_2020.xlsx", range = "C7046:C7827")#C3916:C5219
SPF = read_excel("D:/Project/project/project/project/SRM/SRM data/2020/2020/SP500FUT_2020.xlsx", range = "C7046:C7827")  #C3:C7827

colnames(SP)="SP"
colnames(SPF)="SPF"

SP    = as.numeric(unlist(SP))
SPF    = as.numeric(unlist(SPF))
SP    = data.frame(SP)
SPF    = data.frame(SPF)

data = cbind(SP, SPF)
data = as.matrix(data)

SP_copula_norm_3y = gof(data, priority = "tests", copula = c("normal", "t", "gumbel","frank","clayton"), tests = c("gofKernel", "gofKendallCvM", "gofKendallKS"), M = 100)

SP_copula_3y_1 = unlist(SP_copula_norm_3y)
write.xlsx(SP_copula_3y_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\SP_copula_3y_20210107.xlsx")

#gofGetHybrid(result = SP_copula_norm, p_values = c("MyTest" = 0.3, "AnotherTest" = 0.7), nsets = 2)
SP_copula_t_3y = gof(data, priority = "tests", copula = c("t"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 100)
SP_copula_t_3y_1 = unlist(SP_copula_t_3y)
write.xlsx(SP_copula_t_3y_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\SP_copula_t_3y.xlsx")

SP_copula_gumbel_3y = gof(data, priority = "tests", copula = c("gumbel"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 100)
SP_copula_gumbel_3y_1 = unlist(SP_copula_gumbel_3y)
write.xlsx(SP_copula_gumbel_3y_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\SP_copula_gumbel_3y.xlsx")

SP_copula_frank_3y = gof(data, priority = "tests", copula = c("frank"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 100)
SP_copula_frank_3y_1 = unlist(SP_copula_frank_3y)
write.xlsx(SP_copula_frank_3y_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\SP_copula_frank_3y.xlsx")

SP_copula_clayton_3y = gof(data, priority = "tests", copula = c("clayton"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 100)
SP_copula_clayton_3y_1 = unlist(SP_copula_clayton_3y)
write.xlsx(SP_copula_clayton_3y_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\SP_copula_clayton_3y.xlsx")
