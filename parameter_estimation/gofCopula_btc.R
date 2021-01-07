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
BRR = read_excel("D:/Git_copula/optimal_hedging_ratio_copula/parameter_estimation/data/BRR_Index.xlsx", range = "C3:C687")
BTC = read_excel("D:/Git_copula/optimal_hedging_ratio_copula/parameter_estimation/data/BTC_CME_Future.xlsx", range = "C3:C687") 

colnames(BRR)="BRR"
colnames(BTC)="BTC"

BRR    = as.numeric(unlist(BRR))
BTC    = as.numeric(unlist(BTC))
BRR    = data.frame(BRR)
BTC    = data.frame(BTC)

data = cbind(BRR, BRR)
data = as.matrix(data)

#BTC_copula_norm = gof(data, priority = "tests", copula = c("normal"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 100)
BTC_copula_norm = gof(data, priority = "tests", copula = c("normal"), M = 100)



BTC_copula_norm_1 = unlist(BTC_copula_norm)
write.xlsx(BTC_copula_norm_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\BTC_copula_norm.xlsx")

#gofGetHybrid(result = SP_copula_norm, p_values = c("MyTest" = 0.3, "AnotherTest" = 0.7), nsets = 2)
#BTC_copula_t = gof(data, priority = "tests", copula = c("t"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 100)
BTC_copula_t = gof(data, priority = "tests", copula = c("t"), M = 100)


BTC_copula_t_1 = unlist(BTC_copula_t)
write.xlsx(BTC_copula_t_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\BTC_copula_t.xlsx")

BTC_copula_gumbel = gof(data, priority = "tests", copula = c("gumbel"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 100)
BTC_copula_gumbel_1 = unlist(BTC_copula_gumbel)
write.xlsx(BTC_copula_gumbel_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\BTC_copula_gumbel.xlsx")

BTC_copula_frank = gof(data, priority = "tests", copula = c("frank"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 100)
BTC_copula_frank_1 = unlist(BTC_copula_frank)
write.xlsx(BTC_copula_frank_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\BTC_copula_frank.xlsx")

BTC_copula_clayton = gof(data, priority = "tests", copula = c("clayton"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 100)
BTC_copula_clayton_1 = unlist(BTC_copula_clayton)
write.xlsx(BTC_copula_clayton_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\BTC_copula_clayton.xlsx")

