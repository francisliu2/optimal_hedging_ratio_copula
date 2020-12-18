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

#library(rugarch)



# SP and SPF
SP = read_excel("D:/Project/project/project/project/SRM/SRM data/2020/2020/SP500_2020.xlsx", range = "C7046:C7827")#C3916:C5219
SPF = read_excel("D:/Project/project/project/project/SRM/SRM data/2020/2020/SP500FUT_2020.xlsx", range = "C7046:C7827")  #C3:C7827

colnames(SP)="SP"
colnames(SPF)="SPF"

# # ##SP degarch
# SP1   = data.frame(SP)
# 
# spec = ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(1,1)),
#                   mean.model = list(armaOrder = c(1,1), include.mean = TRUE),
#                   distribution.model = "std")
# 
# fit = ugarchfit(data = SP1, spec = spec)
# 
# SP.std = sqrt(uncvariance(fit))
# 
# SP1_degarch = SP1/(SP.std)
# 
# SP    = as.numeric(unlist(SP1_degarch))

# # ##SPF degarch 
# SPF1   = data.frame(SPF)
# 
# spec = ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(1,1)),
#                   mean.model = list(armaOrder = c(1,1), include.mean = TRUE),
#                   distribution.model = "std")
# 
# fit = ugarchfit(data = SPF1, spec = spec)
# 
# SPF.std = sqrt(uncvariance(fit))
# 
# SPF1_degarch = SPF1/(SPF.std)
# 
# SPF    = as.numeric(unlist(SPF1_degarch))

SP    = as.numeric(unlist(SP))
SPF    = as.numeric(unlist(SPF))
SP    = data.frame(SP)
SPF    = data.frame(SPF)

data = cbind(SP, SPF)
data = as.matrix(data)

SP_copula = gof(data, priority = "tests", copula = c("normal", "t", "clayton", "gumbel", "frank"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 1000)

#gof(data, priority = "tests", copula = c("normal", "t"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 10)

# simulate normal copula 
require(mvtnorm)
S <- matrix(c(1,.8,.8,1),2,2)
AB <- rmvnorm(mean=c(0,0),sig=S,n=1000)

SP_copula = gof(AB, priority = "tests", copula = c("normal", "t", "clayton", "gumbel", "frank"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 1000)
SP_copula_sim = unlist(SP_copula)
write.xlsx(SP_copula_sim, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\SP_copula_sim.xlsx")

