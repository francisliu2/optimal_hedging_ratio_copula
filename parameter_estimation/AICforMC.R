install.packages("xtable")
install.packages("xlsx", dependencies = TRUE)
install.packages("dplyr")
install.packages("Rcpp")
install.packages("goft")
install.packages("fitdistrplus")
install.packages("ExtDist")
install.packages("rugarch")

library(ExtDist)
library(fitdistrplus)
library(goft)
library(xtable)

#Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre-9.0.1')

library(xlsx)
library(readxl)

library(gofCopula)
library(dplyr)
library(disclap)
library(rugarch)

# SP and SPF
SP = read_excel("D:/Project/project/project/project/SRM/SRM data/2020/2020/SP500_2020.xlsx", range = "C3:C7827")#C3916:C5219
SPF = read_excel("D:/Project/project/project/project/SRM/SRM data/2020/2020/SP500FUT_2020.xlsx", range = "C3:C7827")  #C3:C7827
data=read.csv("D:/Git_copula/optimal_hedging_ratio_copula/data/sp500.csv")


colnames(SP)="SP"
colnames(SPF)="SPF"


SP1   = data.frame(SP)

spec = ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(1,1)),
                  mean.model = list(armaOrder = c(1,1), include.mean = TRUE),
                  distribution.model = "std")

fit = ugarchfit(data = SP1, spec = spec)

SP.std = sqrt(uncvariance(fit))

SP1_degarch = SP1/(SP.std)

SP    = as.numeric(unlist(SP1_degarch))



SP2 = SP  
SP2[SP2<0.001] <- NA
SP2<-SP2[complete.cases(SP2)]

# # Loglikelihood and AIC for normal distribuiton

norm.log<-function(param) {

  -sum(dnorm(SP,param[1],param[2],log=T)) }

AIC.norm = -2*optim(c(0,1),ll1)$value + 2*2

# Loglikelihood and AIC for t distribuiton

t.log <-function(par){
  if(par[2]>0 & par[3]>0) return(-sum(log(dt((SP-par[1])/par[2],df=par[3])/par[2])))
  else return(Inf)
}


AIC.t= -2*optim(c(0,0.1,2.5),t.log)$value + 2*3

# Loglikelihood and AIC for laplace distribuiton
est.par <- eLaplace(SP, method="analytic.MLE"); 

laplace.AIc = -2*lLaplace(SP,param=est.par)+2*2





# lognormal distribution

ll1 = function(param){
  if(param[2]>0) return(-sum(dlnorm(SP2,param[1],param[2],log=T)))
  else return(Inf)
}

lnorm.aic = 2*optim(c(3,3),ll1)$value + 2*2

# # Poisson distribution
# fit.po = fitdist(SP,"pois")
# 
# a.po=summary(fit.po)
# a.po$aic

# # Weibull distribution
# fit.wei = fitdist(SP,"weib")
# 
# a.wei=summary(fit.wei)
# a.wei$aic


AIC.result = c("normal",a.norm$aic,"t",a.t$aic,"lognormal", lnorm.aic, "laplace", laplace.AIc)

write.xlsx(AIC.result, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\AIC_sp.xlsx")
