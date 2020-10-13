install.packages("VineCopula")
library(VineCopula)
# SP and SPF
SP = read_excel("D:/Project/project/project/project/SRM/SRM data/2020/2020/SP500_2020.xlsx", range = "C3:C7827")#C3916:C5219
SPF = read_excel("D:/Project/project/project/project/SRM/SRM data/2020/2020/SP500FUT_2020.xlsx", range = "C3:C7827")  #C3:C7827

colnames(SP)="SP"
colnames(SPF)="SPF"

SP    = as.numeric(unlist(SP))

SPF   = as.numeric(unlist(SPF))



hist(SP, breaks = 100, col="green", density = 20)
SP_mean = mean(SP)
SP_std  = sqrt(var(SP))
SP1    = data.frame(SP)
hist(rnorm(nrow(SP1), mean =SP_mean, sd = SP_std), breaks = 100, col="blue", add =T, density = 20, angle=-45)

hist(SPF, breaks = 100, col="green", density = 20) 
SPF_mean = mean(SPF)
SPF_std  = sqrt(var(SPF))
SPF1    = data.frame(SPF)
hist(rnorm(nrow(SPF1), mean =SPF_mean, sd = SPF_std), breaks = 100, col="blue", add =T, density = 20, angle=-45)

corr_kend = cor(cbind(SP, SPF), method = "kendall")

var_SP  = pobs(SP)
var_SPF = pobs(SPF)

selectedCopula = BiCopSelect(var_SP,var_SPF, familyset = NA)

selectedCopula
selectedCopula$par
selectedCopula$family

#Estimate copula parameter

cop_model = tCopula(dim = 2)
m = pobs(as.matrix(cbind(SP,SPF)))
fit <- fitCopula(cop_model, m, method = 'ml')
coef(fit)
tau(tCopula(param = 0.9732))

