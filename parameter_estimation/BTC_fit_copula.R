setwd("D:/gof")
library(gofCopula)
library(VineCopula)
# BRR and BTC_Fut
BRR  = read_excel("D:/Project/project/project/project/SRM/SRM data/BRR.xlsx", range = "A3:C687")
BTCF = read_excel("D:/Project/project/project/project/SRM/SRM data/BTCF.xlsx", range = "A3:C687")  

colnames(BRR)=c("Date","BRR","BRR_return")
colnames(BTCF)=c("Date","BTCF","BTCF_return")




Joint_table = inner_join(BRR, BTCF, by="Date" )

BRR  = Joint_table$BRR_return
BTCF = Joint_table$BTCF_return


hist(BRR, breaks = 100, col="green", density = 20)
BRR_mean = mean(BRR)
BRR_std  = sqrt(var(BRR))
BRR1    = data.frame(BRR)
hist(rnorm(nrow(BRR1), mean =BRR_mean, sd = BRR_std), breaks = 100, col="blue", add =T, density = 20, angle=-45)

hist(BTCF, breaks = 100, col="green", density = 20) 
BTCF_mean = mean(BTCF)
BTCF_std  = sqrt(var(BTCF))
BTCF1    = data.frame(BTCF)
hist(rnorm(nrow(BTCF1), mean =BTCF_mean, sd = BTCF_std), breaks = 100, col="blue", add =T, density = 20, angle=-45)


corr_kend = cor(cbind(BRR, BTCF), method = "kendall")

var_BRR  = pobs(BRR)
var_BTCF = pobs(BTCF)

selectedCopula = BiCopSelect(var_BRR,var_BTCF, familyset = NA)

selectedCopula$familyname
selectedCopula$par
selectedCopula$family

#Estimate copula parameter

cop_model = tCopula(dim = 2)
m = pobs(as.matrix(cbind(BRR,BTCF)))
fit <- fitCopula(cop_model, m, method = 'ml')
coeff=coef(fit)
tau(tCopula(param = 0.6326))

#write to excel file

name  = rbind("Copula", "parameter1", "parameter2")
output = rbind(selectedCopula$familyname,selectedCopula$par,selectedCopula$par2)
output2 = cbind(name, output)

write.xlsx(output2, "BTC_copula.xlsx")
