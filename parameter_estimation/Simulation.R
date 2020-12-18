install.packages("gofCopula")
library(gofCopula)

# simulate normal copula 
require(mvtnorm)
S <- matrix(c(1,.8,.8,1),2,2)
AB <- rmvnorm(mean=c(0,0),sig=S,n=1000)

U= pnorm(AB)
SP_copula_sim = gof(AB, priority = "tests", copula = c("normal", "t", "clayton", "gumbel", "frank"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 1000)
write.xlsx(SP_copula_sim, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\SP_copula_sim.xlsx")
