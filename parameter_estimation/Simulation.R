install.packages("gofCopula")
library(gofCopula)

# simulate normal copula 
require(mvtnorm)
S <- matrix(c(1,.8,.8,1),2,2)
AB <- rmvnorm(mean=c(0,0),sig=S,n=1000)

U= pnorm(AB)
SP_copula_norm = gof(AB, priority = "tests", copula = c("normal"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 100)

SP_copula_norm_1 = unlist(SP_copula_norm)
write.xlsx(SP_copula_norm_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\SP_copula_norm.xlsx")

#gofGetHybrid(result = SP_copula_norm, p_values = c("MyTest" = 0.3, "AnotherTest" = 0.7), nsets = 2)
SP_copula_t = gof(AB, priority = "tests", copula = c("t"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 100)
SP_copula_t_1 = unlist(SP_copula_t)
write.xlsx(SP_copula_t_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\SP_copula_t.xlsx")

SP_copula_gumbel = gof(AB, priority = "tests", copula = c("gumbel"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 100)
SP_copula_gumbel_1 = unlist(SP_copula_gumbel)
write.xlsx(SP_copula_gumbel_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\SP_copula_gumbel.xlsx")

SP_copula_frank = gof(AB, priority = "tests", copula = c("frank"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 100)
SP_copula_frank_1 = unlist(SP_copula_frank)
write.xlsx(SP_copula_frank_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\SP_copula_frank.xlsx")

SP_copula_clayton = gof(AB, priority = "tests", copula = c("clayton"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 100)
SP_copula_clayton_1 = unlist(SP_copula_clayton)
write.xlsx(SP_copula_clayton_1, "D:\\Git_copula\\optimal_hedging_ratio_copula\\parameter_estimation\\SP_copula_clayton.xlsx")
