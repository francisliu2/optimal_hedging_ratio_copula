
library(gofCopula)

setwd("//unicorn7/TeamData/VT/Meng-Jou/gof")
# SP and SPF
SP = read_excel("E:/gof/SRM data/2020/2020/SP500_2020.xlsx", range = "C3:C7827")#C3916:C5219
SPF = read_excel("E:/gof/SRM data/2020/2020/SP500FUT_2020.xlsx", range = "C3:C7827")  #C3:C7827

colnames(SP)="SP"
colnames(SPF)="SPF"

SP    = as.numeric(unlist(SP))
SP    = data.frame(SP)
SPF   = as.numeric(unlist(SPF))
SPF    = data.frame(SPF)

data = cbind(SP, SPF)
data = as.matrix(data)

SP_copula = gof(data, priority = "tests", copula = c("normal", "t", "clayton", "gumbel", "frank"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 1000)

#gof(data, priority = "tests", copula = c("normal", "t"), tests = c("gofRosenblattSnB", "gofRosenblattSnC"), M = 10)

SP_copula_un = unlist(SP_copula)
SP_copula_normal_un=unlist(SP_copula$normal)

#normal copula
name = rbind("method", "copula", "margins", "theta")#, "", "RosenblattSnB", "RosenblattSnC", "Hybrid(1,2)")
output = rbind (unlist(SP_copula$normal$method), unlist(SP_copula$normal$copula), unlist(SP_copula$normal$margins), round(unlist(SP_copula$normal$theta),3))
output2 = cbind(name, output)
output3 = round(SP_copula$normal$res.tests,3)
write.xlsx(output2, "SP_copula_m1000_normal.xlsx", sheetName = "sheet1", row.names = FALSE)
write.xlsx(output3, "SP_copula_m1000_normal.xlsx", sheetName = "sheet2", append = TRUE, row.names = TRUE)

#t copula

name = rbind("method", "copula", "margins", "theta")#, "", "RosenblattSnB", "RosenblattSnC", "Hybrid(1,2)")
output_t = rbind (unlist(SP_copula$t$method), unlist(SP_copula$t$copula), unlist(SP_copula$t$margins), round(unlist(SP_copula$t$theta),3))
output2_t = cbind(name, output_t)
output3_t = round(SP_copula$t$res.tests,3)
write.xlsx(output2_t, "SP_copula_m1000_t.xlsx", sheetName = "sheet1", row.names = FALSE)
write.xlsx(output3_t, "SP_copula_m1000_t.xlsx", sheetName = "sheet2", append = TRUE, row.names = TRUE)

#clayton copula

output_clayton = rbind (unlist(SP_copula$clayton$method), unlist(SP_copula$clayton$copula), unlist(SP_copula$clayton$margins), round(unlist(SP_copula$clayton$theta),3))
output2_clayton = cbind(name, output_clayton)
output3_clayton = round(SP_copula$clayton$res.tests,3)
write.xlsx(output2_clayton, "SP_copula_m1000_clayton.xlsx", sheetName = "sheet1", row.names = FALSE)
write.xlsx(output3_clayton, "SP_copula_m1000_clayton.xlsx", sheetName = "sheet2", append = TRUE, row.names = TRUE)

#Frank

output_frank = rbind (unlist(SP_copula$frank$method), unlist(SP_copula$frank$copula), unlist(SP_copula$frank$margins), round(unlist(SP_copula$frank$theta),3))
output2_frank = cbind(name, output_frank)
output3_frank = round(SP_copula$frank$res.tests,3)
write.xlsx(output2_frank, "SP_copula_m1000_frank.xlsx", sheetName = "sheet1", row.names = FALSE)
write.xlsx(output3_frank, "SP_copula_m1000_frank.xlsx", sheetName = "sheet2", append = TRUE, row.names = TRUE)

#Gumbel

output_gumbel = rbind (unlist(SP_copula$gumbel$method), unlist(SP_copula$gumbel$copula), unlist(SP_copula$gumbel$margins), round(unlist(SP_copula$gumbel$theta),3))
output2_gumbel = cbind(name, output_gumbel)
output3_gumbel = round(SP_copula$gumbel$res.tests,3)
write.xlsx(output2_gumbel, "SP_copula_m1000_gumbel.xlsx", sheetName = "sheet1", row.names = FALSE)
write.xlsx(output3_gumbel, "SP_copula_m1000_gumbel.xlsx", sheetName = "sheet2", append = TRUE, row.names = TRUE)


