
library(pomp)
d <- dacca()
Jvec <- c(500,1000,2000,5000,10000)
Jtimes <- rep(NA,Jvec)
for(j in seq_along(Jvec)){
  Jtimes[j] = system.time(pfilter(d,Np=Jvec[j]))["elapsed"]
}

cbind(J=Jvec,dacca_time=Jtimes)

#          J dacca_time
# [1,]   500      0.525
# [2,]  1000      0.991
# [3,]  2000      1.957
# [4,]  5000      4.896
# [5,] 10000      9.753

  
  