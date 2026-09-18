
#-----------------------------------------------------------------------------------
# Author: Irene Tubikanec
# Date:   2024-05-24
#
# Description: SBP SMC-ABC method for parameter estimation
#              in the stochastic FHN model, proposed in the paper:            
#
#              Inference for the stochastic FitzHugh-Nagumo model 
#              from real action potential data via approximate Bayesian computation, 
#              by A. Samson, M. Tamborrino and I. Tubikanec
#-------------------------------------------------------------------------------

#--------------------------------------------
#load files
#--------------------------------------------

source(file="required_packages.R")
source(file="functions_SMC_ABC_FHN.R")

#--------------------------------------------
#set file name to store the results
#--------------------------------------------

filename_results<-"ABC_Results"
  
#-------------------------------------------------------------------------------
# CHOOSE SETTING FOR SMC-ABC
#-------------------------------------------------------------------------------
  
N<-1000 #number of kept samples in each iteration 
p<-0.5 #percentile for threshold computation

N_pilot<-10^4 #number of distances for the pilot run

#Time horizon of observed and simulated data
T<-200

#Time step of observed data
hobs<-0.02

#Budget (maximum number of simulations/paths allowed)
budget<-10^6

#--------------------------------------------------------------------------
# PREPARE SMC-ABC
#-------------------------------------------------------------------------------

#Time step of simulated synthetic data
hsim<-0.02

#prior distribution
Pr<-matrix(0,nrow=4,ncol=2) 
Pr[1,]<-c(0.01,0.5) #eps
Pr[2,]<-c(0.01,6) #gamma
Pr[3,]<-c(0.01,6) #beta
Pr[4,]<-c(0.01,1) #sig

#starting value for path simulation
startv<-c(0,0)

#parameter values to generate the (simulated) observed dataset
eps<-0.1
gamma<-1.5
beta<-0.8
sig<-0.3

#time grid for observed data
gridobs<-seq(from=0,to=T,by=hobs)
#time grid for simulated synthetic data
gridsim<-seq(from=0,to=T,by=hsim)

#--------------------------------------------------------------------------
# prepare parallel computation
#--------------------------------------------------------------------------

ncl<-detectCores()
cl<-makeCluster(ncl)
registerDoSNOW(cl)

#--------------------------------------------------------------------------
# prepare observed data
#--------------------------------------------------------------------------

#read observed reference data
if(hobs==0.02){
  X<-as.vector(t(read.table("Reference_Data/data_Delta0.02.txt",sep="",header=F)))
}
if(hobs==0.04){
  X<-as.vector(t(read.table("Reference_Data/data_Delta0.04.txt",sep="",header=F)))
}
if(hobs==0.08){
  X<-as.vector(t(read.table("Reference_Data/data_Delta0.08.txt",sep="",header=F)))
}

#cut off reference data (to T)
X<-X[1:((T/hobs)+1)]

#--------------------------------------------------------------------------
# spectrum of the observed data 
#--------------------------------------------------------------------------

#set smoothing parameter for spectrum calculation
span_val<-0.3*T 

#compute spectral density
specX<-spectrum(X,log="no",span=span_val,plot=FALSE)
spx<-specX$freq 
stepP<-diff(spx)[1]

#compute weight for distance calculation
we<-sum(stepP*specX$spec)

#--------------------------------------------------------------------------
# density of the observed data 
#--------------------------------------------------------------------------

#set parameters for density computation
startSupp<--5
endSupp<-5
Lsupport<-1000

#compute density
densX<-density(X,n=Lsupport,from=startSupp,to=endSupp)
stepD<-diff(densX$x)[1]


#-------------------------------------------------------------------------------
# Start Pilot run
#-------------------------------------------------------------------------------

#call ABC pilot function
merge_d<-foreach(i=1:N_pilot,.combine='rbind',.packages = c('SplittingStochasticFHN')) %dopar% {
  ABC_Pilot(T,hsim,gridsim,hobs,gridobs,startv,we,densX$y,stepD,startSupp,endSupp,Lsupport,specX$spec,span_val,stepP,Pr)
}

merge_d<-merge_d[,1:5]

#sort them with respect to the distances
sort_d<-merge_d[order(merge_d[,1]),]

#keep all the sorted values
Dvec=sort_d[,1]

#determine threshold delta_1
delta_r<-Dvec[[N*p]]


#--------------------------------------------------------------------------
# Start iterations for SMC ABC
#--------------------------------------------------------------------------

count_total_sims<-0 #to count the total number of simulations (datasets/paths) required


#--------------------------------------------------------------------------
# Iteration r=1
#--------------------------------------------------------------------------

r<-1

#Call ABC function for iteration r=1
merge_d<-foreach(i=1:N,.combine='rbind',.packages = c('SplittingStochasticFHN')) %dopar% {
  ABC_r1(T,hsim,gridsim,hobs,gridobs,startv,we,densX$y,stepD,startSupp,endSupp,Lsupport,specX$spec,span_val,stepP,Pr,delta_r)
}

merge_d<-merge_d[,1:6]

#sort them with respect to the distances
sort_d<-merge_d[order(merge_d[,1]),]

#keep all the sorted values
Dvec=sort_d[,1]
epsvec=sort_d[,2]
gammavec=sort_d[,3]
betavec=sort_d[,4]
sigvec=sort_d[,5]
countvec=sort_d[,6]

#increase budget counter
count_total_sims<-count_total_sims+sum(countvec) 

#initialize the weights
weights<-rep(1,N)

#normalise the initialized weights
norm_weights<-weights/sum(weights)

#estimate 2*hat_Sigma_r from theta_kept_r-1 (twice the empirical weighted covariance matrix of previous popluation)
theta_kept_r_1<-matrix(0,nrow=N,ncol=4)
theta_kept_r_1[,1]<-epsvec
theta_kept_r_1[,2]<-gammavec
theta_kept_r_1[,3]<-betavec
theta_kept_r_1[,4]<-sigvec
sigma_kernel<-2*cov.wt(theta_kept_r_1,wt=norm_weights)$cov


#--------------------------------------------------------------------------
# Iteratons r>1
#--------------------------------------------------------------------------

while(count_total_sims<budget){ 
  
  #increase number of current iteration r
  r<-r+1

  #determine threshold delta
  delta_r<-Dvec[[N*p]] 
  
  #call ABC-SMC function for iterations r>1
  merge_d<-foreach(i=1:N,.combine='rbind',.packages = c('SplittingStochasticFHN','mvnfast')) %dopar% {
    ABC_SMC(T,hsim,gridsim,hobs,gridobs,startv,we,densX$y,stepD,startSupp,endSupp,Lsupport,specX$spec,span_val,stepP,epsvec,gammavec,betavec,sigvec,norm_weights,delta_r,Pr,sigma_kernel)
  }
  
  merge_d<-merge_d[,1:6]
  
  #sort them with respect to the distances
  sort_d<-merge_d[order(merge_d[,1]),]
  
  #keep all the sorted values
  Dvec<-sort_d[,1]
  epsvec_new<-sort_d[,2]
  gammavec_new<-sort_d[,3]
  betavec_new<-sort_d[,4]
  sigvec_new<-sort_d[,5]
  countvec<-sort_d[,6]
  
  #increase budget counter
  count_total_sims<-count_total_sims+sum(countvec) 
  
  #update the weights for the kept samples of that round
  for(j in 1:N){
    sum<-0
    
    #multivariate perturbation kernel
    x_kernel<-c(epsvec_new[j],gammavec_new[j],betavec_new[j],sigvec_new[j])
    
    for(l in 1:N){
      #multivariate perturbation kernel
      mu_kernel<-c(epsvec[l],gammavec[l],betavec[l],sigvec[l])
      sum<-sum+norm_weights[l]*dmvn(x_kernel,mu_kernel,sigma_kernel)
    }
    weights[j]<-(dunif(epsvec_new[j],min=Pr[1,1],max=Pr[1,2])*dunif(gammavec_new[j],min=Pr[2,1],max=Pr[2,2])*dunif(betavec_new[j],min=Pr[3,1],max=Pr[3,2])*dunif(sigvec_new[j],min=Pr[4,1],max=Pr[4,2]))/sum
  }
  
  #normalize the weights
  norm_weights<-weights/sum(weights)
  
  #update parameter vectors
  epsvec<-epsvec_new
  gammavec<-gammavec_new
  betavec<-betavec_new
  sigvec<-sigvec_new
  
  #estimate 2*hat_Sigma_r from theta_kept_r-1 (twice the empirical weighted covariance matrix of previous popluation)
  theta_kept_r_1[,1]<-epsvec
  theta_kept_r_1[,2]<-gammavec
  theta_kept_r_1[,3]<-betavec
  theta_kept_r_1[,4]<-sigvec
  sigma_kernel<-2*cov.wt(theta_kept_r_1,wt=norm_weights)$cov
  
  #print current iteration
  print(r)
}

#--------------------------------------------------------------------------
# End iterations for SMC_ABC
#--------------------------------------------------------------------------

stopCluster(cl)

#--------------------------------------------------------------------------
# Store results
#--------------------------------------------------------------------------

#kept samples
data_file <- paste("ABC_Results/epskept.txt",sep = "")
write(t(epsvec_new),file = data_file,ncolumns = N,sep = " ")
data_file <- paste("ABC_Results/gammakept.txt",sep = "")
write(t(gammavec_new),file = data_file,ncolumns = N,sep = " ")
data_file <- paste("ABC_Results/betakept.txt",sep = "")
write(t(betavec_new),file = data_file,ncolumns = N,sep = " ")
data_file <- paste("ABC_Results/sigkept.txt",sep = "")
write(t(sigvec_new),file = data_file,ncolumns = N,sep = " ")

#corresponding weights
data_file <- paste("ABC_Results/weightskept.txt",sep = "")
write(t(norm_weights),file = data_file,ncolumns = N,sep = " ")

