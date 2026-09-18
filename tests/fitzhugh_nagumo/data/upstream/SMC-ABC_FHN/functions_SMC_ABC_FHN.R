
#-----------------------------------------------------------------------------------
# Author: Irene Tubikanec
# Date:  2024-05-24
#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
# Functions required for SBP SMC-ABC inference in the stochastic FHN model
#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
# Exponential and covariance matrices for splitting-simulation of the model
#-----------------------------------------------------------------------------------

#Input:
#eps, gamma    model parameters
#t             step size for path simulation

#Output:  exponential matrix appearing in the exact solution of the linear subequation in the splitting procedure

exp_mat_SDEharmosc<-function(t,eps,gamma){
  #set parameters
  a<-1/eps
  b<-gamma
  #create and fill matrix
  ret<-matrix(0, nrow = 2, ncol = 2)
  d<-sqrt(4*a*b-1)
  cdt<-cos(0.5*d*t)
  sdt<-sin(0.5*d*t)
  ret[1,]<-c(cdt+(1/d)*sdt,((-2*a)/d)*sdt)
  ret[2,]<-c(((2*b)/d)*sdt,cdt-(1/d)*sdt)
  #return matrix
  return (exp(-0.5*t)*ret)
}

#-----------------------------------------------------------------------------------

#Input:
#sig, eps, gamma    model parameters
#t                  step size used for path simulation

#Output:  covariance matrix appearing in the exact solution of the linear subequation in the splitting procedure

cov_mat_SDEharmosc<-function(t,sig,eps,gamma){
  #set parameters
  a<-1/eps
  b<-gamma
  #create and fill matrix
  ret<-matrix(0, nrow = 2, ncol = 2)
  d1<-sqrt(4*a*b-1)
  d2<-4*a*b-1
  et<-exp(-t)
  cdt<-cos(d1*t)
  sdt<-sin(d1*t)
  ret[1,]<-c(((a*sig^2)/(-2*b*d2))*(-d2+et*4*a*b-et*cdt+et*sdt*d1),(a*sig^2/d2)*et*(cdt-1))
  ret[2,]<-c((a*sig^2/d2)*et*(cdt-1),((sig^2)/(-2*d2))*(-d2+et*4*a*b-et*cdt-et*sdt*d1))
  #return matrix
  return (ret)
}

#-------------------------------------------------------------------------
# ABC: Pilot simulation
# Acceptance-Rejection ABC procedure to determine the threshold delta_1

#Input:
#T          time horizon for path simulation
#h          step size for path simulation
#grid       time grid for path simulation
#h_obs      step size of observed data
#grid_obs   time grid of observed data
#startv     starting value X0 for path simulation
#we         weight for distance computation
#densX      density of observed data
#stepD, startSupp, endSupp, Lsupport    summary parameters for density
#specX      spectral density of observed data
#span_val, stepP    summary parameters for spectral density
#Pr        prior distribution for model parameters

#Output: Distance value between the summaries of the observed dataset and a simulated synthetic dataset
# and corresponding parameter values
#-------------------------------------------------------------------------

ABC_Pilot<-function(in_T,in_h,in_grid,in_hobs,in_gridobs,in_startv,in_we,in_densX,in_stepD,in_startSupp,in_endSupp,in_Lsupport,in_specX,in_span_val,in_stepP,in_Pr){
  #-----------------------------------------------------------------------
  
  #Input prior information matrix
  Pr<-in_Pr
  
  #Input time grid values
  T<-in_T
  h<-in_h
  grid<-in_grid
  hobs<-in_hobs
  gridobs<-in_gridobs
  
  #Input initial value for path simulation
  startv<-in_startv
  
  #Input weight for distance calculation
  we<-in_we
  
  #Input density (summaries)
  densX<-in_densX
  stepD<-in_stepD
  startSupp<-in_startSupp
  endSupp<-in_endSupp
  Lsupport<-in_Lsupport
  
  #Input spectral density (summaries)
  specX<-in_specX
  stepP<-in_stepP
  span_val<-in_span_val
  
  #-----------------------------------------------------------------------
  #sample theta according to the prior
  
  eps<-runif(1,Pr[1,1],Pr[1,2])
  gamma<-runif(1,eps/4,Pr[2,2]) #to guarantee kappa>0
  beta<-runif(1,Pr[3,1],Pr[3,2])
  sig<-runif(1,Pr[4,1],Pr[4,2])
  
  #-----------------------------------------------------------------------
  #exponential and covariance matrix required for splitting scheme
  
  dm<-exp_mat_SDEharmosc(h,eps,gamma)
  cm<-t(chol(cov_mat_SDEharmosc(h,sig,eps,gamma)))
  
  #-----------------------------------------------------------------------
  
  #simulate synthetic data conditioned on theta
  sol<-FHN_Strang_Cpp(grid,h,startv,dm,cm,eps,beta) #package SplittingFHN
  Ysim<-sol[1,]
  
  #subsample data (to same grid as the observed data)
  indices_obs<-seq(1,length(Ysim),hobs/h)
  Y<-Ysim[indices_obs]
  
  #compute spectral density	
  specY<-spectrum(Y,log="no",span=span_val,plot=FALSE)$spec 
  
  #compute density
  densY<-density(Y,n=Lsupport,from=startSupp,to=endSupp)$y
  
  #-----------------------------------------------------------------------
  #calculate the distance to the observed reference dataset
  
  #density
  d1<-sum(abs(stepD*(densX-densY)))
  
  #spectral density
  d2<-sum(abs(stepP*(specX-specY)))
  
  #-----------------------------------------------------------------------
  #take the weighted sum of the distances as a global distance
  Dist<-d1*we+d2
  
  #-----------------------------------------------------------------------
  #return the distance and the corresponding parameter values
  ret<-array(0,dim=c(1,5,1))
  ret[,,1]<-c(Dist,eps,gamma,beta,sig)  
  
  return(ret)
}


#-------------------------------------------------------------------------
#SMC-ABC: Iteration 1 (r=1)
#Procedure happening within the first for-loop of the SMC-ABC algorithm

#Input:
#T          time horizon for path simulation
#h          step size for path simulation
#grid       time grid for path simulation
#h_obs      step size of observed data
#grid_obs   time grid of observed data
#startv     starting value X0 for path simulation
#we         weight for distance computation
#densX      density of observed data
#stepD, startSupp, endSupp, Lsupport    summary parameters for density
#specX      spectral density of observed data
#span_val, stepP    summary parameters for spectral density
#Pr         prior distribution for model parameters
#delta_1    initial threshold for distance calculation obtained from ABC pilot simulation

#Output: Distance value between the summaries of the observed dataset and a simulated synthetic dataset,
#        corresponding (kept) parameter values, required number of synthetic datasets (budget counter)
#-------------------------------------------------------------------------

ABC_r1<-function(in_T,in_h,in_grid,in_hobs,in_gridobs,in_startv,in_we,in_densX,in_stepD,in_startSupp,in_endSupp,in_Lsupport,in_specX,in_span_val,in_stepP,in_Pr,in_delta_1){
  #-----------------------------------------------------------------------
  
  #Input delta_1
  delta_1<-in_delta_1
  Dist<-delta_1+1.0 #such that Dist>=delta_1
  
  #Input prior information matrix
  Pr<-in_Pr
  
  #Input time grid values
  T<-in_T
  h<-in_h
  grid<-in_grid
  hobs<-in_hobs
  gridobs<-in_gridobs
  
  #Input initial value for path simulation
  startv<-in_startv
  
  #Input weight for distance calculation
  we<-in_we
  
  #Input density (summaries)
  densX<-in_densX
  stepD<-in_stepD
  startSupp<-in_startSupp
  endSupp<-in_endSupp
  Lsupport<-in_Lsupport
  
  #Input spectral density (summaries)
  specX<-in_specX
  stepP<-in_stepP
  span_val<-in_span_val
  
  #Value for computing the number of simulations (datasets/paths) required
  count<-0
  
  #while-loop
  while(Dist>=delta_1){
    
    #-----------------------------------------------------------------------
    #sample theta according to the prior
    
    eps<-runif(1,Pr[1,1],Pr[1,2])
    gamma<-runif(1,eps/4,Pr[2,2]) #to guarantee kappa>0
    beta<-runif(1,Pr[3,1],Pr[3,2])
    sig<-runif(1,Pr[4,1],Pr[4,2])
    
    #-----------------------------------------------------------------------
    #exponential and covariance matrix required for splitting scheme
    
    dm<-exp_mat_SDEharmosc(h,eps,gamma)
    cm<-t(chol(cov_mat_SDEharmosc(h,sig,eps,gamma)))
    
    #-----------------------------------------------------------------------
    
    #simulate synthetic data conditioned on theta
    sol<-FHN_Strang_Cpp(grid,h,startv,dm,cm,eps,beta) #package SplittingFHN
    Ysim<-sol[1,]
    
    #subsample data (to same grid as the observed data)
    indices_obs<-seq(1,length(Ysim),hobs/h)
    Y<-Ysim[indices_obs]
    
    #compute spectral density	
    specY<-spectrum(Y,log="no",span=span_val,plot=FALSE)$spec 
    
    #compute density
    densY<-density(Y,n=Lsupport,from=startSupp,to=endSupp)$y
    
    #-----------------------------------------------------------------------
    #calculate the distance to the observed reference dataset
    
    #density
    d1<-sum(abs(stepD*(densX-densY)))
    
    #spectral density
    d2<-sum(abs(stepP*(specX-specY)))
    
    #-----------------------------------------------------------------------
    #take the weighted sum of the distances as a global distance
    
    Dist<-d1*we+d2
    
    #-----------------------------------------------------------------------
    #increase counter for budget
    count<-count+1
  }
  
  #-----------------------------------------------------------------------
  ret<-array(0,dim=c(1,6,1))
  ret[,,1]<-c(Dist,eps,gamma,beta,sig,count)  
  
  return(ret)
}

#-------------------------------------------------------------------------
#SMC-ABC: Iterations r>1
# Procedure happening within the second for-loop of the SMC-ABC algorithm

#Input:
#T          time horizon for path simulation
#h          step size for path simulation
#grid       time grid for path simulation
#h_obs      step size of observed data
#grid_obs   time grid of observed data
#startv     starting value X0 for path simulation
#we         weight for distance computation
#densX      density of observed data
#stepD, startSupp, endSupp, Lsupport    summary parameters for density
#specX      spectral density of observed data
#span_val, stepP    summary parameters for spectral density
#kept_eps, kept_gamma, kept_beta, kept_sig     kept samples of previous iteration
#kept_weights                                  corresponding weights of previous iteration
#delta_r    current threshold for distance calculation obtained from previous population
#Pr_cont    prior distribution for model parameters
#sigma_kernel  covariance matrix used for perturbation (multivariate normal kernel)

#Output: Distance value between the summaries of the observed dataset and a simulated synthetic dataset,
#        corresponding (kept) parameter values, required number of synthetic datasets (budget counter)
#-------------------------------------------------------------------------

ABC_SMC<-function(in_T,in_h,in_grid,in_hobs,in_gridobs,in_startv,in_we,in_densX,in_stepD,in_startSupp,in_endSupp,in_Lsupport,in_specX,in_span_val,in_stepP,in_kept_eps,in_kept_gamma,in_kept_beta,in_kept_sig,in_kept_weights,in_delta_r,in_Pr,in_sigma_kernel){
  #-----------------------------------------------------------------------
  
  #Input: kept samples from round r-1
  kept_eps<-in_kept_eps
  kept_gamma<-in_kept_gamma
  kept_beta<-in_kept_beta
  kept_sig<-in_kept_sig
  
  #corresponding weights of round r-1
  kept_weights<-in_kept_weights
  
  #threshold delta_r
  delta_r<-in_delta_r
  Dist<-delta_r+0.1 #such that Dist>=delta_r
  
  #Input time grid values
  T<-in_T
  h<-in_h
  grid<-in_grid
  hobs<-in_hobs
  gridobs<-in_gridobs
  
  #Input initial value for path simulation
  startv<-in_startv
  
  #Input weight for distance calculation
  we<-in_we
  
  #Input density (summaries)
  densX<-in_densX
  stepD<-in_stepD
  startSupp<-in_startSupp
  endSupp<-in_endSupp
  Lsupport<-in_Lsupport
  
  #Input spectral density (summaries)
  specX<-in_specX
  stepP<-in_stepP
  span_val<-in_span_val
  
  #Input prior distribution
  Pr<-in_Pr
  
  #Input covariance matrix for perturbation (multivariate normal kernel)
  sigma_kernel<-in_sigma_kernel
  
  #Value for computing the acceptance rate
  count<-0
  
  #while-loop
  while(Dist>=delta_r){
    
    #-----------------------------------------------------------------------
    #sample theta from the weighted set and perturb it (such that it stays within the prior region)
    
    kappa<--1
    eps<--1
    gamma<--1
    beta<--1
    sig<--1
    
    while(kappa<=0||eps<=Pr[1,1]||eps>=Pr[1,2]||gamma<=Pr[2,1]||gamma>=Pr[2,2]||beta<=Pr[3,1]||beta>=Pr[3,2]||sig<=Pr[4,1]||sig>=Pr[4,2]){
      
      #sample
      index<-sample(length(kept_weights),1,prob=kept_weights)
      
      eps<-kept_eps[index] 
      gamma<-kept_gamma[index] 
      beta<-kept_beta[index] 
      sig<-kept_sig[index] 
      
      #perturb (multivariate normal kernel)
      theta_sampled<-c(eps,gamma,beta,sig)
      theta_perturbed<-rmvn(1,mu=theta_sampled,sigma=sigma_kernel)
      eps<-theta_perturbed[1]
      gamma<-theta_perturbed[2]
      beta<-theta_perturbed[3]
      sig<-theta_perturbed[4]
      
      kappa<-(4*gamma)/eps-1
    }
    
    #-----------------------------------------------------------------------
    #exponential and covariance matrix required for splitting scheme
    
    dm<-exp_mat_SDEharmosc(h,eps,gamma)
    cm<-t(chol(cov_mat_SDEharmosc(h,sig,eps,gamma)))
    
    #-----------------------------------------------------------------------
    
    #simulate synthetic data conditioned on theta
    sol<-FHN_Strang_Cpp(grid,h,startv,dm,cm,eps,beta) #package SplittingFHN
    Ysim<-sol[1,]
    
    #subsample data (to same grid as the observed data)
    indices_obs<-seq(1,length(Ysim),hobs/h)
    Y<-Ysim[indices_obs]
    
    #compute spectral density	
    specY<-spectrum(Y,log="no",span=span_val,plot=FALSE)$spec 
    
    #compute density
    densY<-density(Y,n=Lsupport,from=startSupp,to=endSupp)$y
    
    #-----------------------------------------------------------------------
    #calculate the distance to the observed reference dataset
    
    #density
    d1<-sum(abs(stepD*(densX-densY)))
    
    #spectral density
    d2<-sum(abs(stepP*(specX-specY)))
    
    #-----------------------------------------------------------------------
    #take the weighted sum of the distances as a global distance
    Dist<-d1*we+d2
    
    #-----------------------------------------------------------------------
    #increase counter
    count<-count+1
  }  
  
  #-----------------------------------------------------------------------
  ret<-array(0,dim=c(1,6,1))
  ret[,,1]<-c(Dist,eps,gamma,beta,sig,count)  
  
  return(ret)
}

