Author: Irene Tubikanec
Date:   2024-05-24 (last update on 2024-08-29)

Description: This code provides an implementation of the structure-based and preserving (SBP) SMC-ABC method for parameter estimation in the stochastic FitzHugh-Nagumo (FHN) model, proposed in the paper:
         
Inference for the stochastic FitzHugh-Nagumo model from real action potential data via approximate Bayesian computation,
by Adeline Samson, Massimiliano Tamborrino and Irene Tubikanec.

In particular, it uses a structure-preserving splitting method* for path generation (see the package "SplittingStochasticFHN"), structure-based data summaries, a (standard) Gaussian proposal sampler, and uniform prior distributions. Moreover, the code reproduces the estimation results of the setting T=200, Delta_obs=0.02 when simulated data is observed (and can be easily adapted to other settings and real data experiments).

---------------------------------------------------------------------------------------
How the code works:
---------------------------------------------------------------------------------------

1. Install relevant packages:
   
   install.packages("Rcpp")
   install.packages("devtools")
   install.packages("roxygen2")
   install.packages("RcppNumerical")
   install.packages("foreach")
   install.packages("doSNOW")
   install.packages("doParallel")
   install.packages("mvnfast")

2. Install the package SplittingStochasticFHN (Strang-splitting-simulation of a path of the stochastic FHN model, C++ Code):

   install.packages("SplittingStochasticFHN_1.0.tar.gz")

3. Run the R-file main_SMC_ABC_FHN (SBP SMC-ABC inference for the stochastic FHN model)
After termination of the algorithm, the kept ABC posterior samples and corresponding weights are stored into the folder ABC_Results.

4. Run the R-file plot_results (visualization of estimation results)
It will create a figure visualizing the marginal ABC posterior densities. The figure also reports the underlying true parameters values. 

---------------------------------------------------------------------------------------

Code description:
For a detailed description of the code, please consider the respective files

Licence information:
Please consider the txt-files LICENCE and COPYING.GPL.v3.0

*the used splitting method has been introduced in the paper "A splitting method for SDEs with locally Lipschitz drift: Illustration on the FitzHugh-Nagumo model", by Evelyn Buckwar, Adeline Samson, Massimiliano Tamborrino and Irene Tubikanec.


