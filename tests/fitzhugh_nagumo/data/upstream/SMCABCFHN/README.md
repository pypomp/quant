# SMCABCFHN
R package for the SMC-ABC inference for the FHN model proposed in
[1] A. Samson, M. Tamborrino, I. Tubikanec. Inference for the stochastic FitzHugh-Nagumo model from real action potential data via approximate Bayesian computation.  Computational Statistics & Data Analysis, 204, 108095, 2025. https://www.sciencedirect.com/science/article/pii/S0167947324001798

Trajectories of the FHN model are simulated using the structure-preserving splitting numerical scheme proposed in 

[2] E. Buckwar, A. Samson, M. Tamborrino, I. Tubikanec. A splitting method for SDEs with locally Lipschitz drift: Illustration on the FitzHugh-Nagumo model. Applied Numerical Mathematics 179, 191-220, 2022.

The R-package is written and maintained by Massimiliano Tamborrino (firstname dot secondname at warwick.ac.uk).

# What can you find in the package
In this package, we provide the code for the SMC-ABC algorithm using Gaussian kernel proposal, either the canonical version ('standard') or its optimased version ('olcml'). We also accommodate the code for having both canonical ('canonical') and "structure-based" ('structure-based') summary statistics.

The main routine is "SMCABCFHN_numbersim.R", which performs SMC-ABC with automatically decreasing tolerance levels, and user-specified: 1) prior ('unif','lognormal','exp'); 2) sampler ('standard' or 'olcm'); 3) summary statistics ('canonical' or 'structure-based'); 4) computational budget, i.e., total number of model simulations to run. 

# How to install the package
* Tools/Install packages/ select the source folder
*To update The simplest way is to do it via devtools, using devtools::install_github("massimilianotamborrino/SMCABCFHN")

# Output files of "SMCABCFHN_numbersim.R"
The output files (see below) will be saved in the user-specified folder, and will contain information on iteration-stage (t) and attempt. 

Output files: The input "folder" is the name of the folder where you want your results to be stored. The whole path to access to the folder can be given as input. That folder will contain many files upon completion of a run. Here is a description:

- posterior draws: a typical file name ABCdraws_stageX_attemptY.txt This is a matrix of d x N values containing N parameters draws (particles) for each of the d parameters to infer, here d=4. The X in 'stageX' is the iteration number the draws have been sampled at. The Y in 'attemptY' is the ID of the inference run (since it is possible to run several inference runs on the same dataset, one after the other). 
- ess_stageX_attemptY.txt: Value of the effective sample size (ESS) at the given iteration and attempt.
- evaltime_stageX_attemptY.txt: Number of seconds required to accept N draws at the given iteration and given attempt.
- numproposals_stageX_attemptY.txt: Number of particles proposed (i.e. both the accepted and rejected), at the given iteration and attempt.
- numproposals0_stageX_attemptY.txt: Number of particles drawn from the proposal sampler which have been immediately rejected as out of the prior support, at the given iteration and attempt.
- numproposalskappa_stageX_attemptY.txt: Number of particles drawn from the proposal sampler which have been immediately rejected as the condition kappa>0 is not met.
- numproposalsneg_stageX_attemptY.txt: Number of particles drawn from the proposal sampler which have been immediately rejected as being  negative.
- weights_stageX_attemptY.txt: Vector of N normalised importance weights associated to each of the N accepted particles at the given iteration and at the given attempt.
- ABCthreshold_stageX_attemptY.txt: Value of the ABC threshold used at the given iteration and attempt.

