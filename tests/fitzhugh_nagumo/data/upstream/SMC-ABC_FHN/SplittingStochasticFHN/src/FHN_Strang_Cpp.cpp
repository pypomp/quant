#include <Rcpp.h>
using namespace Rcpp;
using namespace std;

//------------------------------------------------------------------------------
//#Matrix-Vector Multiplication
//------------------------------------------------------------------------------

//Input:
//mat    matrix
//vec    vector

//Output: matrix-vector product

// [[Rcpp::export]]
NumericVector mv_FHN_(NumericMatrix mat, NumericVector vec)
{
  NumericVector ret(mat.nrow());
  double temp=0;
  
  for(int i = 0; i < mat.nrow(); i++)
  {
    for(int j = 0; j < vec.size(); j++)
    {
      temp = temp + mat(i,j) * vec[j];
    }
    ret[i]=temp;
    temp=0;
  }
  return ret;
};

//------------------------------------------------------------------------------
// solution of linear SDE subequation of splitting method
//------------------------------------------------------------------------------

//Input:
//vec      vector (initial value)
//dm       exponential matrix of splitting procedure
//cm       Choleski factorization of covariance matrix of splitting procedure
//randvec  vector sampled from a bivariate standard normal distribution

//Output:  one step of the exact solution of the linear SDE subequation of the splitting method

// [[Rcpp::export]]
NumericVector linSDE_FHN_Cpp_(NumericVector vec, NumericMatrix dm, NumericMatrix cm, NumericVector randvec)
{
  NumericVector ret=mv_FHN_(dm,vec)+mv_FHN_(cm,randvec);
  return ret;
};

//------------------------------------------------------------------------------
// solution of nonlinear ODE subequation of splitting method
//------------------------------------------------------------------------------

//Input:
//vec        vector (initial value)
//h          step size
//eps, beta  model parameters

//Output:  one step of the exact solution of the nonlinear ODE subequation of the splitting method

// [[Rcpp::export]]
NumericVector nonlinODE_FHN_Cpp_(NumericVector vec, double h, double eps, double beta)
{
  double a=1.0/eps;
  NumericVector ret(2);
  ret(0)=vec(0)/sqrt((exp(-2.0*a*h)+vec(0)*vec(0)*(1.0-exp(-2.0*a*h))));
  ret(1)=vec(1)+beta*h;
  return ret;
};

//------------------------------------------------------------------------------
//#Strang splitting method
//------------------------------------------------------------------------------

//Input:
//grid_i         time grid 
//h_i            step size
//startv_i       starting value x0
//dm_i           exponential matrix of splitting method
//cm_i           choleski factorization of covariance matrix of splitting method
//eps_i, beta_i  model parameters

//Output: path of the stochastic FHN model

// [[Rcpp::export]]
NumericMatrix FHN_Strang_Cpp_(NumericVector grid_i, double h_i, NumericVector startv_i, NumericMatrix dm_i, NumericMatrix cm_i,  double eps_i, double beta_i)
{
  //assign input values
  NumericVector grid=grid_i;
  double h=h_i;
  NumericVector startv=startv_i;
  
  NumericMatrix dm=dm_i;
  NumericMatrix cm=cm_i;
  
  double eps=eps_i;
  double beta=beta_i;
  
  //determine length "iter" of time grid
  int iter=grid.size();
  
  //generate required bivariate standard normal vectors (stored in a 2x(iter)-matrix)
  NumericMatrix randarr(2,iter);
  randarr(0,_)=rnorm(iter);
  randarr(1,_)=rnorm(iter);
  NumericVector randvec(2);
  
  //initialize path
  NumericMatrix sol(2,iter);
  sol(_, 0)=startv;
  NumericVector newv=startv;
  
  //compute a path of the stochastic FHN model according to the Strang splitting method
  for(int i=1;i<iter;i++)
  {
    randvec=randarr(_,i);
    newv=nonlinODE_FHN_Cpp_(newv,h/2,eps,beta); //half time-step of nonlinear ODE subequation
    newv=linSDE_FHN_Cpp_(newv,dm,cm,randvec); //full time-step of linear SDE subequation
    newv=nonlinODE_FHN_Cpp_(newv,h/2,eps,beta); //half time-step of nonlinear ODE subequation
    sol(_,i)=newv;
  }
  NumericMatrix ret=sol;
  
  //return path
  return sol;
};




