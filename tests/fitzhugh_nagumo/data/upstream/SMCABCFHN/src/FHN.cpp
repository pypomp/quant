#include <Rcpp.h>
#include "FHN_S.h"
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix my_fun(int N, NumericVector mu,NumericMatrix Sigma){
  // calling rnorm()
  Function f("rmvn");
  return f(N, mu, Sigma);
};


// FHN_prior3_ computes the pdf of the exponential prior (if draw=0) or sample from it (if draw=1)
// Input:  - theta, needed only if draw=0 to compute the corresponding pdf
// Output: - out, pdf of the exponential prior (if draw=0) or sample from it (if draw=1)

// [[Rcpp::export]]
NumericVector FHN_prior3_(NumericVector theta,int draw){
  double t1,t2,t3,t4;
  if(draw == 0){
    NumericVector out=(1);
    t1 = theta[0];
    t2 = theta[1];
    t3 = theta[2];
    t4 = theta[3];
    out= R::dexp(t1,1/3,FALSE)*R::dexp(t2,2,FALSE)*R::dexp(t3,2,FALSE)*R::dexp(t4,1,FALSE);
    return(out);}
  else{
    NumericVector out(4); // Here NDIM=4
    out[0] = rexp(1,1/3)[0];
    out[1] = rexp(1,2)[0];
    out[2] = rexp(1,2)[0];
    out[3] = rexp(1,1)[0];
    return(out);
  }}


// FHN_prior2_ computes the pdf of the lognormal prior (if draw=0) or sample from it (if draw=1)
// Input:  - theta, needed only if draw=0 to compute the corresponding pdf
// Output: - out, pdf of the lognormal prior (if draw=0) or sample from it (if draw=1)

// [[Rcpp::export]]
NumericVector FHN_prior2_(NumericVector theta,int draw){
  double t1,t2,t3,t4;
  if(draw == 0){
    NumericVector out=(1);
    t1 = theta[0];
    t2 = theta[1];
    t3 = theta[2];
    t4 = theta[3];
    out= R::dlnorm(t1,0,1,FALSE)*R::dlnorm(t2,0,0.5,FALSE)*R::dlnorm(t3,0,1,FALSE)*R::dlnorm(t4,0,0.75,FALSE);
    return(out);}
  else{
    NumericVector out(4); // Here NDIM=4
    out[0] = rlnorm(1,0,1)[0];
    out[1] = rlnorm(1,0,0.5)[0];
    out[2] = rlnorm(1,0,1)[0];
    out[3] = rlnorm(1,0,0.75)[0];
    return(out);
  }}

// FHN_prior_ computes the pdf of the uniform prior (if draw=0) or sample from it (if draw=1)
// Input:  - theta, needed only if draw=0 to compute the corresponding pdf
// Output: - out, pdf of the uniform prior (if draw=0) or sample from it (if draw=1)

// [[Rcpp::export]]
NumericVector FHN_prior_(NumericVector theta,int draw){
  double t1,t2,t3,t4;
  if(draw == 0){
    NumericVector out=(1);
    t1 = theta[0];
    t2 = theta[1];
    t3 = theta[2];
    t4 = theta[3];
    out= R::dunif(t1,0.01,0.5,false)*R::dunif(t2,t1/4,6.,false)*R::dunif(t3,0.01,6.0,false)*R::dunif(t4,0.01,1.0,false);
    return(out);}
    else{
    NumericVector out(4); // Here NDIM=4
    out[0] = runif(1,0.01,0.5)[0];
    out[1] = runif(1,out[0]/4,6.0)[0];
    out[2] = runif(1,0.01,6.0)[0];
    out[3] = runif(1,0.01,1.0)[0];
    return(out);
    }}

// [[Rcpp::export]]
NumericMatrix FHN_model_(NumericVector theta, double delta, NumericVector X0, int N)
{
  double epsilon,gamma,beta,sigma1=0,sigma2;
  epsilon=theta(0);
  gamma=theta(1);
  beta=theta(2);
  sigma2=theta(3);
  NumericMatrix X1(N+1,1);
  double k=4*gamma/epsilon-1;
  double sk=sqrt(k);
  double delta_half= delta/2, cosk=cos(sk*delta_half),sink=sin(sk*delta_half);
  double cosk2=cos(sk*delta), sink2=sin(sk*delta);
  double var1=pow(sigma1,2), var2=pow(sigma2,2);
  double c3=exp(-delta_half), c4=exp(-delta)*epsilon/(2*k);
  double m11,m12,m21,m22,c11,c12,c22,eps2;
  eps2=pow(epsilon,2);
  m11=c3*(cosk+sink/sk);
  m12=-2*c3*sink/(epsilon*sk);
  m21=c3*2*gamma*sink/sk;
  m22=c3*(cosk-sink/sk);
  c11=c4/gamma*(-4*gamma/eps2*(gamma*var1+var2/epsilon)+
    k*exp(delta)*(var1*(1+gamma/epsilon)+var2/eps2) +
    ((1-3*gamma/epsilon)*var1+var2/eps2)*cosk2-
    sk*sink2*(var1*(1-gamma/epsilon)+var2/eps2));
  c12=c4*(var1*exp(delta)*k-2*(gamma*var1+var2/epsilon)/epsilon+
    (var1*(1-2*gamma/epsilon)+2*var2/eps2)*cosk2-var1*sk*sink2);
  c22=c4*((gamma*var1+var2/epsilon)*(cosk2-4*gamma/epsilon+k*exp(delta))
            -sk*sink2*(gamma*var1-var2/epsilon));
  double CONST=exp(-delta/epsilon), mCONST=1-CONST;
  double bd=beta*delta/2;
  NumericMatrix xi(N,2);
  NumericVector NullMu(2);
  NumericMatrix S(2);
  S(0,0)=c11;
  S(0,1)=c12;
  S(1,0)=c12;
  S(1,1)=c22;
  xi=my_fun(N,NullMu,S);
 double X2;
  X1(0,0)=X0[0];
  X2=X0[1];
  double x1_tilde,x2_tilde;
  for(int i=1;i<(N+1);++i){
    x1_tilde=X1(i-1,0)/sqrt(CONST+pow(X1(i-1,0),2)*mCONST);
    x2_tilde=bd+X2;
    X1(i,0)=m11*x1_tilde+m12*x2_tilde+xi(i-1,0);
    X1(i,0)=X1(i,0)/sqrt(CONST+pow(X1(i,0),2)*mCONST);// f(LT with delta/2 det. step)
    X2=bd+m21*x1_tilde+m22*x2_tilde+xi(i-1,1);// f(LT with delta/2 det.step)
    }
  return(X1);
};
