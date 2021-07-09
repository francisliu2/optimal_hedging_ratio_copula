#include "emmix.h"



void F77_SUB(mstepmvn)(double *y,int *pn, int *pp, int *pg,int *pncov,
double *tau, double *sumtau, double *mu, double *sigma)
{

int p= *pp,n= *pn,g= *pg, ncov= *pncov;

int h,i,j,k;

const double two=2.0,zero= 0.0;

double sum;



for(h=0;h<g;h++) {
   
//pro[h]=sumtau[h]/(double)n;

//   calculate the means 
   
   for(j=0;j<p;j++) {

	  sum=zero;
      
	  for(i=0;i<n;i++)
	        sum += y[j*n+i]*tau[h*n+i];
      
	  if(sumtau[h] < two) 
            mu[h*p+j] = zero;
	  else
     	    mu[h*p+j] = sum/sumtau[h];}

//  calculate the covariances 
       


   for(j=0;j<p;j++)     
	  for(i=0;i<=j;i++)
         sigma[h*p*p+j*p+i] = zero;



   for(i=0;i<n;i++)
	  for(k=0;k<p ;k++)
	     for(j=0;j<=k;j++)
sigma[h*p*p+k*p+j]  += (y[j*n+i]-mu[h*p+j])*(y[k*n+i]-mu[h*p+k])*tau[h*n+i];


   for(j=0;j<p;j++)   { 
      for(i=0;i<=j;i++) { 
	  if(sumtau[h] < two)
        sigma[h*p*p+j*p+i] =zero;
	  else
	    sigma[h*p*p+j*p+i]=sigma[h*p*p+j*p+i]/sumtau[h];
	  
	  sigma[h*p*p+i*p+j]=sigma[h*p*p+j*p+i];}
}

}  

// end of loop (h)

F77_SUB(getcov)(sigma,sumtau,&n,&p,&g,&ncov);


return;
}


void F77_SUB(mstepmvt)(double *y,int *pn, int *pp, int *pg,int *pncov,
double *tau,double *xuu, double *sumtau, double *sumxuu,double *sumxuuln,
double *mu, double *sigma, double *dof)
{

int p= *pp,n= *pn,g= *pg, ncov= *pncov;
int h,i,j,k;
double sum,bx=200;

const double two=2.0,zero= 0.0;


for(h=0;h<g;h++) {
   

//   calculate the means 

   for(j=0;j<p;j++) {

	  sum=zero;
      
	  for( i=0;i<n;i++)
	        sum += y[j*n+i]*tau[h*n+i]*xuu[h*n+i];
      
	  if(sumtau[h] < two) 
            mu[h*p+j] = zero;
	  else
     	    mu[h*p+j] = sum/sumxuu[h];}

//  calculate the covariances 
       

   for(j=0;j<p;j++)     
	  for( i=0;i<=j;i++)
         sigma[h*p*p+j*p+i] = zero;



   for(i=0;i<n;i++)
	  for( k=0;k<p ;k++)
	     for( j=0;j<=k;j++)
sigma[h*p*p+k*p+j] += (y[j*n+i]-mu[h*p+j])*(y[k*n+i]-mu[h*p+k])*tau[h*n+i]*xuu[h*n+i];


   for( j=0;j<p;j++)   { 
      for(i=0;i<=j;i++) { 
	  
	  if(sumtau[h] < two)
        sigma[h*p*p+j*p+i] =zero;
	  else
	    sigma[h*p*p+j*p+i]=sigma[h*p*p+j*p+i]/sumtau[h];
	  
	  sigma[h*p*p+i*p+j]=sigma[h*p*p+j*p+i];}}

}


// calculate the degrees of freedom

F77_SUB(getdof)(&n, &g, sumtau, sumxuuln, dof,&bx);

if(ncov!=3)
F77_SUB(getcov)(sigma,sumtau,&n,&p,&g,&ncov);


return;
}


