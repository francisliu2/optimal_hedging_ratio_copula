
#include "emmix.h"


void F77_SUB(estepmvn)(double *y, int *pn, int *pp, int *pg,
double * pro, double * mu, double * sigma, 
double * tau, double * sumtau,double * loglik,int * pinfo)
{

int n = * pn, g = * pg;
	  
int i,k,info =0;

const double zero = 0.0;
double sum;

//-------------------------------------------------------


F77_SUB(denmvn)(y,pn,pp,pg,mu,sigma,tau,&info);

if(info) {*pinfo=2;return;}


//-------------------------------------------------------

F77_SUB(gettau)(tau,pro,loglik,pn,pg,&info);

if(info) {*pinfo=3;return;}

for(k=0;k<g;k++) {
	sum=zero;

    for(i=0;i<n;i++)
      sum += tau[k*n+i];

	sumtau[k]=sum;
	
	pro[k]=sumtau[k]/(double)n;
    
    if(sumtau[k]<2.0) 
	pro[k]=zero;
}

//-------------------------------------------------------

*pinfo=info;

return;
}

void F77_SUB(estepmvt)(double *y, int *pn, int *pp, int *pg,
double * pro, double * mu, double * sigma, double *dof,
double * tau, double * xuu,double * sumtau,double * sumxuu,
double * sumxuuln, double * loglik,int * pinfo)
{

int n = * pn, p = *pp, g = * pg;
	  
// some constant variables:


const double two =  2.0,zero = 0.0;


// local variables  
 
int i,k,info=0;
double temp,sum1,sum2,sum3;


//-------------------------------------------------------


F77_SUB(denmvt2)(y,pn,pp,pg,mu,sigma,dof,tau,xuu,&info);

if(info) {*pinfo=2;return;}


//-------------------------------------------------------

F77_SUB(gettau)(tau,pro,loglik,pn,pg,&info);

if(info) {*pinfo=3;return;}

for(k=0;k<g;k++) {
	sum1=zero;
	sum2=zero;
	sum3=zero;

    for(i=0;i<n;i++) {
		sum1   += tau[k*n+i];
		sum2   += tau[k*n+i]*xuu[k*n+i];
		sum3   += tau[k*n+i]*(log(xuu[k*n+i])-xuu[k*n+i]);
	}

	sumtau[k]=sum1;
	sumxuu[k]=sum2;
	sumxuuln[k]=sum3;

	temp = ( (double)p +dof[k] )/two;
    sumxuuln[k] += - ( log(temp)-digamma(temp) )*sumtau[k]; 

	pro[k]=sumtau[k]/(double)n;
    
    if(sumtau[k]<two) 
	pro[k]=zero;
}

//-------------------------------------------------------

	
*pinfo=info;

return;
}





