#include "emmix.h"



void F77_SUB(scaestepmvn)(double *y,int *pn, int *pp, int *pg,
double *tau, double *mu, double *ety,double *etyy)
{

int p= *pp,n= *pn,g= *pg;

int h,i,j,k;

const double zero= 0.0;

double sum;



for(h=0;h<g;h++) {
   

//   calculate the means 
   
   for(j=0;j<p;j++) {

	  sum=zero;
      
	  for(i=0;i<n;i++)
	        sum += y[j*n+i]*tau[h*n+i];
      
      ety[h*p+j] = sum;
  }

//  calculate the covariances 
       
	  for(k=0;k<p ;k++)
	     for(j=0;j<=k;j++) {
            
			sum=zero;

            for(i=0;i<n;i++)
sum+=(y[j*n+i]-mu[h*p+j])*(y[k*n+i]-mu[h*p+k])*tau[h*n+i];

etyy[h*p*p+k*p+j]=sum;
etyy[h*p*p+j*p+k]=sum;

}
}
// end of loop (h)

return;
}


void F77_SUB(scaestepmvt)(double *y,int *pn, int *pp, int *pg,
double *tau,double *xuu,double *mu, double *ewy, double *ewyy)
{

int p= *pp,n= *pn,g= *pg;
int h,i,j,k;
double sum;

const double zero= 0.0;


for(h=0;h<g;h++) {
   

//   calculate the means 

   for(j=0;j<p;j++) {

	  sum=zero;
      
	  for( i=0;i<n;i++)
	        sum += y[j*n+i]*tau[h*n+i]*xuu[h*n+i];
      
      ewy[h*p+j] = sum;}

//  calculate the covariances 
       

	  for( k=0;k<p ;k++)
	     for( j=0;j<=k;j++) {
sum=zero;

for(i=0;i<n;i++)
sum+=(y[j*n+i]-mu[h*p+j])*(y[k*n+i]-mu[h*p+k])*tau[h*n+i]*xuu[h*n+i];

ewyy[h*p*p+k*p+j]=sum;

ewyy[h*p*p+j*p+k]=sum;

}
}


return;
}


