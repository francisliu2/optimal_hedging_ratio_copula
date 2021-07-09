#include "emmix.h"


void initfit_(double *y, int *pn, int *pp,int *pg, int *pncov,int *pdist,
double * pro, double *mu, double * sigma, double *dof, double *delta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *sumtau, double *sumvt,double *sumzt,double *sumlnv, 
double *ewy,double *ewz,double *ewyy, 
double *loglik, int *clust, int *pinfo, int *maxloop)
{

*pinfo =0;	  

if(*pdist == 1) 
	initmvn_(y,pn,pp,pg,pncov,pro,mu,sigma,    
		tau,sumtau,ewy,ewz,ewyy,loglik,clust,pinfo,maxloop);
else if(*pdist == 2)	
    initmvt_(y,pn,pp,pg,pncov,pro,mu,sigma,   dof,
	tau,ev,sumtau,sumvt,sumlnv,ewy,ewz,ewyy, 
	loglik,clust,pinfo,maxloop);
else if(*pdist == 3)	
    initmsn_(y, pn,pp,pg,pncov,pro,mu, sigma,     delta, 
    tau, ev, elnv, sumtau,sumvt,ewy,ewz,ewyy, 
	loglik,clust,pinfo,maxloop);
else if(*pdist == 4)	
    initmst_(y, pn,pp,pg,pncov,pro,mu, sigma, dof,delta, 
    tau, ev, elnv, ez1v, ez2v,
	sumtau,sumvt,sumzt,sumlnv,ewy,ewz,ewyy, 
	loglik,clust,pinfo,maxloop);



return;

}


void F77_SUB(initmvn)(double *y, int *pn, int *pp,int *pg,int *pncov,
double * pro, double *mu, double * sigma, 
double *tau, double *sumtau,double *ewy,double *ewz,double *ewyy, 
double *loglik, int *clust, int *perror, int *maxloop)
{
int n=*pn;
int h,i;

const double zero=0.0, one=1.0;
double sum;

for(h=0;h< *pg;h++) {

sum=zero;

for(i=0;i<n;i++) {
	tau[h*n+i]= zero;
	if(clust[i]==h+1) 
		tau[h*n+i]= one;
	sum += tau[h*n+i];
}
sumtau[h]=sum;

pro[h]=sumtau[h]/(double)n;
}


F77_SUB(mstepmvn)(y,pn,pp,pg,pncov,tau,sumtau,mu,sigma);

*perror=0;

for(i=0;i< *maxloop;i++) {
 
//  E-Step
F77_SUB(estepmvn)(y,pn,pp,pg,pro,mu,sigma,tau,sumtau,loglik,perror);

if(*perror) {return;}

//  M-step
F77_SUB(mstepmvn)(y,pn,pp,pg,pncov,tau,sumtau,mu,sigma);
       
}      


return;

}


void F77_SUB(initmvt)(double *y, int *pn, int *pp,int *pg,int *pncov,
double * pro, double *mu, double * sigma, double *dof,
double *tau, double *xuu, double *sumtau, double *sumxuu,double *sumxuuln,
double *ewy,double *ewz,double *ewyy, 
double *loglik, int *clust, int *pinfo, int *maxloop)
{
int n=*pn,g=*pg;
int h,i;

const double zero=0.0, one=1.0;
double sum;

for(h=0;h<g;h++) {

sum=zero;

for(i=0;i<n;i++) {
	tau[h*n+i]= zero;
	if(clust[i]==h+1) 
		tau[h*n+i]= one;
	sum += tau[h*n+i];
}
sumtau[h]=sum;
dof[h]= (double)4;
pro[h]=sumtau[h]/(double)n;

}

F77_SUB(mstepmvn)(y,pn,pp,pg,pncov,tau,sumtau,mu,sigma);

*pinfo=0;

for(i=0;i < *maxloop;i++) {
 
//  E-Step
F77_SUB(estepmvt)(y,pn,pp,pg,pro,mu,sigma,dof,tau,
xuu,sumtau,sumxuu,sumxuuln,loglik,pinfo);

if(*pinfo) {return;}

//  M-step
F77_SUB(mstepmvt)(y,pn,pp,pg,pncov,tau,xuu,sumtau,sumxuu,sumxuuln,mu,sigma,dof);
       
}      


return;
}






void F77_SUB(initmsn)(double *y, int *pn, int *pp,int *pg, int *pncov,
double * pro, double *mu, double * sigma, double *delta,
double *tau, double *ev, double *vv, double *sumtau, double *sumev,
double *ewy,double *ewz,double *ewyy, 
double *loglik,int *clust,int *pinfo, int *maxloop)
{
	
int n=*pn,g=*pg;
int i,h;

const double one = 1.0,zero=0.0;
double sum;

for(h=0;h<g;h++) {
 sum=zero;
 
 for(i=0;i<n;i++) {
 tau[h*n+i]=zero;
 
 if(clust[i] == (h+1)) {
    tau[h*n+i]= one;
    sum += tau[h*n+i];
 } //endif

 }//end i loop
 sumtau[h]=sum;

 pro[h]= sumtau[h]/(double)n;
 

} // end h loop



//------------------------------------------------------------------

F77_SUB(mstepmvn)(y,pn,pp,pg,pncov,tau,sumtau,mu,sigma);


F77_NAME(skew)(y,pn,pp,pg,tau,sumtau,mu,sigma,delta);

//------------------------------------------------------------------

*pinfo=0;

for(i=0;i< *maxloop;i++) {

//  E-Step
F77_NAME(estepmsn)(y,pn,pp,pg,pro,mu,sigma,delta,tau,ev,vv,sumtau,sumev,loglik,pinfo);


if(*pinfo) {return;}


//  M-step

F77_NAME(mstepmsn)(y,pn,pp,pg,pncov,tau,ev,vv,sumtau,sumev,mu,sigma,delta);



}// end of i loop

return;
}


void F77_SUB(initmst)(double *y, int *pn, int *pp,int *pg, int *pncov,
double *pro, double *mu, double * sigma, double *dof, double *delta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *sumtau, double *sumvt,double *sumzt,double *sumlnv, 
double *ewy,double *ewz,double *ewyy, 
double *loglik,int *clust,int *pinfo, int *maxloop)
{
	
int n=*pn,g=*pg;
int i,h, method[g];

const double one = 1.0,zero=0.0;
double sum;

for(h=0;h<g;h++) {
 sum=zero;
 
 for(i=0;i<n;i++) {
 tau[h*n+i]=zero;
 
 if(clust[i] == (h+1)) {
    tau[h*n+i]= one;
    sum += tau[h*n+i];
 } //endif

 }//end i loop

 sumtau[h]=sum;

 pro[h]= sumtau[h]/(double)n;
 
 method[h] = 3;

 dof[h] = (double)4;

} // end h loop



//------------------------------------------------------------------

F77_SUB(mstepmvn)(y,pn,pp,pg,pncov,tau,sumtau,mu,sigma);


F77_NAME(skew)(y,pn,pp,pg,tau,sumtau,mu,sigma,delta);

//------------------------------------------------------------------

*pinfo=0;
	  
for(i=0;i < *maxloop;i++) {


//  E-Step
F77_NAME(estepmst)(y,pn,pp,pg,pro,mu,sigma,dof,delta,
     tau,ev,elnv,ez1v,ez2v,sumtau,sumvt,sumzt,sumlnv,
     loglik,pinfo,method);

if(*pinfo) return;

//  M-step

F77_NAME(mstepmst)(y,pn,pp,pg,pncov,
	tau,ev,ez1v,ez2v,sumtau,sumvt,sumzt,
     mu,sigma,delta);


}// end of i loop

return;
}

