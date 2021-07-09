#include "emmix.h"



//   start from initial partition

void emskewfit1(double *y, int *pn, int *pp,int *pg,  int *pncov,int *pdist,
double * pro, double *mu, double * sigma, double *dof, double *delta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *sumtau, double *sumvt,double *sumzt,double *sumlnv, 
double *ewy,double *ewz,double *ewyy, 
double *loglik, double *lk, double *aic, double *bic,
int *clust, int *pinfo,  int *itmax, double *epsilon, int *maxloop)
{
int error = 0;

initfit_(y, pn, pp,pg, pncov,pdist, 
pro, mu, sigma, dof, delta,
tau, ev, elnv, ez1v,ez2v,sumtau,sumvt,sumzt,sumlnv,
ewy,ewz,ewyy,
loglik, clust, &error, maxloop);

if(error) {return;}


emskewfit2(y, pn, pp,pg, pncov,pdist, pro, mu, sigma, dof,delta,
tau,ev,elnv,ez1v,ez2v,sumtau,sumvt,sumzt,sumlnv,
loglik, lk, aic,bic,
clust,&error,itmax, epsilon);

*pinfo=error;

return;
}


//   start from initial values

void emskewfit2(double *y, int *pn, int *pp,int *pg,  int *pncov,int *pdist,
double * pro, double *mu, double * sigma, double *dof, double *delta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *sumtau, double *sumvt,double *sumzt,double *sumlnv, 
double *loglik, double *lk, double *aic, double *bic,
int *clust, int *pinfo,  int *itmax, double *epsilon)
{

int vk=0,n=*pn,p=*pp,g=*pg;

double tmp=0,two=2;

// call main functions

if((*pdist) == 1)
emmvn_(y, pn, pp,pg,pncov,pro,mu, sigma,
	tau,sumtau,						
	loglik,lk,pinfo,itmax,epsilon);
else if((*pdist) == 2)
emmvt_(y, pn, pp,pg,pncov,pro,mu, sigma, dof,
	tau, ev,sumtau,sumvt,sumlnv,					
	loglik,lk,pinfo,itmax,epsilon);
else if((*pdist) == 3)
emmsn_(y, pn,pp,pg,pncov,pro,mu, sigma, delta, 
	tau, ev, elnv,sumtau,sumvt,			
	loglik,lk,pinfo,itmax,epsilon);
else if((*pdist) == 4)
emmst_(y, pn,pp,pg,pncov,pro,mu, sigma, dof,delta, 
	tau, ev, elnv, ez1v, ez2v,sumtau,sumvt,sumzt,sumlnv,
	loglik,lk,pinfo,itmax, epsilon);


if(*pinfo <= 1)
{

if (*pncov ==1)
   vk =(g-1) + p*g + p*(p+1)/2;
else if (*pncov ==3 ) 
   vk =(g-1) + p*g + g*p*(p+1)/2;
else if (*pncov == 2) 
   vk =(g-1) + p*g + p;
else if (*pncov == 4) 
   vk =(g-1) + p*g + g*p;
else if (*pncov == 5) 
   vk=g-1+p*g+g;

if(*pdist == 2) vk=vk+g;
else if(*pdist == 3) vk = vk + p*g ;
else if(*pdist == 4) vk = vk + g  + p*g;


	   
tmp = (double)vk;

//      Calculate the value of the Akaike Information
            *aic = -two*(*loglik) + two * tmp;
//      Calculate the value of the Bayesian Information
            *bic = -two*(*loglik) + log((double)n)* tmp;

F77_SUB(tau2clust)(tau, pn, pg, clust);

}

return;
}


/*

mixture of multivariate Normal distributions

*/

void emmvn_(double *y, int *pn, int *pp,int *pg,int *pncov,
double * pro, double *mu, double * sigma, double *tau, double *sumtau,
double *loglik, double *lk, int *pinfo,int *itmax, double *epsilon)
{
int it;

double zero=0.0;

//double sigm[p*p],inv[p*p],lgdet,tmp,tmp2=1;

//int  p=*pp,g=*pg,i,j,k,count,info =0,save[p];


for(it=0;it< *itmax;it++) lk[it]=zero;

//   start of EM algorithm loop

*pinfo =0;



for(it=0;it< *itmax;it++) {//start of EM loops

 
//  E-Step
F77_SUB(estepmvn)(y,pn,pp,pg,pro,mu,sigma,tau,sumtau,loglik,pinfo);


if(*pinfo) {*pinfo = *pinfo+10; return;}

lk[it]= *loglik;

/*
------------------------------------------------

tmp =1;


for(k=0;k<g;k++) { //the third dimension 

	for(i=0;i<p;i++) // row
		for(j=i;j<p;j++) // column
		sigm[j*p+i]=sigma[k*p*p+j*p+i];

F77_SUB(inverse3)(sigm, inv, &lgdet, &p, &info,&count, save);

if(info) {*pinfo=5;return;}

tmp = tmp*lgdet;

}


Rprintf("at %3d th iteration, prod(det(sigma_i))=%12.5f, the ratio (divided by previous one) is %9.5f \n",it,tmp,tmp/tmp2);

tmp2=tmp;

------------------------------------------------
*/






//  M-step
F77_SUB(mstepmvn)(y,pn,pp,pg,pncov,tau,sumtau,mu,sigma);


//  Check if converge

if(it == (*itmax-1) ) {*pinfo = 1;break;}
	
if(it < imin2(19,*itmax-1)) continue;

if( (fabs(lk[it-10]-lk[it] ) < fabs(lk[it-10]* (*epsilon)))
&& (fabs(lk[it-1]-lk[it]) < fabs(lk[it-1]*(*epsilon)))) break;   

}      

//   end of EM algorithm loop


return;
}

/*

mixture of multivariate t distributions

*/


void emmvt_(double *y, int *pn, int *pp,int *pg,int *pncov,
double * pro, double *mu, double * sigma, double *dof,
double *tau,double *xuu,
double *sumtau, double *sumxuu,double *sumxuuln,
double *loglik, double *lk, int *pinfo,int *itmax, double *epsilon)
{
int it;

double zero=0.0;

for(it = 0;it < *itmax;it++) lk[it]=zero;


//   start of EM algorithm loop

*pinfo =0;

      
for(it = 0;it < *itmax;it++) {

 
//  E-Step
F77_SUB(estepmvt)(y,pn,pp,pg,pro,mu,sigma,dof,tau,
	xuu,sumtau,sumxuu,sumxuuln,loglik,pinfo);

if(*pinfo) {*pinfo = *pinfo+10; return;}

lk[it]= *loglik;


//  M-step
F77_SUB(mstepmvt)(y,pn,pp,pg,pncov,tau,xuu,sumtau,sumxuu,sumxuuln,mu,sigma,dof);


//  Check if converge

if(it == (*itmax-1) ) {*pinfo = 1;break;}
	
if(it < imin2(19,*itmax-1)) continue;

if( (fabs(lk[it-10]-lk[it] ) < fabs(lk[it-10]* (*epsilon)))
&& (fabs(lk[it-1]-lk[it])< fabs(lk[it-1]*(*epsilon)))) break;   

}      

//   end of EM algorithm loop

return;
}



void emmsn_(double *y, int *pn, int *pp,int *pg, int *pncov,
double * pro, double *mu, double * sigma, double *delta,
double *tau, double *ev, double *vv, 
double *sumtau, double *sumev,
double *loglik, double *lk, int *pinfo,int *itmax,double *epsilon)
{
  
 
int it;

const double zero=0.0;



for(it = 0;it < *itmax;it++) lk[it]=zero;
      

//   start of EM algorithm loop

*pinfo = 0;

for(it = 0;it < *itmax;it++) {

 
//  E-Step
F77_NAME(estepmsn)(y,pn,pp,pg,pro,mu,sigma,delta,
     tau,ev,vv,sumtau,sumev,loglik,pinfo);

if(*pinfo) {*pinfo = *pinfo+10; return;}

lk[it]= *loglik;

//  M-step

F77_NAME(mstepmsn)(y,pn,pp,pg,pncov,tau,ev,vv,sumtau,sumev,mu,sigma,delta);


//  Check if converge

if(it == (*itmax-1) ) {(*pinfo) = 1;break;}
	
if(it < imin2(19,*itmax-1)) continue;

if( (fabs(lk[it-10]-lk[it] ) < fabs(lk[it-10]* (*epsilon)))
&& (fabs(lk[it-1]-lk[it])< fabs(lk[it-1]*(*epsilon)))) break;   

}      

//   end of EM algorithm loop


return;
}




void emmst_(double *y, int *pn, int *pp,int *pg, int *pncov,
double * pro, double *mu, double * sigma, double *dof, double *delta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *sumtau, double *sumvt,double *sumzt,double *sumlnv, 
double *loglik, double *lk, int *pinfo, int *itmax,  double *epsilon)
{
int it,g=*pg;
int meth[g];

double zero=0.0, bx=200.0;

for(it=0;it<g;it++) {
meth[it] =1;
}


for(it = 0;it < *itmax;it++) lk[it]=zero;
      

//   start of EM algorithm loop

*pinfo =0;

for(it = 0;it < *itmax;it++) {

 
//  E-Step
F77_NAME(estepmst)(y,pn,pp,pg,pro,mu,sigma,dof,delta,
     tau,ev,elnv,ez1v,ez2v,sumtau,sumvt,sumzt,sumlnv,
     loglik,pinfo,meth);

if(*pinfo) {*pinfo = *pinfo+10; return;}

lk[it]= *loglik;


//  M-step

F77_NAME(mstepmst)(y,pn,pp,pg,pncov,
	tau,ev,ez1v,ez2v,sumtau,sumvt,sumzt,
    mu,sigma,delta);



// calculate the degrees of freedom

F77_SUB(getdof)(pn, pg, sumtau, sumlnv, dof, &bx);

//  auto switch:  discarded

//for(h=0;h< *pg;h++) {  
//    if(meth[h] == 3) dof[h] = (double)50;
//    if(meth[h] == 3) dof[h] = (double)4;
//    if(dof[h] >= bx) meth[h]=meth[h]+1; 
//    if(dof[h] >= bx) meth[h] = 3;
//	

//}


//  Check if converge

if(it == (*itmax-1) ) {*pinfo = 1;break;}
	
if(it < imin2(19,*itmax-1)) continue;

if( (fabs(lk[it-10]-lk[it] ) < fabs(lk[it-10]* (*epsilon)))
&& (fabs(lk[it-1]-lk[it])< fabs(lk[it-1]*(*epsilon)))) break;   

}      

//   end of EM algorithm loop

return;
}


