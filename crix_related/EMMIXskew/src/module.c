
#include "emmix.h"

//------------------------------

// some functions used by EMMIX

//------------------------------



double F77_SUB(mygammln)(double * x)
{
return(lgammafn(x[0]));
}


double F77_SUB(mvphit)(double *x, double *v)
{
return(pt(x[0],v[0],1,0));
}



double F77_SUB(mvphin)(double *x)
{
double mu=0.,sd=1.;

return(pnorm5(x[0],mu,sd,1,0));
}




double F77_SUB(mydnorm)(double *x)
{
double mu=0.,sd=1.;
return(dnorm4(x[0],mu,sd,0));
}

double F77_SUB(mydigamma)(double *x)
{

return(digamma(x[0]));

}




//the max and min of the diagonal elements of a p by p matrix v
void absrng_(double * v, int* p, double * vmin, double * vmax)
{
int m = *p,i;  
double tmp;

tmp= fabs(v[0]);
vmin[0] = tmp;
vmax[0] = tmp;

if(m==1) return;

for(i=1;i< m;i++)
{
    tmp = fabs(v[i*m+i]);
	vmin[0] = fmin2(tmp,vmin[0]);
	vmax[0] = fmax2(tmp,vmax[0]);
}
return;
}


void nonzeromax_(double *v, int *p, double *vmax)
{
int m = *p,i;  
double tmp;
double zero=0.0;

vmax[0] = zero;

for(i=0;i<m;i++) {


if(fabs((tmp = v[i]))> zero) {vmax[0] = tmp;break;}

} //end loop

if(m==1) return;

for(i=0;i< m;i++)
{
    if(fabs((tmp = v[i]))> zero){
	vmax[0] = (tmp > vmax[0])? tmp: vmax[0];}
}
return;
}




void Zeroin_(			/* An estimate of the root */
    double *ax,				/* Left border | of the range	*/
    double *bx,				/* Right border| the root is seeked*/
    double *fa,double *fb,
	double (*f)(double x, void *info),	/* Function under investigation	*/
    void *info,				/* Add'l info passed on to f	*/
	double *Tol,			/* Acceptable tolerance		*/
    int *Maxit,				/* Max # of iterations */
	double *root)           /* returned root    */
{
	
*root= R_zeroin2(*ax, *bx,*fa,*fb,f, info,Tol, Maxit);
	
return;
	
}


double Tequ(double dof,void *pinfo)
{
MINFO *myinfo;
myinfo=pinfo;
double x = dof/(double)2;
return ( ( log(x)-digamma(x)+(double)1 )*(*myinfo).stau + (*myinfo).sxuu);

}


void F77_SUB(getdof)(int *pn, int *pg, double *sumtau, double *sumxuuln, double *dof, double *b)
{

int h, n=*pn, g = *pg;

int Maxit;
MINFO pinfo;

double ax=2.0001,bx= *b,root=0.0,Tol;
double fa,fb;

//bx = (double)200;

// calculate dof[k]

for(h=0;h<g;h++) {

if(sumtau[h] >= 2.0) {


Maxit=30;
Tol=1e-3;

pinfo.stau = sumtau[h]/(double)n;
pinfo.sxuu = sumxuuln[h]/(double)n;

fa = Tequ(ax, &pinfo);
fb = Tequ(bx, &pinfo);

Zeroin_(&ax,&bx,&fa,&fb,Tequ,&pinfo,&Tol,&Maxit,&root);	

dof[h]= root;

//printf("\nMAxit=%5d, Tol = %10.5f",Maxit,Tol);

if( (Maxit <0 && Tol > 1e-4) || root <= ax)
dof[h]=bx;


} //end if
else
dof[h]=4.0;


} // end for

return;
}






void myrevsort_(double  *a, int *ib, int *pn)
{
int ii, j, ir, i,n=*pn;
double ra;

if(n == 1) return;

for(i=1;i<n;i++){
    
	ir = n-i;
	ra =  a[ir];
	ii = ib[ir];

	for(j=0;j< ir;j++){

	if(ra > a[j]) {
		ra =  a[j];
		ii = ib[j];
	
		 a[j] =  a[ir];
		ib[j] = ib[ir];

		 a[ir] = ra;
		ib[ir] = ii;
	} //endif

	}//end j loop
} //end i loop

return; 

}


void oldmyrevsort_(double *a, int *ib, int *pn)
{
revsort(a, ib, *pn);

// this program can not sort (1,1,1) properly!!!


return;
}



void SingularityHandler(double *sigma, double * sigm,double *inv, int *pp, int *pcount, int * save,double eps)
{

int p = *pp, info=0;
char uplo[]="U";
int i,j,k,ii,idx[p];

double wy[p]; 
double zero=0.0;


//------------------------------------------------

*pcount =0;
//start of singularity handler


//initialize some variables
for(i=0;i<p;i++){
wy[i]=sigm[i*p+i];
idx[i]=i;
save[i]=0;
} //end for i

myrevsort_(wy,idx,&p);



//sigm[i][j] = sigma[i][j]

for(i=0;i<p;i++) // row
	for(j=0;j<p;j++) // column
		sigm[j*p+i]=sigma[j*p+i];


//------------------------------------------------
// start of main loop

for(i=0;i<p;i++) {
j=idx[p-i-1];

// redefine the marginal distributions
for(k=0;k<p;k++){
sigm[j*p+k]=zero;
sigm[k*p+j]=zero;}//endfor k


sigm[j*p+j]=eps;
save[i]=j;
(*pcount)++;



for(k=0;k<p;k++) // row
	for(ii=0;ii<p;ii++) // column
		inv[ii*p+k]=sigm[ii*p+k];


info=0;
// Cholesky decomposition for inv
F77_NAME(dpotrf)(uplo,&p, inv, &p, &info);

if(!info && i < (p-1)){
j=idx[p-i-2];
if(sigm[j*p+j] >= eps) break;
} 


} 
// end of main loop
//------------------------------------------------

return;

}



// inverse of semi positive definite symmetrirc matrix
void F77_SUB(inverse3)(double *sigma, double * inv, double *pdet, 
int *pp, int * pinfo,int *pcount, int *save)
{

int p = *pp, info=0;
int incx = 1,incy = 1;
char uplo[]="U",trans[]="T",diag[]="N";
int i,j,k,ii;

double vmin,vmax,det=0.0; 
double sigm[p*p],wy[p]; 
double eps=1e-4,one =1.0,zero=0.0;

//--------------------------------------------------


// 1. handle p=1

*pinfo = 0;
*pcount=0;


if(p==1) {
     
	 if(sigma[0] < (eps/(double)100) ) {
	        
			inv[0]=(double)10/sqrt(eps);
            *pdet = eps/(double)100;
			save[0]=0;
			*pcount=1;}
	 else {
	        inv[0]=one/sqrt(sigma[0]);
            *pdet = sigma[0]; 
			save[0]=0;
	    }

     return;
}

// 2. handle p > 1


//sigm[i][j] = sigma[i][j]

for(i=0;i<p;i++) {// row
	for(j=i;j<p;j++) {// column
		sigm[j*p+i]=sigma[j*p+i];}}


absrng_(sigm,&p,&vmin,&vmax);


F77_NAME(dpotrf)(uplo,&p, sigm, &p, &info);


//------------------------------------------------

//start of singularity handler

/* sinularity check*/
if( info != 0 || vmin < eps ) {

//if( info != 0 && vmin < eps ) {

/*
when sigm is degenerated, we need handle it
*/

SingularityHandler(sigma,sigm,inv, &p, pcount,save,eps);


// Cholesky decomposition for sigm
F77_NAME(dpotrf)(uplo,&p, sigm, &p, &info);

}//endif

// end of singularity handler


//------------------------------------------------

// Calculate the determinant 
det = one;
for(j=0;j<p;j++)  
    det *= sigm[j*p+j];
det *= det;


//-----------------------------------------------


// inv = solve(sigm)

for(k=0;k<p;k++) {// row
	for(ii=0;ii<p;ii++) // column
   	    inv[ii*p+k]=zero;
    inv[k*p+k]= one;
}

for(j=0;j< p;j++) 
{
F77_NAME(dcopy)( &p, inv+j*p, &incx, wy, &incy);
F77_NAME(dtrsv)(uplo,trans,diag,&p, sigm, &p, wy, &incy);
F77_NAME(dcopy)( &p, wy, &incx, inv+j*p, &incy);
}


//  inv = solve(sigm)

//-----------------------------------------------


*pinfo  = info;
*pdet   = det;

return;
}

// inverse of semi positive definite symmetrirc matrix
void inverse4_(double *sigma, double * inv,int *pp, int *pcount, int * save)
{

int p = *pp, info=0;
int incx = 1,incy = 1;
char uplo[]="U",trans[]="T",diag[]="N";
int i,j,k,ii;

double vmin,vmax; 
double sigm[p*p],wy[p]; 
double eps=1e-4,one =1.0,zero=0.0;

//--------------------------------------------------


// 1. handle p=1

*pcount=0;

if(p==1) {
     
	 if(sigma[0] < (eps/(double)100) ) {
	        inv[0]=(double)10/sqrt(eps);
			save[0]=0;*pcount=1;}
	 else {
	        inv[0]=one/sqrt(sigma[0]);
			save[0]=0;
	    }
	 return;
}

// 2. handle p > 1


//sigm[i][j] = sigma[i][j]

for(i=0;i<p;i++) {// row
	for(j=0;j<p;j++) {// column
		sigm[j*p+i]=sigma[j*p+i];}}


absrng_(sigm,&p,&vmin,&vmax);


F77_NAME(dpotrf)(uplo,&p, sigm, &p, &info);


//------------------------------------------------

//start of singularity handler

/* sinularity check*/
if( info != 0 || vmin < eps ) {

//if( info != 0 && vmin < eps ) {
/*
when sigm is degenerated, we need handle it

when vmin is too small, we may have overflow/underflow problem!
*/

SingularityHandler(sigma,sigm,inv, &p, pcount, save,eps);

for(i=0;i< *pcount; i++) {
	j=save[i];
	sigm[j*p+j]=one;
}

//------------------------------------------------

// Cholesky decomposition for sigm
F77_NAME(dpotrf)(uplo,&p, sigm, &p, &info);

}//endif

// end of singularity handler


//------------------------------------------------


// inv = solve(sigm)

for(k=0;k<p;k++) {// row
	for(ii=0;ii<p;ii++) // column
   	    inv[ii*p+k]=zero;
    
	inv[k*p+k]= one;
}

for(j=0;j< p;j++) 
{
F77_NAME(dcopy)( &p, inv+j*p, &incx, wy, &incy);
F77_NAME(dtrsv)(uplo,trans,diag,&p, sigm, &p, wy, &incy);
F77_NAME(dcopy)( &p, wy, &incx, inv+j*p, &incy);
}


//  inv = solve(sigm)


//-----------------------------------------------


return;
}



/*

getcov() is based on FORTRAN program emmix.f

c: ncov=1, common variance; nocv=2, common diagonal variance
c  nocv=3, general variance;ncov=4, diagonal variance.
c  nocv=5, sigma(h)*I_P


*/


void F77_SUB(getcov)(double *sigma, double *sumtau, int *pn,int *pp, int *pg, int *pncov)
{
int n= *pn,p=*pp,g=*pg,ncov=*pncov;
int i,j,k;
double sigm[p*p],tmp,zero=0;


//    Compute pooled estimate of common covariance matrix

if(ncov == 1 || ncov ==2) {

for(i=0;i<p;i++) {
	for(j=i;j<p;j++) {
       sigm[j*p+i]=zero;

       for(k=0;k<g;k++) 
	   sigm[j*p+i] += sumtau[k]*sigma[k*p*p+j*p+i];

       sigma[j*p+i]=sigm[j*p+i]/(double)n;
       sigma[j*p+i]=sigma[i*p+j];
}}



if(ncov == 2) {
for(i=0;i<p;i++) 
	for(j=0;j<p;j++) 
       if(i != j) sigma[j*p+i]=zero;  
}


if(g>1) {
for(k=1;k<g;k++)
  for(i=0;i<p;i++) 
	for(j=i;j<p;j++) {
       sigma[k*p*p+j*p+i]=sigma[j*p+i];
       sigma[k*p*p+i*p+j]=sigma[i*p+j];
}}

} //endif


if(ncov == 4 || ncov ==5) {


for(k=0;k<g;k++)
   for(i=0;i<p;i++) 
	for(j=0;j<p;j++) 
       if(i != j) sigma[k*p*p+j*p+i]=zero;  

// new staff here	
if(ncov == 5) {
   for(k=0;k<g;k++) {
	    
		tmp=zero;
        
		for(i=0;i<p;i++) 
	       tmp += sigma[k*p*p+i*p+i];
		
        for(i=0;i<p;i++) 
	       sigma[k*p*p+i*p+i]=tmp/(double)p; }}

} //end if

return;

}




/*

calculate the posterior probability

*/

void F77_SUB(gettau2)(double *tau,const double *pro, double *loglik, 
const int *pn, const int*pg, int *pinfo)
{
int n = * pn, g = * pg;
	  
// some constant variables:

double one =  1.0, zero = 0.0;


// local variables  
 
double prok,sum,temp;
double tmax,wx[g];
int i,k,incy=1;

*pinfo=0;*loglik=zero;

for(i=0;i<n;i++){


//------------------------------------

//  tau[i][k]= log( \pi_k * f_k(y_i))

//------------------------------------



F77_NAME(dcopy)( &g, tau+i, &n, wx, &incy);

for(k=0;k<g;k++) {
		prok=pro[k];
		if(prok <= zero) 
			wx[k]=zero;
		else 
			wx[k] += log(prok);
} //end k loop


nonzeromax_(wx,&g,&tmax);


if(fabs(tmax)<= zero )//{	*pinfo = 6;	   return;}
{
	for(k=0;k<g;k++) 
		wx[k]=zero;

	sum = one;
}
else {
//------------------------------------

//  sum =    sum _ i^k ( \pi_k * f_k(y_i))

//--------------------------------------


sum = zero;

for(k=0;k<g;k++) {
		if(pro[k] > zero) {
			temp=wx[k]-tmax;
			wx[k]=exp(temp);
			sum += wx[k];
		} // end if
} // end k loop

*loglik += log(sum)+tmax;

if(sum < one){*pinfo = 7;	   return;}

sum = one/sum;

} //end if

F77_NAME(dcopy)( &g, wx, &incy, tau+i, &n);

//------------------------------------

//  tau[i][k]= ( \pi_k * f_k(y_i)) / sum

//--------------------------------------

F77_NAME(dscal)( &g, &sum, tau+i, &n);

temp  = tau[i];
tmax = temp;
wx[0]= zero;

if(g>1) {

for(k=1;k< g;k++)//from second
{
    temp = tau[k*n+i];
	tmax = temp > tmax? temp:tmax;
	wx[k]= zero;
}  // end j loop

if(tmax<0.8)
F77_NAME(dcopy)( &g, wx, &incy, tau+i, &n);

} // end if


} // end i loop


return;

}


void F77_SUB(tau2clust2)(double * tau, int *pn, int* pg, int * clust)
{
int g = *pg, n= *pn,i,j;  
double tmp,vmax;


// tau[i,j]

for(i=0;i<n;i++) { 

tmp= (tau[i]);
vmax = tmp;
clust[i]=1;

if(g>1) {

for(j=1;j< g;j++)//from second
{
    tmp = (tau[j*n+i]);
    if(tmp > vmax) {
		vmax = tmp;
		clust[i]=j+1;
	}
}  // end j loop

if(vmax<0.8) clust[i]=0; //outliers!

} // end if

}  // end i loop

return;
}



void F77_SUB(gettau)(double *tau,const double *pro, double *loglik, 
const int *pn, const int*pg, int *pinfo)
{
int n = * pn, g = * pg;
	  
// some constant variables:

double one =  1.0, zero = 0.0;


// local variables  
 
double prok,sum,temp;
double tmax,wx[g];
int i,k,incy=1;

*pinfo=0;*loglik=zero;

for(i=0;i<n;i++){


//------------------------------------

//  tau[i][k]= log( \pi_k * f_k(y_i))

//------------------------------------



F77_NAME(dcopy)( &g, tau+i, &n, wx, &incy);

for(k=0;k<g;k++) {
		prok=pro[k];
		if(prok > zero) 
			wx[k]+= log(prok);
		else 
			wx[k]=zero; 
} //end k loop


nonzeromax_(wx,&g,&tmax);


if(fabs(tmax)<= zero ){	*pinfo = 6;	   return;}


//------------------------------------

//  sum =    sum _ i^k ( \pi_k * f_k(y_i))

//--------------------------------------


sum = zero;

for(k=0;k<g;k++) {
		if(pro[k] > zero) {
			temp=wx[k]-tmax;
			wx[k]=exp(temp);
			sum += wx[k];
		} // end if
} // end k loop

*loglik += log(sum)+tmax;

if(sum < one){*pinfo = 7;	   return;}

F77_NAME(dcopy)( &g, wx, &incy, tau+i, &n);
	

sum = one/sum;


//------------------------------------

//  tau[i][k]= ( \pi_k * f_k(y_i)) / sum

//--------------------------------------


F77_NAME(dscal)( &g, &sum, tau+i, &n);

} // end i loop


return;

}



void F77_SUB(tau2clust)(double * tau, int *pn, int* pg, int * clust)
{
int g = *pg, n= *pn,i,j;  
double tmp,vmax;


// tau[i,j]

for(i=0;i<n;i++) { 

tmp= (tau[i]);
vmax = tmp;
clust[i]=1;

if(g>1) {

for(j=1;j< g;j++)
{
    tmp = (tau[j*n+i]);
    if(tmp >= vmax) {
		vmax = tmp;
		clust[i]=j+1;
	}

}  // end j loop

} // end if

}  // end i loop

return;
}


