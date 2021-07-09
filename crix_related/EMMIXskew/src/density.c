#include "emmix.h"


void ddmix(double *x, int *pn, int *pp, int *pg,int *pdist,
const double *mu, const double *sigma, const double *dof,const double *delta,
double *den, int *pinfo)
{

if     (*pdist == 1)
	 F77_SUB(denmvn)(x,pn,pp,pg,mu,sigma,           den,pinfo);      
else if(*pdist == 2)
	 denmvt(x,pn,pp,pg,mu,sigma,dof,                den,pinfo);      
else if(*pdist == 3)
	 F77_NAME(denmsn)(x,pn,pp,pg,mu,sigma,    delta,den,pinfo);      
else if(*pdist == 4)
	 F77_NAME(denmst)(x,pn,pp,pg,mu,sigma,dof,delta,den,pinfo); 


return;

}

      
void denmvt(double *y, int *pn, int *pp, int *pg,
const double *mu, const double *sigma, const double *dof,
double *den, int *pinfo)
{
int n = * pn, p = *pp, g = * pg;
	  
// some constant variables:

double logpi = 2.0*M_LN_SQRT_PI;

double two =  2.0,one =  1.0,minus = -1.0, zero = 0.0;

int  incx=1,incy=1;
char trans[]="N";

// local variables  
 
double sigm[p*p],inv[p*p],wy[p],wx[p],wd[p]; 
double temp,tempyy,lgdet;
double lgconst;
int i,j,k,count,info =0,save[p];


//--------------------------------------------------

//sigm[i][j] = sigma[i][j][k]

for(k=0;k<g;k++) { //the third dimension 
	for(i=0;i<p;i++) // row
		for(j=i;j<p;j++) // column
		sigm[j*p+i]=sigma[k*p*p+j*p+i];

F77_SUB(inverse3)(sigm, inv, &lgdet, &p, &info,&count, save);

if(info) {*pinfo=5;return;}


/*


tau[i][k] = log(f_k(y_i)


*/
lgconst = lgammafn( (dof[k]+(double)p )/two )-(log(dof[k])+logpi)*((double)p)/two
	-lgammafn(dof[k]/two) - log(lgdet)/two;

F77_NAME(dcopy)( &p, mu+(k*p), &incx, wx, &incy);


for(i=0;i<n;i++) {
F77_NAME(dcopy)( &p, y+i, &n, wy, &incy);
F77_NAME(daxpy)( &p, &minus, wx, &incx, wy, &incy);

/* y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y,  */
F77_NAME(dgemv)(trans, &p, &p, &one, inv, &p, wy, &incx, &zero, wd, &incy);


tempyy = F77_NAME(ddot)(&p, wd, &incx, wd,&incy);

temp = -log(one+tempyy/dof[k])*(dof[k]+ (double)p)/two;

den[k*n+i] = (lgconst+temp);


} // end i loop


}  // end k loop

//-------------------------------------------------------

return;

}

void F77_SUB(denmvt2)(double *y, int *pn, int *pp, int *pg,
const double *mu, const double *sigma, const double *dof,
double *den,double *xuu, int *pinfo)
{
int n = * pn, p = *pp, g = * pg;
	  
// some constant variables:

double logpi = 2.0*M_LN_SQRT_PI;

double two =  2.0,one =  1.0,minus = -1.0, zero = 0.0;

int incx=1,incy=1;
char trans[]="N";

// local variables  
 
double sigm[p*p],inv[p*p],wy[p],wx[p],wd[p]; 
double temp,tempyy,lgdet;
double lgconst;
int i,j,k,info =0,count, save[p];


//--------------------------------------------------

//sigm[i][j] = sigma[i][j][k]

for(k=0;k<g;k++) { //the third dimension 
	for(i=0;i<p;i++) // row
		for(j=i;j<p;j++) // column
		sigm[j*p+i]=sigma[k*p*p+j*p+i];

F77_SUB(inverse3)(sigm, inv, &lgdet, &p, &info,&count, save);

if(info) {*pinfo=5;return;}


/*


tau[i][k] = log(f_k(y_i))


*/
lgconst = lgammafn( (dof[k]+(double)p )/two )-(log(dof[k])+logpi)*((double)p)/two
	-lgammafn(dof[k]/two) - log(lgdet)/two;

F77_NAME(dcopy)( &p, mu+(k*p), &incx, wx, &incy);


for(i=0;i<n;i++) {
F77_NAME(dcopy)( &p, y+i, &n, wy, &incy);
F77_NAME(daxpy)( &p, &minus, wx, &incx, wy, &incy);

/* y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y,  */
F77_NAME(dgemv)(trans, &p, &p, &one, inv, &p, wy, &incx, &zero, wd, &incy);


tempyy = F77_NAME(ddot)(&p, wd, &incx, wd,&incy);

temp = -log(one+tempyy/dof[k])*(dof[k]+ (double)p)/two;

den[k*n+i] = (lgconst+temp);
xuu[k*n+i] = (dof[k]+ (double)p)/(dof[k]+tempyy);


} // end i loop


}  // end k loop

//-------------------------------------------------------

return;

}


void F77_SUB(denmvn)(double *y, int *pn, int *pp, int *pg,
const double * mu, const double * sigma, double *den, int *pinfo)
{
int n = * pn, p = *pp, g = * pg;
	  
// some constant variables:

double lg2pi = 2.0*M_LN_SQRT_2PI; //1.837877066409345;

double two =  2.0,one =  1.0,minus = -1.0, zero = 0.0;

int incx=1,incy=1;
char trans[]="N";

// local variables  
 
double sigm[p*p],inv[p*p],wy[p],wx[p],wd[p]; 
double temp,lgdet;
double lgconst;
int i,j,k,count,info =0,save[p];


//--------------------------------------------------

//sigm[i][j] = sigma[i][j][k]

for(k=0;k<g;k++) { //the third dimension 
	for(i=0;i<p;i++) // row
		for(j=i;j<p;j++) // column
		sigm[j*p+i]=sigma[k*p*p+j*p+i];

F77_SUB(inverse3)(sigm, inv, &lgdet, &p, &info,&count, save);

if(info) {*pinfo=5;return;}


/*


tau[i][k] = log(f_k(y_i)


*/
lgdet = log(lgdet);
lgconst = ((double)p * lg2pi + lgdet)/two;

F77_NAME(dcopy)( &p, mu+(k*p), &incx, wx, &incy);


for(i=0;i<n;i++) {
F77_NAME(dcopy)( &p, y+i, &n, wy, &incy);
F77_NAME(daxpy)( &p, &minus, wx, &incx, wy, &incy);

/* y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y,  */
F77_NAME(dgemv)(trans, &p, &p, &one, inv, &p, wy, &incx, &zero, wd, &incy);


temp = F77_NAME(ddot)(&p, wd, &incx, wd,&incy);

den[k*n+i] = - (lgconst+temp/two);

} // end i loop


}// end k loop


return;
}
