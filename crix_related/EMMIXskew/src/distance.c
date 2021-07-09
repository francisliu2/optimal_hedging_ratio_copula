

#include "emmix.h"




// three distances, intra,inter, manlonobis

void intradist_(double *y, int *pn, int *pp, int *pg, int * clust,
double * sigma, double * tau, double * dist, double *dist2, int * pinfo)
{

int n = * pn, p = *pp, g = * pg;
	  
double one =  1.0,minus = -1.0, zero = 0.0;

int incx=1,incy=1;
char trans[]="N";

// local variables  
 
double sigm[p*p],inv[p*p],wy[p],wx[p],wd[p]; 
double coco,temp,sum,tmp,sss;
int i,j,k,count,save[p];

double temp2,sum2,tttt,tmp2,sss2;


//--------------------------------------------------


temp = zero;sum  = zero;

temp2 = zero;sum2  = zero;

for(k=0;k<g;k++) { //the k th cluster
	
tmp = zero;sss = zero;


tmp2 = zero;sss2 = zero;

//-----------------------------------------------------------
//sigm[i][j] = sigma[i][j][k]

	
for(i=0;i<p;i++) // row
	for(j=0;j<p;j++) // column
		sigm[j*p+i]=sigma[k*p*p+j*p+i];



/*

*/

inverse4_(sigm,inv,&p,&count,save);


//-----------------------------------------------------------

//distance between yj,yi

//D1(j,i)

//-----------------------------------------------------------

for(j =0;j < n;j++) { // j th point


if(clust[j] != (k+1) ) continue;

F77_NAME(dcopy)( &p, y+j, &n, wy, &incy);

//Rprintf("\nhello, j=%3d\n",j);


for(i=j+1;i < n;i++) { //i th point

if(clust[i] != (k+1)) continue;


F77_NAME(dcopy)( &p, y+i, &n, wx, &incy);
F77_NAME(daxpy)( &p, &minus, wy, &incy, wx, &incx);


/* y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y,  */
//        call dgemv("N", p, p, one, inv, p, wx, 1, zero, wy, 1);

F77_NAME(dgemv)(trans, &p, &p, &one, inv, &p, wx, &incx, &zero, wd, &incy);

//dtrsv_(uplo,trans,diag,&p, sigm, &p, wy, &incy);

coco = (tau[k*n+i]*tau[k*n+j] );
tttt = ddot_(&p, wd, &incx, wd,&incy);




// tttt = (yi - yj) ^T (S) ^(-1) (yi - yj)
//Rprintf("\nhello,i=%3d, j=%3d, ddot(wy,wy)=%10.2f\n",i,j,tttt);




tmp += (tttt* coco) ;
sss  += coco;

tmp2 += tttt;
sss2 += one;


}  // end i loop


}  // end j loop

temp  +=tmp;
sum   +=sss;


if(sss >= 1.0) 
dist[k]=tmp/sss; 
else
dist[k]=zero;

temp2  +=tmp2;
sum2   +=sss2;

if(sss2 >= 1.0) 
dist2[k]=tmp2/sss2;
else
dist2[k]=zero;


}  // end k loop
//---------------------------------------------------------

if(sum >= 1.0) 
dist[g]=temp/sum;  // introdistance
else
dist[g]=zero;

if(sum2 >= 1.0) 
dist2[g]=temp2/sum2;  // introdistance
else
dist2[g]=zero;



return;
}


void interdist_(double *y, int *pn, int *pp, int *pg, int *clust,
double * sigma, double * tau, double * dist, double *dist2, int * pinfo)
{


int n = * pn,  p = * pp, g = *pg;
	  
// some constant variables:

double one =  1.0,minus = -1.0, zero = 0.0;

int  incx=1,incy=1;
char trans[]="N";

// local variables  
 
double sigm[p*p],wy[p],wx[p],wd[p],inv[p*p]; 
double coco,temp,sum, tmp,sss;
int i,j,k,r,count,save[p];


double temp2,sum2,tttt,tmp2,sss2;

if(g == 1) {
	dist[1]=zero;dist2[1]=zero;
	return;}

temp = zero;
sum  = zero;

temp2 = zero;
sum2  = zero;




for(k=0;k<g;k++) { //the k th cluster 
	dist[k*g+k]=zero;
	dist2[k*g+k]=zero;


for(r=k+1;r<g;r++) { // the r th clust


tmp = zero;
sss = zero;

tmp2 = zero;
sss2 = zero;

// D2(k,r)


//--------------------------------------------------

//sigm[i][j] = sigma[i][j][k] + sigma[i][j][r]

//-----------------------------------------------------------
	
for(i=0;i<p;i++) // row
	for(j=0;j<p;j++) // column
		sigm[j*p+i]=(sigma[k*p*p+j*p+i]+sigma[r*p*p+j*p+i]);


/*
        call inverse2(sigm,inv,logdet,p,info,counter,save)
*/



//inverse2_(sigm,inv,&det,&p,pinfo,&count,idx);
inverse4_(sigm,inv,&p,&count,save);

if(*pinfo != 0)
	 return;

//-----------------------------------------------------------

//distance between yj,yi

for(j =0;j < n;j++) { // j th point

if(clust[j] != (k+1)) continue;

F77_NAME(dcopy)( &p, y+j, &n, wy, &incy);


for(i=0;i < n;i++) { //i th point

if(clust[i] != (r+1)) continue;

F77_NAME(dcopy)( &p, y+i, &n, wx, &incx);
F77_NAME(daxpy)( &p, &minus, wy, &incy, wx, &incx);


/* y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y,  */
//        call dgemv("N", p, p, one, inv, p, wx, 1, zero, wy, 1);

F77_NAME(dgemv)(trans, &p, &p, &one, inv, &p, wx, &incx, &zero, wd, &incy);


//dtrsv_(uplo,trans,diag,&p, sigm, &p, wy, &incy);

coco = (tau[k*n+j]*tau[r*n+i] );
tttt = ddot_(&p, wd, &incx, wd,&incy);

// tttt = (yi - yj) ^T (s1+s2) ^(-1) (yi - yj)

//-----------------------------------------------------------




tmp += (tttt* coco) ;
sss  += coco;



tmp2 += tttt;
sss2 += one;

}  // end i loop

}  // end j loop


if(sss >= 1.0) {
dist[r*g+k] = tmp/sss;
dist[k*g+r] = tmp/sss;
}
else
{
dist[r*g+k] = zero;
dist[k*g+r] = zero;
}

if(sss2 >=1.0) {

dist2[r*g+k] = tmp2/sss2;
dist2[k*g+r] = tmp2/sss2;
}
else
{
dist2[r*g+k] = zero;
dist2[k*g+r] = zero;
}


temp += tmp;
sum  += sss;

temp2 += tmp2;
sum2  += sss2;


}  // end of r loop




}  // end of k loop
//---------------------------------------------------------



if(sum >= 1.0) 
dist[g*g] = temp/sum; // intercluster distance
else
dist[g*g]=zero;



if(sum2 >= 1.0) 
dist2[g*g] = temp2/sum2; // intercluster distance
else
dist2[g*g]=zero;


return;
}


void mahalonobis_(int *pp, int *pg, double * mu, 
double * sigma, double * dist, int * pinfo)
{

int p = *pp, g = *pg;
int count,save[p];
	  
// some constant variables:

double one =  1.0,minus = -1.0, zero = 0.0;

int  incx=1,incy=1;
char trans[]="N";

// local variables  
 
double sigm[p*p],wy[p],wx[p],inv[p*p]; 
int i,j,k,r;

pinfo[0]=0;


for(k=0;k<g;k++) { //the k th cluster 

dist[k*g+k]= zero;

for(r=k+1;r<g;r++) { // the r th clust


// D2(k,r)


//--------------------------------------------------

//sigm[i][j] = sigma[i][j][k] + sigma[i][j][r]

//-----------------------------------------------------------
	

for(i=0;i<p;i++) // row
	for(j=i;j<p;j++) // column
		sigm[j*p+i]=(sigma[k*p*p+j*p+i]+sigma[r*p*p+j*p+i]);


/*
*/

inverse4_(sigm,inv,&p,&count,save);


//-----------------------------------------------------------

//distance between mu_k,mu_r


F77_NAME(dcopy)( &p, mu+k*p, &incx, wy, &incy);
F77_NAME(dcopy)( &p, mu+r*p, &incx, wx, &incy);

F77_NAME(daxpy)( &p, &minus, wy, &incx, wx, &incy);

/* y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y,  */
//        call dgemv("N", p, p, one, inv, p, wx, 1, zero, wy, 1);

F77_NAME(dgemv)(trans, &p, &p, &one, inv, &p, wx, &incx, &zero, wy, &incy);

dist[r*g+k] = F77_NAME(ddot)(&p, wy, &incx, wy,&incy);

//-----------------------------------------------------------

dist[k*g+r] = dist[r*g+k];

}  // end of r loop

}  // end of k loop
//---------------------------------------------------------

return;
}

