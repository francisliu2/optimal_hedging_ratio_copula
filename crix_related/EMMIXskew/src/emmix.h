#include <R.h>
#include <Rmath.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include <R_ext/Applic.h>




/*  Interface Functions */

// A structure for parameter estimation
typedef struct _EMMIX_{
double *pro;
double *mu;
double *sigma;
double *dof;
double *delta;
} EMMIX, *PEMMIX;


void initfit_( /*  get parameter estimates from initial partition*/
double *y, int *pn, int *pp,int *pg, int *pncov,int *pdist, 
double * pro, double *mu, double * sigma, double *dof, double *delta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *sumtau, double *sumvt,double *sumzt,double *sumlnv, 
double *ewy,double *ewz,double *ewyy, 
double *loglik, int *clust, int *pinfo, int *maxloop);


void emskewfit1( /*  start from initial partition */
double *y, int *pn, int *pp,int *pg,  int *pncov,int *pdist,
double *pro, double *mu, double * sigma, double *dof, double *delta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *sumtau, double *sumvt,double *sumzt,double *sumlnv, 
double *ewy,double *ewz,double *ewyy, 
double *loglik, double *lk, double *aic, double *bic,
int *clust, 
int *pinfo,  int *itmax, double *epsilon, int *maxloop);

void emskewfit2( /*  start from initial values */
double *y, int *pn, int *pp,int *pg, int *pncov, int *pdist, 
double *pro, double *mu, double * sigma, double *dof, double *delta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *sumtau, double *sumvt,double *sumzt,double *sumlnv, 
double *loglik, double *lk, double *aic, double *bic,
int *clust, 
int *pinfo,  int *itmax, double *epsilon);



/*

	main functions to do the EM algorithm

*/

void emmvn_( // multivariate normal distribution
double *y, int *pn, int *pp,int *pg,int *pncov,
double *pro, double *mu, double * sigma, //parameters
double *tau, double *sumtau,//conditional expectations
double *loglik, double *lk,  //loglikelihood values
int *pinfo,int *itmax, double *epsilon);

void emmvt_( // multivariate t distribution
double *y, int *pn, int *pp,int *pg,int *pncov,
double * pro, double *mu, double * sigma, double *dof,//parameters
double *tau,double *xuu,//conditional expectations
double *sumtau, double *sumvt,double *sumlnv, 
double *loglik, double *lk, //loglikelihood values
int *pinfo,int *itmax, double *epsilon);

void emmsn_( // multivariate skew normal distribution
double *y, int *pn, int *pp,int *pg, int *pncov,
double *pro, double *mu, double * sigma, double *delta,//parameters
double *tau, double *ev, double *vv, //conditional expectations
double *sumtau, double *sumvt,
double *loglik, double *lk, //loglikelihood values
int *pinfo, int *itmax,  double *epsilon);

void emmst_(// multivariate skew t distribution
double *y, int *pn, int *pp,int *pg, int *pncov,
double *pro, double *mu, double * sigma, double *dof, double *delta,//parameters
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,//conditional expectations
double *sumtau, double *sumvt,double *sumzt,double *sumlnv, 
double *loglik, double *lk, //loglikelihood values
int *pinfo,int *itmax, double *epsilon);


/* Initial Values */

void F77_SUB(initmvn)(  // multivariate normal distribution
double *y, int *pn, int *pp,int *pg,int *pncov, 
double *pro, double *mu, double *sigma, 
double *tau, double *, 
double *ewy,double *ewz,double *ewyy, 
double *loglik, int *clust,int *pinfo, int *);

void F77_SUB(initmvt)( // multivariate t distribution
double *y, int *pn, int *pp,int *pg,int *pncov, 
double *pro, double *mu, double *sigma, double *dof,
double *tau, double *xuu, 
double *sumtau, double *sumvt,double *sumlnv, 
double *ewy,double *ewz,double *ewyy, 
double *loglik, int *clust,int *pinfo, int *);


void F77_SUB(initmsn)( // multivariate skew normal distribution
double *y, int *pn, int *pp,int *pg, int *pncov,
double *pro, double *mu, double *sigma, double *delta,
double *tau, double *ev, double *vv, 
double *sumtau, double *sumev,
double *ewy,double *ewz,double *ewyy, 
double *loglik,int *clust,int *pinfo, int *);

void F77_SUB(initmst)( // multivariate skew t distribution
double *y, int *pn, int *pp,int *pg, int *pncov,
double *pro, double *mu, double *sigma, double *dof, double *delta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *sumtau, double *sumvt,double *sumzt,double *sumlnv, 
double *ewy,double *ewz,double *ewyy, 
double *loglik,int *clust,int *pinfo, int *);



/*


procedures of E-Step


*/

void F77_SUB(estepmvn)( // multivariate normal distribution
double *y, int *pn, int *pp, int *pg,
double *pro, double * mu, double * sigma, 
double * tau, double * sumtau,
double * loglik,int *pinfo);

void F77_SUB(estepmvt)( // multivariate t distribution
double *y, int *pn, int *pp, int *pg,
double *pro, double *mu, double *sigma, double *dof,
double *tau, double *xuu,double *sumtau,
double * sumxuu,double *sumxuuln, 
double * loglik,int * pinfo);

void F77_NAME(estepmsn)( // multivariate skew normal distribution
const double *y, int *pn, int *pp,int *pg,
double *pro, double *mu, double * sigma, double *delta,
double *tau, double *ev, double *vv, double *sumtau, double *sumev, 
double *loglik, int *pinfo);

void F77_NAME(estepmst)(// multivariate skew t distribution
double *y, int *pn, int *pp,int *pg,
double * pro, double *mu, double * sigma, double *dof, double *delta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *sumtau, double *sumvt, double *sumzt,double *sumlnv,
double *loglik, int *pinfo, int *method);





/*


procedures of M-Step


*/



void F77_SUB(mstepmvn)( // multivariate normal distribution
double *y,int *pn, int *pp, int *pg,int *pncov,
double *tau, double *sumtau, double *mu, double *sigma);

void F77_SUB(mstepmvt)( // multivariate t distribution
double *y,int *pn, int *pp, int *pg,int *pncov,
double *tau,double *xuu, double *sumtau, double *sumxuu, double *sumxuuln,
double *mu, double *sigma, double *dof);

void F77_NAME(mstepmsn)(// multivariate skew normal distribution
double *y, int *pn, int *pp,int *pg, int *pncov,
double *tau, double *ev, double *vv,double *sumtau, double *sumev, 
double *mu, double * sigma, double *delta);


void F77_NAME(mstepmst)( // multivariate skew t distribution
double *y, int *pn, int *pp,int *pg, int *pncov,
double *tau, double *ev, double *ez1v,double *ez2v,
double *sumtau, double *sumvt, double *sumzt,
double *mu, double * sigma, double *delta);







/* 

Calculate Density  

*/

void ddmix(double *x, int *pn, int *pp, int *pg,int *pdist,
const double *mu, const double *sigma, const double *dof,const double *delta,
double *den, int *pinfo);



void F77_SUB(denmvn)(// multivariate normal distribution
double *x, int *pn, int *pp, int *pg,
const double * mu, const double * sigma,
double *den, int *pinfo);

void denmvt(// multivariate t distribution
double *x, int *pn, int *pp, int *pg,
const double *mu, const double *sigma, const double *dof,
double *den, int *pinfo);


void F77_SUB(denmvt2)(// multivariate t distribution
double *y, int *pn, int *pp, int *pg,
const double *mu, const double *sigma, const double *dof,
double *den,double *xuu, int *pinfo);

void F77_NAME(denmst)(// multivariate skew t distribution
double *x, int *pn, int *pp, int *pg,
const double *mu, const double *sigma, const double *dof,const double *delta,
double *den, int *pinfo);



void F77_NAME(denmst2)(// multivariate skew t distribution
double *y, int *pn, int *pp, int *pg,
const double *mu, const double *sigma, const double *dof,const double *delta,
double *tau,double *ev,double *elnv,double *,double *,
int *error,int *method);

void F77_NAME(denmsn)(// multivariate skew normal distribution
double *x, int *pn, int *pp, int *pg,
const double *mu, const double *sigma, const double *delta,
double *den, int *pinfo);


void F77_NAME(denmsn2)(// multivariate skew normal distribution
double *y, int *pn, int *pp, int *pg,
const double *mu, const double *sigma, const double *delta,
double *tau,double *ev,double *vv,int *perror);




/*
-----------------------------------------------


Discriminant Analysis (DA)


-----------------------------------------------
*/

void F77_NAME(predmixdamsn)(
double *x,int *n,int *p,int *g, 
double *pro,double *mu,double *Sigma,             double *delta,
double *tau,int *info);

void F77_NAME(predmixdamst)(
double *x,int *n,int *p,int *g, 
double *pro,double *mu,double *Sigma,double *dof, double *delta,
double *tau,int *info);

void F77_NAME(emskewpred)(
double *x,int *n,int *p,int *g,int *dist, 
double *pro,double *mu,double *sigma,double *dof, double *delta,
double *tau,int *, int *info);



void F77_NAME(emskewda)(
double *x,int *n,int *p,int *g,int *ncov,int *dist, 
double *pro,double *mu,double *sigma,double *dof, double *delta,
double *tau,double *ev,double *elnv,double *,double *,
double *,double *,double *,double *,
double *ewy,double *ewz, double *ewyy,
double *loglik,double *lk,int *clust,int *itmax,double *epsilon,int *error);

void F77_NAME(emmvnda)(
double *x,int *n,int *p,int *g,int *ncov,
double *pro,double *mu,double *sigma,
double *tau,double *sumtau,
double *ewy,double *ewz,double *ewyy, 
double *loglik,double *lk,int *clust,int *itmax,double *epsilon,int *error);
      
void F77_NAME(emmvtda)(
double *x,int *n,int *p,int *g,int *ncov,
double *pro,double *mu,double *sigma,double *dof, 
double *tau,double *xuu,double *,double *,double *,
double *ewy,double *ewz,double *ewyy, 
double *loglik,double *lk,int *clust,int *itmax,double *epsilon,int *error);
      
void F77_NAME(emmsnda)(
double *x,int *n,int *p,int *g,int *ncov,
double *pro,double *mu,double *sigma,             double *delta,
double *tau,double *,double *,double *,double *,
double *ewy,double *ewz, double *ewyy,
double *loglik,double *lk,int *clust,int *itmax,double *epsilon,int *error);
      
void F77_NAME(emmstda)(
double *x,int *n,int *p,int *g,int *ncov,
double *pro,double *mu,double *sigma,double *dof, double *delta,
double *tau,double *,double *,double *,double *,
double *,double *,double *,double *,
double *ewy,double *ewz, double *ewyy,
double *loglik,double *lk,int *clust,int *itmax,double *epsilon,int *error);

void F77_NAME(estepmvnda)(
double *x,int *n,int *p,int *g,
double *pro,double *mu,double *sigma, 
double *ewy,double *ewz,double *ewyy, 
double *loglik,int *clust,int *error);
	  
void F77_NAME(estepmvtda)(
double *x,int *n,int *p,int *g,
double *pro,double *mu,double *sigma,double *dof, 
double *tau,double *xuu,
double *ewy,double *ewz,double *ewyy, 
double *loglik,int *clust,int *error);
      
void F77_NAME(estepmsnda)(
double *x,int *n,int *p,int *g,
double *pro,double *mu,double *sigma,double *delta,
double *tau,double *,double *,
double *ewy,double *ewz,double *ewyy, 
double *loglik,int *clust,int *error);
      
void F77_NAME(estepmstda)(
double *x,int *n,int *p,int *g,
double *pro,double *mu,double *sigma,double *dof, double *delta,
double *tau,double *,double *,double *,double *,
double *ewy,double *ewz,double *ewyy, 
double *loglik,int *clust,int *error,int *);



// Calculate Density Modes


void F77_NAME(emskewmod)(//
int *dist,int *p,int *g,double *step,
double *mu,double *Sigma,double *dof,double *delta,
double *emean,double *esigma,double *modpts,int *info);

void F77_NAME(mixmodmsn)(
int *p,int *g,double *step,
double *mu,double *Sigma,          double *delta,
double *emean,double *esigma,double *modpts,int *info);

void F77_NAME(mixmodmst)(
int *p,int *g,double *step,
double *mu,double *Sigma,double *dof,double *delta,
double *emean,double *esigma,double *modpts,int *info);

/*------------------------------------------------


Functions used to estimate the degrees of freedom (dof)


*/


extern double R_zeroin2(double ax, double bx, double fa, double fb, 
	  double (*f)(double x, void *info), void *info, 
	  double *Tol, int *Maxit);



// a wrapper of zeroin() in zeroin.c
void Zeroin_(			/* An estimate of the root */
double *ax,				/* Left border | of the range	*/
double *bx,				/* Right border| the root is seeked*/
double *fa,double *fb,
double (*f)(double x, void *info), //meantau,        /* mean tau    */
void *info,     /* double *meanxuuln,mean xuuln */
double *Tol,			/* Acceptable tolerance		*/
int *Maxit,				/* Max # of iterations */
double *root);         /* returned root   */

// A strcture used in T and Skew T distributions.

typedef struct _MINFO_ {
double stau;
double sxuu;
} MINFO,*PMINFO;



double Tequ(double dof,void *info);

void F77_SUB(getdof)(int *pn, int *pg, double *sumtau, double *sumxuuln, 
double *dof, double *bx);


//---------------------------------------




void F77_SUB(tau2clust)(double * tau, int *pn, int* pg, int * clust);


void F77_SUB(getcov)(double *sigma,double *sumtau,int *n,int *p,int *g,int *ncov);


void F77_SUB(gettau)(double *tau,const double *pro, double *loglik, 
const int *pn, const int*pg, int *pinfo);

void F77_SUB(gettau2)(double *tau,const double *pro, double *loglik, 
const int *pn, const int*pg, int *pinfo);

void F77_SUB(tau2clust2)(double * tau, int *pn, int* pg, int * clust);

/*

functions to get the inverse of semi positive definite symmetric matrix

*/
void SingularityHandler(
double *sigma, double * sigm,double *inv, 
int *pp, int *pcount, int * save,double eps);


void F77_SUB(inverse3)(double *sigma, double * inv, double * pdet, 
int *pp, int * pinfo,int *pcount, int * save);

void inverse4_(double *sigma, double * inv,
int *pp, int *pcount, int * save);


void absrng_(double * ,int* , double * , double * );

void nonzeromax_(double* v, int *p, double *vmax);

void myrevsort_(double *a, int *ib, int *pn);


/*

some functions for skew t mixture

*/


double fnn(double y,double dof, double p, int k);

void F77_SUB(intsum)(double *pux, double *pdist, double *pdof, 
double *ret, int *pp, int *pL);



/* Other Functions   */


void F77_NAME(skew)(/*  calculate the skewness of each variable  */
double *y,int *pn, int *pp,int *pg,
double *tau, double *sumtau,double *mu, double * sigma,double *delta);

double F77_SUB(mygammln)(double * x);
double F77_SUB(mvphit)(double *x, double *v);
double F77_SUB(mvphin)(double *x);


double F77_SUB(mydnorm)(double *x);
double F77_SUB(mydigamma)(double *x);



// distances
// 

void intradist_(double *y, int *pn, int *pp, int *pg, int * clust,
double * sigma, double * tau, double * dist, double *dist2, int * pinfo);

void interdist_(double *y, int *pn, int *pp, int *pg, int *clust,
double * sigma, double * tau, double * dist, double *dist2, int * pinfo);

void mahalonobis_(int *pp, int *pg, double * mu, 
double * sigma, double * dist, int * pinfo);


/*

void rdmvn_(int *pn, int *pp,int *pg,
double *mu,double * sigma,                               
double *y, int *pinfo);

void swap(double * smat,int , int , int);

void sortmat(double * sigma, double * smat, int *pp, int *order);

void reorder(double * smat, double *,int p, int* order);

void cholpivot(double *sigma, double *sigm, int *pp, int *pinfo);

*/

//--------------------

// joint clustering and alignment (jca)


void F77_NAME(denmst3)(// multivariate skew t distribution
double *y, int *pn, int *pp, int *pg,
const double * pro,const double *mu, const double *sigma, const double *dof,const double *delta,
double *tau,double *ev,double *elnv,double *,double *,
double *ewy,double *loglik,
int *error,int *method);

// EM-based Supervised Cluster Analysis (EM-SCA)


void emscamvn(double *y, int *pn, int *pp,int *pg, int *pncov,
double *pro, double *mu, double * sigma, double *dof, double *delta,
double *spro, double *smu, double * ssigma, double *sdof, double *sdelta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *umtau, double *umvt,double *umzt,double *umlnv,
double *sumtau, double *sumvt,double *sumzt,double *sumlnv,
double *ewy,double *ewz,double *ewyy,int *pnnn,
double *ewx,double *ewv,double *ewxx,
double *loglik, double *lk, int *itmax,  double *epsilon, int *pinfo);


void emscamvt(double *y, int *pn, int *pp,int *pg, int *pncov,
double *pro, double *mu, double * sigma, double *dof, double *delta,
double *spro, double *smu, double * ssigma, double *sdof, double *sdelta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *umtau, double *umvt,double *umzt,double *umlnv,
double *sumtau, double *sumvt,double *sumzt,double *sumlnv,
double *ewy,double *ewz,double *ewyy,int *pnnn,
double *ewx,double *ewv,double *ewxx,
double *loglik, double *lk, int *itmax,  double *epsilon, int *pinfo);


void emscamsn(double *y, int *pn, int *pp,int *pg, int *pncov,
double *pro, double *mu, double * sigma, double *dof, double *delta,
double *spro, double *smu, double * ssigma, double *sdof, double *sdelta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *umtau, double *umvt,double *umzt,double *umlnv,
double *sumtau, double *sumvt,double *sumzt,double *sumlnv,
double *ewy,double *ewz,double *ewyy,int *pnnn,
double *ewx,double *ewv,double *ewxx,
double *loglik, double *lk, int *itmax,  double *epsilon, int *pinfo);

void emscamst(double *y, int *pn, int *pp,int *pg, int *pncov,
double *pro, double *mu, double * sigma, double *dof, double *delta,
double *spro, double *smu, double * ssigma, double *sdof, double *sdelta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *umtau, double *umvt,double *umzt,double *umlnv,
double *sumtau, double *sumvt,double *sumzt,double *sumlnv,
double *ewy,double *ewz,double *ewyy,int *pnnn,
double *ewx,double *ewv,double *ewxx,
double *loglik, double *lk, int *itmax,  double *epsilon, int *pinfo);


void emmixsca(double *y, int *pn, int *pp,int *pg, int *pncov,int *pndist,
double *pro,double *mu, double * sigma, double *dof, double *delta,
double *spro,double *smu, double * ssigma, double *sdof, double *sdelta,
double *tau, double *ev, double *elnv, double *ez1v,double *ez2v,
double *umtau, double *umvt,double *umzt,double *umlnv,
double *sumtau, double *sumvt,double *sumzt,double *sumlnv, 
double *ewy,double *ewz,double *ewyy,int *pnnn,
double *ewx,double *ewv,double *ewxx,
double *loglik, double *lk,
int *clust, 
int *itmax,  double *epsilon, int *pinfo);


//-------------------------------------------------------------

void F77_SUB(scaestepmvn)(double *y,int *pn, int *pp, int *pg,
double *tau, 
double *mu, double *ety,double *etyy);

void F77_SUB(scaestepmvt)(double *y,int *pn, int *pp, int *pg,
double *tau,double *xuu, 
double *mu, double *ewy, double *ewyy);

void F77_NAME(scaestepmst)(double *y,int *pn, int *pp, int *pg,
double *tau,double *ev,double *ez1v,double *ez2v, 
double *mu, double *delta,double *ewy,double *ewz, double *ewyy);

void F77_NAME(scaestepmsn)(double *y,int *pn, int *pp, int *pg,
double *tau,double *ev,double *vv, 
double *mu, double *delta,double *ewy,double *ewz, double *ewyy);


/* the end of emmix.h */
