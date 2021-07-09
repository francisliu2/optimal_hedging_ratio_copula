\name{EmSkewfit}
\alias{EmSkewfit}
\alias{EmSkewfit1}
\alias{EmSkewfit2}
\title{Fit the Multivariate Skew Mixture Models}
\description{
The engines to fit the data into mixture models using initial partition or initial values.
set. 
}
\usage{
EmSkewfit1(dat, g, clust,        distr, ncov, itmax, epsilon,initloop=20)
EmSkewfit2(dat, g,         init, distr, ncov, itmax, epsilon)
}
\arguments{
\item{dat}{The dataset, an n by p numeric matrix, where n is number of observations and p the dimension of data.}
\item{g}{The number of components of the mixture model}
\item{distr}{ A three letter string indicating the type of distribution to be fit. See Details.}
\item{ncov}{ A small integer indicating the type of covariance structure. See Details.}
\item{clust}{A vector of integers specifying the initial partitions of the data}	
\item{init}{ A list containing the initial parameters for the mixture model. See details.}	
\item{itmax}{ A big integer specifying the maximum number of iterations to apply}
\item{epsilon}{A small number used to stop the EM algorithm loop when the relative difference between log-likelihood at each iteration become sufficient small.}
\item{initloop}{A integer specifying the number of initial loops}
}
\details{
The distribution type, determined by the \code{distr} parameter, which may take any one of the following values:
"mvn" for a multivariate normal, "mvt" for a multivariate t-distribution, "msn" for a multivariate skew normal distribution and "mst" for a multivariate skew t-distribution.

The covariance matrix type, represented by the \code{ncov} parameter, may be any one of the following:
\code{ncov}=1 for a common variance, \code{ncov}=2 for a common diagonal variance, \code{ncov}=3 for a general variance, \code{ncov} =4 for a diagonal variance, \code{ncov}=5 for 
sigma(h)*I(p)(diagonal covariance with same identical diagonal element values).

The parameter \code{init} is a list with elements: \code{pro}, a numeric vector of the mixing proportion of each component; \code{mu}, a p by g matrix with each column as its corresponding mean; 
\code{sigma}, a three dimensional p by p by g array with its jth component matrix (p,p,j) as the covariance matrix for jth component of mixture models;
\code{dof}, a vector of degrees of freedom for each component; \code{delta}, a p by g matrix with its columns corresponding to skew parameter vectors.

}
\value{
  \item{error}{Error code, 0 = normal exit;  1 = did not converge within \code{itmax} iterations;   2 = failed to get the initial values; 3 = singularity}
  \item{aic}{Akaike Information Criterion (AIC) }
  \item{bic}{Bayes Information Criterion (BIC)}
  \item{pro}{A vector of mixing proportions, see Details.}
  \item{mu}{A numeric matrix with each column corresponding to the mean, see Details.}
  \item{sigma}{An array of dimension (p,p,g) with first two dimension corresponding covariance matrix of each component, see Details.}
  \item{dof}{A vector of degrees of freedom for each component, see Details.}
  \item{delta}{A p by g matrix with each column corresponding to a skew parameter vector, see Details.}
  \item{clust}{A vector of final partition}
  \item{loglik}{The loglikelihood at convergence}
  \item{lk}{A vector of loglikelihood at each EM iteration}
  \item{tau}{An n by g matrix of posterior probability for each data point}
}
\references{

McLachlan G.J. and Krishnan T. (2008). The EM Algorithm and Extensions (2nd). New Jersay: Wiley.

McLachlan G.J. and Peel D. (2000). Finite Mixture Models. New York: Wiley.
}
\seealso{\code{\link{init.mix}},\code{\link{initEmmix}},\code{\link{EmSkew}},
\code{\link{rdemmix}},\code{\link{rdemmix2}},\code{\link{rdmvn}},\code{\link{rdmvt}},\code{\link{rdmsn}},
\code{\link{rdmst}}.
}
\examples{

n1=300;n2=300;n3=400;
nn <-c(n1,n2,n3)
n=1000
p=2
ng=3


sigma<-array(0,c(2,2,3))
for(h in 2:3) sigma[,,h]<-diag(2)
sigma[,,1]<-cbind( c(1,0),c(0,1))
mu  <- cbind(c(4,-4),c(3.5,4),c( 0, 0))

# for other distributions, 
#delta <- cbind(c(3,3),c(1,5),c(-3,1))
#dof    <- c(3,5,5)


pro   <- c(0.3,0.3,0.4)



distr="mvn"
ncov=3

#first we generate a data set
set.seed(111) #random seed is set 
dat <- rdemmix(nn,p,ng,distr,mu,sigma,dof=NULL,delta=NULL)

#start from initial partition
clust<- rep(1:ng,nn)
obj1 <- EmSkewfit1(dat, ng, clust, distr, ncov, itmax=1000, epsilon=1e-4)


#start from initial values
#alternatively, if we define initial values like 

init<-list()

init$pro<-pro
init$mu<-mu
init$sigma<-sigma


# for other distributions, 
#delta <- cbind(c(3,3),c(1,5),c(-3,1))
#dof    <- c(3,5,5)
#init$dof<-dof
#init$delta<-delta

obj2 <-EmSkewfit2(dat, ng, init, distr, ncov,itmax=1000, epsilon=1e-4)
}
\keyword{cluster}
\keyword{datasets}
