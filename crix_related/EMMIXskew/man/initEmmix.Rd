\name{initEmmix}
\alias{initEmmix}
\alias{init.mix}
\title{Initialize Emmix Parameters}
\description{
Obtains intial parameter set for use in the EM algorithm. Grouping of the data occurs through one of three possible 
clustering methods: k-means, random start, and hierarchical clustering.
}
\usage{
initEmmix(dat, g, clust, distr, ncov,maxloop=20)
init.mix( dat, g, distr, ncov, nkmeans, nrandom, nhclust,maxloop=20)
}
\arguments{
\item{dat}{The dataset, an n by p numeric matrix, where n is number of observations and p the dimension of data.}
\item{g}{The number of components of the mixture model}
\item{distr}{ A three letter string indicating the type of distribution to be fit. See Details.}
\item{ncov}{ A small integer indicating the type of covariance structure. See Details.}
\item{clust}{An initial partition of the data  }
\item{nkmeans}{An integer to specify the number of KMEANS partitions to be used to find the best initial values}
\item{nrandom}{ An integer to specify the number of random partitions to be used to find the best initial values}
\item{nhclust}{A logical value to specify whether or not to use hierarchical cluster methods. If TRUE, the Complete Linkage method will be used.}
\item{maxloop}{An integer to specify how many iterations to be tried to find the initial values,the default value is 10.}
}
\details{
The distribution type, determined by the \code{distr} parameter, which may take any one of the following values:
"mvn" for a multivariate normal, "mvt" for a multivariate t-distribution, "msn" for a multivariate skew normal distribution and "mst" for a multivariate skew t-distribution.

The covariance matrix type, represented by the \code{ncov} parameter, may be any one of the following:
\code{ncov}=1 for a common variance, \code{ncov}=2 for a common diagonal variance, \code{ncov}=3 for a general variance, \code{ncov} =4 for a diagonal variance, \code{ncov}=5 for 
sigma(h)*I(p)(diagonal covariance with same identical diagonal element values).

The return values include following components: \code{pro}, a numeric vector of the mixing proportion of each component; \code{mu}, a p by g matrix with each column as its corresponding mean; 
\code{sigma}, a three dimensional p by p by g array with its jth component matrix (p,p,j) as the covariance matrix for jth component of mixture models;
\code{dof}, a vector of degrees of freedom for each component; \code{delta}, a p by g matrix with its columns corresponding to skew parameter vectors.

When the dataset is huge, it becomes time-consuming to use a large maxloop to try every initial partition. The default is 10.   
During the procedure to find the best inital clustering and intial values, for t-distribution and skew t-distribution, we don't estimate the degrees of freedom \code{dof}, instead they are fixed at 4 for each component.
}
\value{
  \item{pro}{A vector of mixing proportions, see Details.}
  \item{mu}{A numeric matrix with each column corresponding to the mean, see Details.}
  \item{sigma}{An array of dimension (p,p,g) with first two dimension corresponding covariance matrix of each component, see Details.}
  \item{dof}{A vector of degrees of freedom for each component, see Details.}
  \item{delta}{A p by g matrix with each column corresponding to a skew parameter vector, see Details.}
}
\references{ 
McLachlan G.J. and Krishnan T. (2008). The EM Algorithm and Extensions (2nd). New Jersay: Wiley.

McLachlan G.J. and Peel D. (2000). Finite Mixture Models. New York: Wiley.

}

\seealso{\code{\link{EmSkew}}}
\examples{
sigma<-array(0,c(2,2,3))
for(h in 2:3) sigma[,,h]<-diag(2)
sigma[,,1]<-cbind( c(1,0.2),c(0.2,1))
mu  <- cbind(c(4,-4),c(3.5,4),c( 0, 0))
delta <- cbind(c(3,3),c(1,5),c(-3,1))
dof    <- c(3,5,5)
pro   <- c(0.3,0.3,0.4)
n1=300;n2=300;n3=400;
nn<-c(n1,n2,n3)
n=1000
p=2
ng=3
distr="mvn"
ncov=3
#first we generate a data set
set.seed(111) #random seed is set 
dat <- rdemmix(nn,p,ng,distr,mu,sigma,dof,delta)
clust<- rep(1:ng,nn)
initobj1 <- initEmmix(dat,ng,clust,distr, ncov)
initobj2 <- init.mix( dat,ng,distr,ncov,nkmeans=10,nrandom=0,nhclust=FALSE)
}
\keyword{cluster}
\keyword{datasets}
