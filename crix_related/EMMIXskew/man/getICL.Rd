\name{getICL}
\alias{getICL}
\title{The ICL criterion }
\description{
Calculate the Integrated Completed Likelihood(ICL) criterion 
}
\usage{
getICL(x, n, p, g, distr, ncov, pro, mu, sigma, dof, delta, clust)
}
\arguments{
\item{x}{An n by p data matrix}
\item{n}{The total number of points}
\item{p}{Dimension of data}
\item{g}{the number of components of the mixture model}
\item{distr}{A three letter string indicating the type of distribution to be fit.}
\item{ncov}{A small integer indicating the type of covariance structure.}
\item{pro}{A vector of mixing proportions}
\item{mu}{A numeric matrix with each column corresponding to the mean}
\item{sigma}{An array of dimension (p,p,g) with first two dimension corresponding covariance matrix of each component}
\item{dof}{A vector of degrees of freedom for each component}
\item{delta}{A p by g matrix with each column corresponding to a skew parameter vector}
\item{clust}{A vector of partition}
}
\value{
\item{ICL}{ICL value}
}
\references{
Biernacki C. Celeux G., and Govaert G. (2000). Assessing a Mixture Model for Clustering with the integrated Completed Likelihood.  IEEE Transactions on Pattern Analysis and Machine Intelligence. 22(7). 719-725.
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

pro   <- c(0.3,0.3,0.4)

distr="mvn"
ncov=3

#first we generate a data set
set.seed(111) #random seed is set 
dat <- rdemmix(nn,p,ng,distr,mu,sigma,dof=NULL,delta=NULL)

#start from initial partition
clust<- rep(1:ng,nn)
obj <- EmSkewfit1(dat, ng, clust, distr, ncov, itmax=1000,epsilon=1e-4)

getICL(dat,n,p,ng, distr,ncov,obj$pro,obj$mu,obj$sigma,obj$dof,
obj$delta,obj$clust)

}
\keyword{cluster}
\keyword{datasets}
