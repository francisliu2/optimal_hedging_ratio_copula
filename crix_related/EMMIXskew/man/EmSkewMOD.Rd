\name{EmSkewMOD}
\alias{EmSkewMOD}
\title{Calculate modes}
\description{
Calculate the mode points for each component of skew mixture models.
}


\usage{
EmSkewMOD(p,g,distr,mu,sigma,dof,delta,nrand=10000)
}
\arguments{
\item{p}{The dimension of the data}
\item{g}{The number of components to be fit}
\item{distr}{A three letter string of distribution id}
\item{mu }{A numeric matrix with each column corresponding to the mean}
\item{sigma}{An array of dimension (p,p,g) with first two dimensions corresponding covariance matrix of each component}
\item{dof }{A vector of degrees of freedom for each component}
\item{delta}{A matrix with each column as skew parameter vector}
\item{nrand}{The number of random numbers.}
}
\value{
A p by g matrix of mode points for each component of the skew mixture model.
}
\examples{

p=2
g=3

sigma<-array(0,c(2,2,3))
for(h in 2:3) sigma[,,h]<-diag(2)
sigma[,,1]<-cbind( c(1,0),c(0,1))
mu  <- cbind(c(4,-4),c(3.5,4),c( 0, 0))

delta <- cbind(c(3,3),c(1,5),c(-3,1))
dof    <- c(3,5,5)

distr="mst"

EmSkewMOD(p,g,distr,mu,sigma,dof,delta,nrand=10000)

}
\keyword{cluster}
\keyword{datasets}
