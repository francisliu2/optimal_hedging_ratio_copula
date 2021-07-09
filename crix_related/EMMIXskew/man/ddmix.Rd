\name{ddmix}
\alias{ddmix}
\title{Density Functions of Mixture Models}
\description{
Calculate the density of multivariate mixture models at data points for each component
}
\usage{
ddmix(  dat, n, p, g, distr, mu, sigma, dof=NULL, delta=NULL)
}
\arguments{
  \item{dat}{The dataset}
  \item{n}{The total number of points}
  \item{p}{Dimension of data}
  \item{g}{The number of clusters}
  \item{distr}{A three letter string indicating the distribution; "mvn" for normal, "mvt" for t distribution, "msn" for skew normal, and "mst" for skew t distribution. }
  \item{mu}{A numeric mean matrix with each column corresponding to the mean}
  \item{sigma }{An array of dimension (p,p,g) with first two dimensions corresponding covariance matrix of each component}
  \item{dof }{A vector of degrees of freedom for each component}
  \item{delta}{A matrix with each column as skew parameter vector}
}
\value{
ddmix gives an n by g matrix of logorithm of density at each data point for each component.
}
\references{
McLachlan G.J. and Krishnan T. (2008). The EM Algorithm and Extensions (2nd). New Jersay: Wiley.

McLachlan G.J. and Peel D. (2000). Finite Mixture Models. New York: Wiley.

}
\seealso{\code{\link{ddmvn}},\code{\link{ddmvt}},\code{\link{ddmsn}},\code{\link{ddmst}}. }
\examples{

p=2
g=3

#mixing propotion of each component
pro   <- c(0.3,0.3,0.4)

#specify mean and covariance matrix for each component

sigma<-array(0,c(2,2,3))
for(h in 2:3) sigma[,,h]<-diag(2)
sigma[,,1]<-cbind( c(1,0),c(0,.1))

mu  <- cbind(c(4,-4),c(3.5,4),c( 0, 0))

#specify other parameters for "mvt","msn","mst"

delta <- cbind(c(3,3),c(1,5),c(-3,1))
dof    <- c(3,5,5)

#specify the distribution
distr <- "mst"

y <- c(1,2)

n=1

#then the density value at y for the mixture model is 

ddmix(y, n, p, g, distr, mu, sigma, dof, delta)

}
\keyword{cluster}
