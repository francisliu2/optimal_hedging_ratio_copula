\name{ddmsn}
\alias{ddmsn}
\alias{rdmsn}
\title{The Multivariate Skew Normal Distribution}
\description{
Density and random generation for Multivariate Skew Normal distributions with mean vector \code{mean}, 
covariance matrix \code{cov}, and skew parameter vector \code{del}. 
}
\usage{
ddmsn(dat,n, p, mean, cov, del)
rdmsn(    n, p, mean, cov, del)
}
\arguments{
  \item{dat}{An n by p numeric matrix, the dataset}
  \item{n}{An integer, the number of observations}
  \item{p}{An integer, the dimension of data}
  \item{mean}{A length of p vector, the mean}
  \item{cov}{A p by p matrix, the covariance}
  \item{del}{A length of p vector, the skew parameter}
}
\value{
\code{ddmsn} gives the density values; \code{rdmsn} generates the random numbers 
}
\seealso{
\code{\link{rdemmix}},\code{\link{ddmvn}},\code{\link{ddmvt}},
\code{\link{ddmst}},\code{\link{rdmvn}},\code{\link{rdmvt}},
\code{\link{rdmst}}.
}
\examples{

n <- 100
p <- 2

mean <- rep(0,p)
cov  <- diag(p)
del<- c(0,1)

set.seed(3214)

y   <- rdmsn(  n,p,mean,cov,del)

den <- ddmsn(y,n,p,mean,cov,del)

}
\keyword{cluster}
\keyword{datasets}
