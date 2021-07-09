\name{ddmst}
\alias{ddmst}
\alias{rdmst}
\title{The Multivariate Skew t-distribution}
\description{
Density and random generation for Multivariate Skew t-distributions with mean vector \code{mean}, 
 covariance matrix \code{cov}, degrees of freedom \code{nu}, and skew parameter verctor \code{del}. 
}
\usage{
ddmst(dat,n, p, mean, cov, nu, del)
rdmst(    n, p, mean, cov, nu, del)
}
\arguments{
  \item{dat}{An n by p numeric matrix, the dataset}
  \item{n}{An integer, the number of observations}
  \item{p}{An integer, the dimension of data}
  \item{mean}{A length of p vector, the mean}
  \item{cov}{A p by p matrix, the covariance}
  \item{nu }{A positive number, the degrees of freedom}
  \item{del}{A length of p vector, the skew parameter}
}
\value{
\code{ddmst} gives the density values; \code{rdmst} generates the random numbers 
}
\seealso{
\code{\link{rdemmix}},\code{\link{ddmvn}},\code{\link{ddmvt}},
\code{\link{ddmsn}},\code{\link{rdmvn}},\code{\link{rdmvt}},
\code{\link{rdmsn}}.
}
\examples{

n <- 100
p <- 2

mean <- rep(0,p)
cov  <- diag(p)
nu <- 3
del <- c(0,1)

set.seed(3214)

y   <- rdmst(  n,p,mean,cov,nu,del)

den <- ddmst(y,n,p,mean,cov,nu,del)

}
\keyword{cluster}
\keyword{datasets}
