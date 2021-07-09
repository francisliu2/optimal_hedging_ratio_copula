\name{ddmvt}
\alias{ddmvt}
\alias{rdmvt}
\title{The Multivariate t-Distribution}
\description{
Density and random generation for Multivariate t-distributions with mean vector \code{mean}, 
covariance matrix \code{cov}, and degrees of freedom \code{nu}. 
}
\usage{
ddmvt(dat,n, p, mean, cov, nu)
rdmvt(    n, p, mean, cov, nu)
}
\arguments{
  \item{dat}{An n by p numeric matrix, the dataset}
  \item{n}{An integer, the number of observations}
  \item{p}{An integer, the dimension of data}
  \item{mean}{A length of p vector, the mean}
  \item{cov}{A p by p matrix, the covariance}
  \item{nu }{A positive number, the degrees of freedom}
}
\value{
\code{ddmvt} gives the density values; \code{rdmvt} generates the random numbers 
}
\seealso{
\code{\link{rdemmix}},\code{\link{ddmvn}},\code{\link{ddmsn}},
\code{\link{ddmst}},\code{\link{rdmvn}},\code{\link{rdmsn}},
\code{\link{rdmst}}.
}
\examples{
n <- 100
p <- 2

mean   <- rep(0,p)
cov    <- diag(p)
nu     <-  3

set.seed(3214)

x   <- rdmvt(    n,p,mean,cov,nu)

den <- ddmvt(x  ,n,p,mean,cov,nu)

}
\keyword{cluster}
\keyword{datasets}
