\name{ddmvn}
\alias{ddmvn}
\alias{rdmvn}
\title{The Multivariate Normal Distribution}
\description{
Density and random generation for Multivariate Normal distributions with mean vector \code{mean}, and covariance matrix \code{cov}. 
}
\usage{
ddmvn(dat,n, p, mean, cov)
rdmvn(    n, p, mean, cov)
}
\arguments{
  \item{dat}{An n by p numeric matrix, the dataset}
  \item{n}{An integer, the number of observations}
  \item{p}{An integer, the dimension of data}
  \item{mean}{A length of p vector, the mean}
  \item{cov}{A p by p matrix, the covariance}
}
\value{
\code{ddmvn} gives the density values; \code{rdmvn} generates the random numbers 
}
\seealso{
\code{\link{rdemmix}},\code{\link{ddmvt}},\code{\link{ddmsn}},
\code{\link{ddmst}},\code{\link{rdmvt}},\code{\link{rdmsn}},
\code{\link{rdmst}}.
}
\examples{
n <- 100
p <- 2

mean <- rep(0,p)
cov  <- diag(p)

set.seed(3214)

y <-   rdmvn(  n,p,mean,cov)

den <- ddmvn(y,n,p,mean,cov)

}
\keyword{cluster}
\keyword{datasets}
