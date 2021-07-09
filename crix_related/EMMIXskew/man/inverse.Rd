\name{inverse}
\alias{inverse}
\title{
Inverse of a covariance matrix 

}
\description{
Calculate the inverse of a covariance matrix.}
\usage{
inverse(sigma, p)
}
\arguments{
  \item{sigma}{The covariance matrix.}
  \item{p}{The dimension of the matrix.}
}
\value{
The inverse of the covariance matrix.}
\author{Kui Wang}
\note{ The covariance matrix may be singular. This is of use only for the clustering of the data.}
\examples{
a<- matrix(c(1,0,0,0),ncol=2)
a
inverse(a,2)

}
