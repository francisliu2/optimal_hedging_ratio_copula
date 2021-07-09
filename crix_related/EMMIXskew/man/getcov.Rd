\name{getcov}
\alias{getcov}
\title{
Covariance
}
\description{
Recalculate covariance for a given structure.
}
\usage{
getcov(msigma, sumtau, n, p, g, ncov)
}
\arguments{
  \item{msigma}{An array for the covariance matrices. }
  \item{sumtau}{A vector for the expected number of observations from each component.}
  \item{n}{An integer for the sample size.}
  \item{p}{An integer for the dimension.}
  \item{g}{An integer for the number of components.}
  \item{ncov}{An integer for type of covariance.}
}
\value{
The covariance array.}
\references{
McLachlan G.J. and Krishnan T. (2008). The EM Algorithm and Extensions (2nd). Hoboken, New Jersey: Wiley.

McLachlan G.J. and Peel D. (2000). Finite Mixture Models. New York: Wiley.

}
