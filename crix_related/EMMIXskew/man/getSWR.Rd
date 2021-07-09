\name{getSWR}
\alias{getSWR}
\alias{interdist}
\alias{intradist}
\alias{mahalonobis}
\title{Calculate modes}
\description{
Calculate the distances.
}


\usage{
getSWR(dat,g,sigma,clust, tau)
intradist(dat,g,sigma, clust, tau) 
interdist(dat,g,sigma, clust, tau) 
mahalonobis(p, g, mu, sigma) 
}
\arguments{
\item{dat}{The dataset, an n by p numeric matrix, where n is number of observations and p the dimension of data.}
\item{p}{The dimension of the data}
\item{g}{The number of components to be fit}
\item{mu }{A numeric matrix with each column corresponding to the mean}
\item{sigma}{An array of dimension (p,p,g) with first two dimensions corresponding covariance matrix of each component}
\item{clust}{ A vector of integers specifying the initial partitions of the data; the default is NULL.}	
\item{tau}{An n by g matrix of posterior probability for each data point}
}
\keyword{cluster}
\keyword{datasets}
