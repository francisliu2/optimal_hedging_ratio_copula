\name{conplot}
\alias{conplot}
\alias{conplot2}
\alias{conplot3}
\alias{mypanel2}
\alias{mypanel3}
\alias{mypanel4}
\alias{panel.density}
\title{
Functions of Contours}
\description{
These functions are called by EmSkew.contours,EmSkew.filter and EmSkew.flow to plot the contours of (skew) mixture density after fitting to the data.
}
\usage{
conplot(x, y, pro, mu, sigma, dof, delta, distr, grid = 300, 
nrand = 6000, levels = seq(5, 95, by = 20), col = "white")
conplot2(x, y, pro, mu, sigma, dof, delta, distr, grid = 300, 
    nrand = 6000, levels = seq(5, 95, by = 20)) 
conplot3(x, y, pro, mu, sigma, dof, delta, modpts,distr, grid =300, 
    nrand = 10000, levels = seq(5, 95, by = 20)) 
mypanel2(x,y,...)
mypanel3(x,y,...)
mypanel4(x,y,...)
panel.density(x, col=1,...)
}
\arguments{
  \item{x}{
A vector of observations on variable x.}
  \item{y}{
A vector of observations on variable y.}
  \item{pro}{
A vector of mixing proportions in the (skew) mixture model.}
  \item{mu}{
A matrix with each column corresponding to the mean or location vector of one mixture component.
}
  \item{sigma}{
An array of covariance matrices for each component of the mixture distribution.
}
  \item{dof}{
A vector of degrees of freedom when "distr"ibution is "mvt" or "mst".
}
  \item{delta}{
A matrix with each column as skew parameter vector of one component when "distr"ibution is "msn" or "mst".
}
  \item{distr}{
A three letter string indicating component distribution, "mvn"=normal distribution, "mvt"=t-distribution,"msn"=skew normal distribution, "mst"=skew t-ditribution. 
}
  \item{modpts}{The mode points.}
  
  \item{grid}{
An integer for the number of grid points in one direction.
}
  \item{nrand}{
A large integer for the number of random numbers being drawn. 

}

  \item{levels}{
A vector of contour percentage levels for the plots. It should be in the range of 0 to 100.
}
  \item{col}{
The colour of contour lines.
}
  \item{...}{other }
  }
\details{
In most case, users do not call this function directly, instead they call the function emmix.flow.}

\seealso{
\code{\link{EmSkew.flow}}
}

