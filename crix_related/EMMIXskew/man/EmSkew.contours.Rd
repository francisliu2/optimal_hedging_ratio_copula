\name{EmSkew.contours}
\alias{EmSkew.contours}
\alias{EmSkew.flow}
\alias{EmSkew.filter}
\title{Scatter plot and Contours}
\description{
Contour of fitted mixture density cab be an important indicator of goodness-of-fit.
Together with the heatmaps, this function provides a general idea of how well the mixtue model fits to the data.
}
\usage{
EmSkew.flow(S, obj = NULL, distr="",diag.panel=TRUE,upper.panel="type2", 
lower.panel ="type3", levels=seq(5, 95,by=20),attop=FALSE,clust=NULL,
title="",path="",plot=TRUE)
EmSkew.contours(S, obj = NULL, clust = NULL,distr="",diag.panel=TRUE,
upper.panel="type2",lower.panel="type3",levels=seq(5,95,by=20),
plot=TRUE,title="",path='',attop =FALSE) 

EmSkew.filter(S, g=1,distr="mst",diag.panel=TRUE,upper.panel="type2", 
    lower.panel = "type3", levels = 90, attop = FALSE,
    title="",path="",plot=TRUE) 

}
\arguments{
\item{g}{The number of components to filter the data at FSC and SSC channels.}
\item{S}{A dataframe.}
\item{obj}{A list including the parameters of a (skew) mixture model.}
\item{distr}{A three letter string specifying the distribution.}
\item{diag.panel}{A logical value, plot density.}
\item{upper.panel}{A string for the panel to be used in upper triangle.}
\item{lower.panel}{A string for the panel to be used in lower triangle.}
\item{levels}{A vector of contour percentage levels for the plots. It should be in the range of 0 to 100.}
\item{attop}{A logical value indicating the direction of diagonal panels.}
\item{clust}{A vector with the cluster labels for each observation of the sample.}
\item{title}{The png file name.}
\item{path}{The path to the folder where plots are stored.}
\item{plot}{A logical variable, whether plot it in the windows.}
}

\details{R package geneplotter is required. }
\note{
The input list 'obj' must be assigned a component "distr".
}

\seealso{
\code{\link{conplot}}
}

