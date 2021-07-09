\name{mvt.dof}
\alias{mvt.dof}
\title{Degrees of freedom}
\description{Calculate the degrees of freedom}
\usage{
mvt.dof(sumtau, sumlnv, lx = 2+1e-04, ux = 200)
}
\arguments{
  \item{sumtau}{A quantity.}
  \item{sumlnv}{A quantity.}
  \item{lx}{Lower limit.}
  \item{ux}{Upper limit.}
}
\details{
It is called by msmvt, jcamvt and jcamst. When there is no solution between lx and ux, ux will be returned.}
\value{degrees of freedom.}
\note{
It is called by msmvt, jcamst and jcamvt.}

