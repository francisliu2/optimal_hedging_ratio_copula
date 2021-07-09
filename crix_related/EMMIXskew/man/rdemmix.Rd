\name{rdemmix}
\alias{rdemmix}
\alias{rdemmix2}
\alias{rdemmix3}
\title{Simulate Data Using Mixture Models}
\description{
Generate random number from specified mixture models, including univariate and multivariate 
Normal distribution, t-distribution, Skew Normal distribution, and Skew t-distribution. 
}
\usage{
rdemmix(nvect,p,g,distr,    mu,sigma,dof=NULL,delta=NULL)
rdemmix2(n,   p,g,distr,pro,mu,sigma,dof=NULL,delta=NULL)
rdemmix3(n,   p,g,distr,pro,mu,sigma,dof=NULL,delta=NULL)
}
\arguments{
  \item{nvect}{A vector of how many points in each cluster,c(n1,n2,..,ng) }
  \item{n}{The total number of points}
  \item{p}{Dimension of data}
  \item{g}{The number of clusters}
  \item{distr}{A three letter string indicating the distribution type}
  \item{pro}{A vector of mixing proportions, see Details.}
  \item{mu}{A numeric matrix with each column corresponding to the mean, see Details.}
  \item{sigma}{An array of dimension (p,p,g) with first two dimension corresponding covariance matrix of each component, see Details.}
  \item{dof}{A vector of degrees of freedom for each component, see Details.}
  \item{delta}{A p by g matrix with each column corresponding to a skew parameter vector, see Details.}
}
\details{
The distribution type, determined by the \code{distr} parameter, which may take any one of the following values:
"mvn" for a multivariate normal, "mvt" for a multivariate t-distribution, "msn" for a multivariate skew normal distribution and "mst" for a multivariate skew t-distribution.
\code{pro}, a numeric vector of the mixing proportion of each component; \code{mu}, a p by g matrix with each column as its corresponding mean; 
\code{sigma}, a three dimensional p by p by g array with its jth component matrix (p,p,j) as the covariance matrix for jth component of mixture models;
\code{dof}, a vector of degrees of freedom for each component; \code{delta}, a p by g matrix with its columns corresponding to skew parameter vectors.

}
\value{
both \code{rdemmix} and \code{rdemmix2} return an n by p numeric matrix of generated data; 

\code{rdemmix3} gives a list with components \code{data}, the generated data, and \code{cluster}, the clustering of data.
}
\references{ 
McLachlan G.J. and Krishnan T. (2008). The EM Algorithm and Extensions (2nd). New Jersay: Wiley.

McLachlan G.J. and Peel D. (2000). Finite Mixture Models. New York: Wiley.

}
\seealso{\code{\link{rdmvn}},\code{\link{rdmvt}},\code{\link{rdmsn}},
\code{\link{rdmst}}.
}
\examples{
#specify the dimension of data, and number of clusters
#the number of observations in each cluster
n1=300;n2=300;n3=400;
nn<-c(n1,n2,n3)

p=2
g=3



#specify the distribution
distr <- "mvn"

#specify mean and covariance matrix for each component

sigma<-array(0,c(2,2,3))
for(h in 2:3) sigma[,,h]<-diag(2)
sigma[,,1]<-cbind( c(1,-0.1),c(-0.1,1))
mu  <- cbind(c(4,-4),c(3.5,4),c( 0, 0))

#reset the random seed 
set.seed(111) 
#generate the dataset
dat <- rdemmix(nn,p,g,distr,    mu,sigma)



# alternatively one can use
pro   <- c(0.3,0.3,0.4)
n=1000
set.seed(111)
dat <- rdemmix2(n,p,g,distr,pro,mu,sigma)
plot(dat)

# and 

set.seed(111)
dobj <- rdemmix3(n,p,g,distr,pro,mu,sigma)
plot(dobj$data)


#other distributions such as "mvt","msn", and "mst".

#t-distributions

dof    <- c(3,5,5)
dat <- rdemmix2(n,p,g,"mvt",pro,mu,sigma,dof)
plot(dat)

#Skew Normal distribution
delta <- cbind(c(3,3),c(1,5),c(-3,1))
dat <- rdemmix2(n,p,g,"msn",pro,mu,sigma,delta=delta)
plot(dat)


#Skew t-distribution
dat <- rdemmix2(n,p,g,"mst",pro,mu,sigma,dof,delta)
plot(dat)



}
\keyword{cluster}
\keyword{datasets}
