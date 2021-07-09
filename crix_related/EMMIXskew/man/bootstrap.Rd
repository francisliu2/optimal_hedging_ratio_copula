\name{bootstrap}
\alias{bootstrap}
\alias{bootstrap.noc}
\title{Bootstrap}
\description{
The standard error analysis and the bootstrap analysis of -2log(Lambda). 
}
\usage{
bootstrap(x,n,p,g,distr,ncov,popPAR,B=99,replace=TRUE,
itmax=1000,epsilon=1e-5)
bootstrap.noc(x,n,p,g1,g2,distr,ncov,B=99,replace=TRUE,
itmax=1000,epsilon=1e-5)
}
\arguments{
\item{n}{The number of observations}
\item{p}{The dimension of data}
\item{B}{The number of simulated data or replacements to be tried}
\item{x}{The dataset, an n by p numeric matrix, where n is number of observations and p the dimension of data.}
\item{g}{The number of components of the mixture model}
\item{g1,g2}{The range of the number of components of the mixture model}
\item{distr}{ A three letter string indicating the type of distribution to be fit. See Details.}
\item{ncov}{ A small integer indicating the type of covariance structure. See Details.}
\item{popPAR}{A list with components \code{pro}, a numeric vector of the mixing proportion of each component; \code{mu}, a p by g matrix with each column as its corresponding mean; 
\code{sigma}, a three dimensional p by p by g array with its jth component matrix (p,p,j) as the covariance matrix for jth component of mixture models;
\code{dof}, a vector of degrees of freedom for each component; \code{delta}, a p by g matrix with its columns corresponding to skew parameter vectors.}	
\item{replace}{A logical value indicating whether replacement to be used}
\item{itmax}{ A big integer specifying the maximum number of iterations to apply}
\item{epsilon}{A small number used to stop the EM algorithm loop when the relative difference between log-likelihood at each iteration become sufficient small.}
}
\details{
The distribution type, \code{distr}, is one of the following values:
"mvn" for a multivariate normal, "mvt" for a multivariate t-distribution, "msn" for a multivariate skew normal distribution and "mst" for a multivariate skew t-distribution.

The covariance matrix type, represented by the \code{ncov} parameter, may be any one of the following:
\code{ncov}=1 for a common variance, \code{ncov}=2 for a common diagonal variance, \code{ncov}=3 for a general variance, \code{ncov} =4 for a diagonal variance, \code{ncov}=5 for 
sigma(h)*I(p)(diagonal covariance with same identical diagonal element values).

When \code{replace} is FALSE, parametric bootstrap is used; otherwise replacement method is used.
}
\value{
\code{bootstrap} gives standard errors. 
\code{bootstrap.noc} returns a list with components \code{ret}, a B by (g2-g1) matrix of -2log(Lambda),
\code{vlk}, the loglikehood for each g in the range of g1 to g2, and \code{pvalue}, the p-values of g vs g+1. The results of fitting mixture models are stored in curent working directory, which can be used via command in R: obj <- dget("ReturnOf_g_???.ret").

}
\references{

McLachlan G.J. and Krishnan T. (2008). The EM Algorithm and Extensions (2nd). New Jersay: Wiley.

McLachlan G.J. and Peel D. (2000). Finite Mixture Models. New York: Wiley.

}
\seealso{\code{\link{EmSkew}},\code{\link{rdemmix}}}
\examples{

n1=300;n2=300;n3=400;
nn <-c(n1,n2,n3)
n <- sum(nn)
p <- 2
g <- 3


sigma<-array(0,c(p,p,g))
for(h in 1:3) sigma[,,h]<-diag(p)

mu  <- cbind(c(4,-4),c(3.5,4),c( 0, 0))

# for other distributions, 
#delta <- cbind(c(3,3),c(1,5),c(-3,1))
#dof    <- c(3,5,5)

distr="mvn"
ncov=3

#first we generate a data set
set.seed(111) #random seed is set 
dat <- rdemmix(nn,p,g,distr,mu,sigma,dof=NULL,delta=NULL)

#start from initial partition
clust<- rep(1:g,nn)
obj <- EmSkewfit1(dat,g,clust,distr,ncov,itmax=1000,epsilon=1e-5)

# do bootstrap (stadard error analysis)

\dontrun{
std <- bootstrap(dat,n,p,g,distr,ncov,obj,B=19,
replace=TRUE,itmax=1000,epsilon=1e-5)
print(std)

# do booststrap analysis of -2log(Lambda).

# alternatively data can be input as follow,
# dat <- read.table("mydata.txt",header=TRUE)
# p   <- ncol(dat)
# n   <- nrow(dat)

lad <- bootstrap.noc(dat,n,p,2,4,distr,ncov,B=19,
replace=FALSE,itmax=1000,epsilon=1e-5)
print(lad)

# return of g=2
obj2 <- dget("ReturnOf_g_2.ret")

# return of g=3
obj3 <- dget("ReturnOf_g_3.ret")

# return of g=4
obj4 <- dget("ReturnOf_g_4.ret")

#The posterior probability matrix for (g=3) is obtained by

tau <- obj3$tau


}

}
\keyword{cluster}
\keyword{datasets}

