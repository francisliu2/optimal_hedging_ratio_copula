\name{EmSkew}
\alias{EmSkew}
\title{The EM Algorithm and Skew Mixture Models}
\description{
As a main function, EmSkew fits the data into the specified multivariate mixture models via the EM Algorithm. Distributions (univariate and multivariate) 
available include Normal distribution, t-distribution, Skew Normal distribution, and Skew t-distribution. 
}
\usage{
EmSkew(dat, g, distr="mvn", ncov=3,clust=NULL,init=NULL,itmax=1000, 
epsilon=1e-6, nkmeans=0, nrandom=10,nhclust=FALSE,debug=TRUE,
initloop=20)
}
\arguments{
\item{dat}{The dataset, an n by p numeric matrix, where n is number of observations and p the dimension of data.}
\item{g}{The number of components of the mixture model}
\item{distr}{A three letter string indicating the type of distribution to be fitted, the default value is "mvn", the Normal distribution. See Details.}
\item{ncov}{A small integer indicating the type of covariance structure; the default value is 3. See Details.}
\item{clust}{ A vector of integers specifying the initial partitions of the data; the default is NULL.}	
\item{init}{A list containing the initial parameters for the mixture model. See details. The default value is NULL.}	
\item{itmax}{ A big integer specifying the maximum number of iterations to apply; the default value is 1000.}
\item{epsilon}{A small number used to stop the EM algorithm loop when the relative difference between log-likelihood at each iteration become sufficient small; the default value is 1e-6.}
\item{nkmeans}{An integer to specify the number of KMEANS partitions to be used to find the best initial values; the default value is 0.}
\item{nrandom}{An integer to specify the number of random partitions to be used to find the best initial values; the default value is 10.}
\item{nhclust}{A logical value to specify whether or not to use hierarchical cluster methods; the default is FALSE. If TRUE, the Complete Linkage method will be used.}
\item{debug}{A logical value, if it is TRUE, the output will be printed out; FALSE silent; the default value is TRUE.}
\item{initloop}{A integer specifying the number of initial loops when searching the best intial partitions.}
}
\details{
The distribution type, determined by the \code{distr} parameter, which may take any one of the following values:
"mvn" for a multivariate normal, "mvt" for a multivariate t-distribution, "msn" for a multivariate skew normal distribution and "mst" for a multivariate skew t-distribution.

The covariance matrix type, represented by the \code{ncov} parameter, may be any one of the following:
\code{ncov}=1 for a common variance, \code{ncov}=2 for a common diagonal variance, \code{ncov}=3 for a general variance, \code{ncov} =4 for a diagonal variance, \code{ncov}=5 for 
sigma(h)*I(p)(diagonal covariance with same identical diagonal element values).

The parameter \code{init} requires following elements: \code{pro}, a numeric vector of the mixing proportion of each component; \code{mu}, a p by g matrix with each column as its corresponding mean; 
\code{sigma}, a three dimensional p by p by g array with its jth component matrix (p,p,j) as the covariance matrix for jth component of mixture models;
\code{dof}, a vector of degrees of freedom for each component; \code{delta}, a p by g matrix with its columns corresponding to skew parameter vectors.

Since we treat the list of \code{pro},\code{mu},\code{sigma},\code{dof},and \code{delta} as a common 
structure of parameters for our mixture models, we need to include all of them in the initial parameter list 
\code{init} by default although in some cases it does not make sense, 
for example, \code{dof} and \code{delta} is not applicable to normal mixture model. But in most cases, the user only need give relevent paramters in the list.   

When the parameter list \code{init} is given, the program ignores both initial partition \code{clust} and automatic partition methods such as \code{nkmeans}; 
only when both \code{init} and \code{clust} are not available, the program uses automatic approaches such as k-Means partition method to find the best inital values.
All three automatic approaches are used to find the best initial partition and initial values if required.  

The return values include all potential parameters \code{pro},\code{mu},\code{sigma},\code{dof},and \code{delta}, 
but user should not use or interpret irrelevant information arbitrarily. For example, \code{dof} and \code{delta} for Normal mixture models.
}
\value{
  \item{error}{Error code, 0 = normal exit;  1 = did not converge within \code{itmax} iterations;   2 = failed to get the initial values; 3 = singularity}
  \item{aic}{Akaike Information Criterion (AIC) }
  \item{bic}{Bayes Information Criterion (BIC)}
  \item{ICL}{Integrated Completed Likelihood Criterion (ICL)}
  \item{pro}{A vector of mixing proportions.}
  \item{mu}{A numeric matrix with each column corresponding to the mean.}
  \item{sigma}{An array of dimension (p,p,g) with first two dimension corresponding covariance matrix of each component.}
  \item{dof}{A vector of degrees of freedom for each component, see Details.}
  \item{delta}{A p by g matrix with each column corresponding to a skew parameter vector.}
  \item{clust}{A vector of final partition}
  \item{loglik}{The log likelihood at convergence}
  \item{lk}{A vector of log likelihood at each EM iteration}
  \item{tau}{An n by g matrix of posterior probability for each data point}
}
\references{
Biernacki C. Celeux G., and Govaert G. (2000). Assessing a Mixture Model for Clustering with the integrated Completed Likelihood.  IEEE Transactions on Pattern Analysis and Machine Intelligence. 22(7). 719-725.

McLachlan G.J. and Krishnan T. (2008). The EM Algorithm and Extensions (2nd). New Jersay: Wiley.

McLachlan G.J. and Peel D. (2000). Finite Mixture Models. New York: Wiley.
}
\seealso{
\code{\link{initEmmix}},\code{\link{rdemmix}}.
}
\examples{
#define the dimension of dataset

n1=300;n2=300;n3=400;
nn<-c(n1,n2,n3)

p  <- 2
ng <- 3

#define the parameters
sigma<-array(0,c(2,2,3))
for(h in 2:3) sigma[,,h]<-diag(2)
sigma[,,1]<-cbind( c(1,0.2),c(0.2,1))
mu     <- cbind(c(4,-4),c(3.5,4),c( 0, 0))

#and other parameters if required for "mvt","msn","mst"
delta  <- cbind(c(3,3),c(1,5),c(-3,1))
dof    <- c(3,5,5)

pro   <- c(0.3,0.3,0.4)

distr="mvn"
ncov=3

# generate a data set

set.seed(111) #random seed is reset 

dat <- rdemmix(nn,p,ng,distr,mu,sigma)



# the following code can be used to get singular data (remarked off)
#	dat[1:300,2]<--4 
#	dat[300+1:300,1]<-2
##	dat[601:1000,1]<-0
##	dat[601:1000,2]<-0



#fit the data using KMEANS to get the initial partitions (10 trials)
obj <- EmSkew(dat,ng,distr,ncov,itmax=1000,epsilon=1e-5,nkmeans=10)


# alternatively, if we define initial values like 
initobj<-list()

initobj$pro  <- pro
initobj$mu   <- mu
initobj$sigma<- sigma


initobj$dof  <- dof
initobj$delta<- delta


# then we can fit the data from initial values
obj <- EmSkew(dat,ng,distr,ncov,init=initobj,itmax=1000,epsilon=1e-5)

# finally, if we know inital partition such as 
clust       <- rep(1:ng,nn)


# then we can fit the data from given initial partition
obj <- EmSkew(dat,ng,distr,ncov,clust=clust,itmax=1000,epsilon=1e-5)

# plot the 2D contours

colnames(dat)<- paste("x",1:p,sep='')

# dev.new()
EmSkew.flow(dat,obj)

}
\keyword{cluster}
\keyword{datasets}

