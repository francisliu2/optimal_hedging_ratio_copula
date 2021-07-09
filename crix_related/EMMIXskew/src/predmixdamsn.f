c#-------------------------------------------------------------
c#
c#  Multivariate Skew Normal Mixture Models
c
c   Discriminant Analysis;  prediction
c#
c#-------------------------------------------------------------


      subroutine predmixdamsn(x,n,p,G,pro,mu,sigma,delta,
     &tau,error)

c-----------------------------------------------------------------------------

      implicit NONE
      integer            n, p, G, error
      double precision   x(n,*), tau(n,  *  ),delta(p,*)
      double precision   mu(p,*), Sigma(p,p,*), pro(  *  )

c-----------------------------------------------------------------------------

      double precision        zero, loglik
      parameter              (zero = 0.d0)


c-----------------------------------------------------------------------------


      error  = 0


      call denmsn(x,n,p,g,mu,sigma,delta,tau,error)


      if(error .ne. 0) then
      error=22
      return
      endif


      loglik = zero

      call gettau(tau,pro,loglik,n,g,error)

      if(error .ne. 0) then
      error=23
      return
      endif


c-----------------------------------------------------------------------------

      return
      end



