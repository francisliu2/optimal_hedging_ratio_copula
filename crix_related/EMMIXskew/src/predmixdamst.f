c#-------------------------------------------------------------
c#
c#  Multivariate Skew T Mixture Models
c
c   Discriminant Analysis; prediction
c#
c#-------------------------------------------------------------

    
      
      
      subroutine predmixdamst(x,n,p,G,pro,mu,Sigma,dof,delta,
     &tau,error)

      implicit NONE

      integer            n, p, G, error

      double precision   x(n,*), tau(n,  *  ),dof(*),delta(p,*)

      double precision   mu(p,*), Sigma(p,p,*), pro(  *  )


c-----------------------------------------------------------------------------

      double precision        zero,loglik
      parameter              (zero = 0.d0)


c-----------------------------------------------------------------------------

      error  = 0

 

      call denmst(x,n,p,g,mu,sigma,dof,delta,tau,error)


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

      


