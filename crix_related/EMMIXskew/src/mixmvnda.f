c#-------------------------------------------------------------
c#
c#  Multivariate Normal Mixture Models
c
c   Discriminant Analysis
c#
c#-------------------------------------------------------------


      subroutine emmvnda(y,n,p,g,ncov,
     &pro,mu,sigma,tau,sumtau,ewy,ewz,ewyy,
     &loglik,lk,clust,itmax,epsilon,error)

      implicit none
c---------------------------------------------------------------------
c   global variable
      integer n,p,g,ncov,clust(*),itmax,error
      double precision y(n,*),pro(*), mu(p,*), sigma(p,p,*)

      double precision tau(n,*),sumtau(*),loglik,lk(*),epsilon
      double precision ewy(p,*),ewz(p,*),ewyy(p,p,*)

c---------------------------------------------------------------------
c   local variables

      integer it,maxloop

      double precision        zero, one
      parameter              (zero = 0.d0, one = 1.d0)

c---------------------------------------------------------------------
      maxloop=10

      error=0

      loglik=zero

      call initmvn(y,n,p,g,ncov,pro,mu,sigma,
     &tau,sumtau,ewy,ewz,ewyy,
     &loglik,clust,error,maxloop)
      
      if(error .ne. 0) then
            error=4+error
            return
      endif

c-----------------------------------------------------------------------------

      do it=1,itmax

         lk(it)=zero
      
      enddo

      do 1000 it=1,itmax
      
c  E-Step
     
      call  estepmvnda(y,n,p,g,pro,mu,sigma,
     &tau,sumtau,loglik,clust,error)
         
      if(error.ne.0) then
         return
      endif

      lk(it)=loglik

c  M-step
      call mstepmvn(y,n,p,g,ncov,tau,sumtau,mu,sigma)

c  Check if converge
         if(it.ge.itmax) error=1
         if(it .le. 10) goto 1000
         if( ((abs(lk(it-10)-lk(it))) .lt. (abs(lk(it-10))* epsilon))
     &   .and. ((abs(lk(it-1)-lk(it))) .lt. (abs(lk(it-1))* epsilon)) )
     &   goto 5000

1000  continue

c   end of loop
c---------------------------------------------------------------------

5000  continue

      call scaestepmvn(y,n,p,g,tau,mu,ewy,ewyy)

c-----------------------------------------------------------------------------

      return
      end


      subroutine estepmvnda(x,n,p,g,pro,mu,sigma,
     &tau,sumtau,loglik,clust,error)

      implicit NONE

      integer  n,p,g,clust(*),error

      double precision   loglik

      double precision   x(n,*),tau(n,*),sumtau(*)

      double precision   pro(*),mu(p,*), sigma(p,p,*)

c-----------------------------------------------------------------------------

      integer i,  k

      double precision  sum

c-----------------------------------------------------------------------------

      double precision zero, one, two
      parameter(zero = 0.d0, one = 1.d0, two = 2.d0)

c-----------------------------------------------------------------------------
      error  = 0
      loglik=zero

      call denmvn(x,n,p,g,mu,sigma,tau,error)

      if(error .ne. 0) then
       error=2
       return
      endif


c calculate the loglikelihood

      call gettau(tau,pro,loglik,n,g,error)

      if(error .ne. 0) then
            error=3
            return
      endif

c-----------------------------------------------------------------------------

      do k=1,g
        sum=zero

        do i=1,n
          tau(i,k)=zero
          if(clust(i)==k) tau(i,k)=one
          sum=sum+tau(i,k)
        enddo

        sumtau(k)=sum
        pro(k)=sumtau(k)/dble(n)

        if(sumtau(k) .lt. two) then
	   pro(k)=zero    
        endif

      end do


c-----------------------------------------------------------------------------
      return
      end
