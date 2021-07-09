
c#-------------------------------------------------------------
c#
c#  Multivariate Skew T Mixture Models
c
c   Discriminant Analysis
c#
c#-------------------------------------------------------------

      subroutine emmvtda(y,n,p,g,ncov,
     &pro,mu,sigma,dof,tau,xuu,sumtau,sumxuu,sumxuuln,
     &ewy,ewz,ewyy,loglik,lk,clust,itmax,epsilon,error)
c---------------------------------------------------------------------
      implicit none

c   global variables
      integer n,p,g,error,itmax,ncov,clust(*)

      double precision pro(*), mu(p,*), sigma(p,p,*), dof(*)

      double precision y(n,*),tau(n,*),xuu(n,*)

      double precision sumtau(*),sumxuu(*),sumxuuln(*)

      double precision epsilon, loglik,lk(*)
      double precision ewy(p,*),ewz(p,*),ewyy(p,p,*)

c---------------------------------------------------------------------
c   local variables
      integer it,maxloop
      double precision zero, one, two
      parameter (zero = 0.d0, one = 1.d0, two = 2.d0)

c---------------------------------------------------------------------

      maxloop=20

      error=0

      loglik=zero

      call initmvt(y,n,p,g,ncov,pro,mu,sigma,dof, 
     &tau,xuu,sumtau,sumxuu,sumxuuln,
     &ewy,ewz,ewyy,loglik,clust,error,maxloop)
      
      if(error .ne. 0) then
            error=4+error
            return
      endif

c---------------------------------------------------------------------

c   start of EM algorithm loop

      do it=1,itmax

         lk(it)=zero
      
      enddo
      
      loglik=zero

      do 1000 it=1,itmax

c  E-Step

      call estepmvtda(y,n,p,g,pro,mu,sigma,dof,
     &tau,xuu,sumtau,sumxuu,sumxuuln,loglik,clust,error)

      if(error.ne.0) then
         return
      endif

      lk(it)=loglik

c  M-step
      call mstepmvt(y,n,p,g,ncov,tau,xuu,
     &sumtau,sumxuu,sumxuuln,mu,sigma,dof)


c  Check if converge
         if(it.ge.itmax) error=1
         if(it .le. min(20,itmax)) goto 1000
         if( ((abs(lk(it-10)-lk(it))) .lt. (abs(lk(it-10))* epsilon))
     &   .and. ((abs(lk(it-1)-lk(it))) .lt. (abs(lk(it-1))* epsilon)) )
     &   goto 5000

1000  continue

c   end of loop
c---------------------------------------------------------------------

5000  continue
      call scaestepmvt(y,n,p,g,tau,xuu,
     &mu,ewy,ewyy)


      return
      end


ccc    e-step


      subroutine estepmvtda(x,n,p,g,pro,mu,sigma,dof,
     &tau,xuu,sumtau,sumxuu,sumxuuln,loglik,clust,error)
      implicit NONE

      integer            n, p, g, error,clust(*)

      double precision   loglik

      double precision   x(n,*), tau(n,  *  ),dof(*)

      double precision   mu(p,*), Sigma(p,p,*), pro(  *  )

      double precision   xuu(n,*), sumxuu(*),sumtau(*),sumxuuln(*)

c-----------------------------------------------------------------------------

      integer                 i,  k

      double precision  temp,sum1,sum2,sum3 

c-----------------------------------------------------------------------------

      double precision        zero, one, two
      parameter              (zero = 0.d0, one = 1.d0, two = 2.d0)

      double precision   mydigamma
      external mydigamma
c-----------------------------------------------------------------------------
      error  = 0
      loglik=zero

      call denmvt2(x,n,p,g,mu,sigma,dof,tau,xuu,error)

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

      do 100 k=1,g
        
	sum1=zero
	sum2=zero
	sum3=zero

        do i=1,n

          tau(i,k)=zero

          if(clust(i)==k) tau(i,k)=one
          
	  sum1 = sum1+tau(i,k)
          sum2 = sum2+tau(i,k)*xuu(i,k)
          sum3 = sum3+tau(i,k)*(log(xuu(i,k))-xuu(i,k))

        enddo

	sumtau(k)=sum1
        sumxuu(k)=sum2

	temp = (dble(p) +dof(k))/two;
        sumxuuln(k) = sum3-(log(temp)-mydigamma(temp))*sumtau(k)

	pro(k)=sumtau(k)/dble(n)

        if(sumtau(k) .lt. two) then
	   pro(k)=zero    
        endif

100   continue
c-----------------------------------------------------------------------------
      return
      end




