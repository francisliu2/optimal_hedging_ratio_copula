c#-------------------------------------------------------------
c#
c#  Multivariate Skew Normal Mixture Models
c
c   Discriminant Analysis
c#
c#-------------------------------------------------------------


      subroutine emmsnda(y,n,p,g,ncov,
     &pro,mu,sigma,delta,
     &tau,ev,vv,sumtau,sumev,ewy,ewz,ewyy,
     &loglik,lk,clust,itmax,epsilon,error)

      implicit none
c---------------------------------------------------------------------
c   global variables
      integer n,p,g,ncov,itmax,clust(*),error
      double precision y(n,*),loglik, lk(*), epsilon 
      double precision pro(*),mu(p,*), sigma(p,p,*),delta(p,*)
      double precision tau(n,*),ev(n,*), vv(n,*)
      double precision sumtau(*),sumev(*)
      double precision ewy(p,*),ewz(p,*),ewyy(p,p,*)

c---------------------------------------------------------------------
c   local variables

      integer it,maxloop

      double precision        zero, one, two
      parameter              (zero = 0.d0, one = 1.d0, two = 2.d0)

c---------------------------------------------------------------------
      maxloop=20

      error=0

      loglik=zero

      call initmsn(y,n,p,g,ncov,pro,mu,sigma,delta,
     &tau,ev,vv,sumtau,sumev,
     &ewy,ewz,ewyy,loglik,clust,error,maxloop)

      if(error .ne. 0) then
            error=4
            return
      endif

c-----------------------------------------------------------------------------

c   start of em algorithm loop

      do it=1,itmax
      
          lk(it)=zero

      end do

      loglik=zero

      do 1000 it=1,itmax
      

c  e-step
      call estepmsnda(y,n,p,g,pro,mu,sigma,delta,
     &tau,ev,vv,sumtau,sumev,loglik,clust,error)

      if(error.ne.0) then
        return
      endif

      lk(it)=loglik

c  m-step
      call mstepmsn(y,n,p,g,ncov,tau,ev,vv,sumtau,sumev,
     &mu,sigma,delta)



c  check if converge
        if(it.ge.itmax) error=1
        if(it .le. min(20,itmax)) goto 1000
        if( (abs(lk(it-10)-loglik) .lt. (abs(lk(it-10))*epsilon))
     &   .and.(abs(lk(it-1)-loglik) .lt. (abs(lk(it-1))*epsilon)))
     &    goto 5000

1000  continue

c   end of loop

5000  continue

      call scaestepmsn(y,n,p,g,tau,ev,vv,
     &mu,delta,ewy,ewz,ewyy)

      return
      end




ccc    e-step

c-----------------------------------------------------------------------------

      subroutine estepmsnda(x,n,p,g,pro,mu,sigma,delta,
     &tau,ev,vv,sumtau,sumev,loglik,clust,error)
c-----------------------------------------------------------------------------

      implicit none
      integer            n, p, g, error,clust(*)
      double precision   x(n,*), tau(n,*),ev(n,*), vv(n,*)
      double precision   pro(*),mu(p,*), sigma(p,p,*),delta(p,*)
      double precision   loglik,sumtau(*),sumev(*)

c-----------------------------------------------------------------------------

      integer            i,  h
c-----------------------------------------------------------------------------

      double precision        zero,one,two
      parameter              (zero = 0.d0,one=1.0d0,two=2.0d0)
      double precision sum,tmp

c-----------------------------------------------------------------------------


      error  = 0
      loglik = zero


      call denmsn2(x,n,p,g,mu,sigma,delta,
     &tau,ev,vv,error)

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


      do 10 h=1,g
        sum = zero
        tmp = zero
        
	do i=1,n
         tau(i,h)=zero
         if(clust(i) .eq. h) tau(i,h)=one

	 sum = sum + tau(i,h)
         tmp = tmp + vv(i,h)*tau(i,h)
        end do

	sumtau(h)=sum
        sumev(h) =tmp

	pro(h)=sumtau(h)/dble(n)

        if(sumtau(h) .lt. two) then
	   pro(h)=zero
        endif

10    continue

c-----------------------------------------------------------------------------

      return
      end




