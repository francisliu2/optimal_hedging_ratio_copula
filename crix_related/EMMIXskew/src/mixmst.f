
c#-------------------------------------------------------------
c#
c#  Multivariate Skew T Mixture Models
c#
c#-------------------------------------------------------------




      subroutine estepmst(x,n,p,g,pro,mu,sigma,dof,delta,
     &tau,ev,elnv,ez1v,ez2v,sumtau,sumvt,sumzt,sumlnv,
     &loglik,error,method)

      implicit NONE

      integer n, p, g, error, method(g)

      double precision   loglik

      double precision   x(n,*), tau(n,*),dof(*),delta(p,*)

      double precision   pro(*),mu(p,*),sigma(p,p,*) 

      double precision   ev(n,*),elnv(n,*),sumtau(*),sumlnv(*)

      double precision   ez1v(n,*),ez2v(n,*),sumvt(*),sumzt(*)

c-----------------------------------------------------------------------------
      integer i,k
      double precision zero, one, two
      parameter (zero = 0.d0, one = 1.d0, two = 2.d0)
      double precision sum1,sum2,sum3,sum4
c-----------------------------------------------------------------------------

      error  = 0
      loglik = zero

      call denmst2(x,n,p,g,mu,sigma,dof,delta,
     &tau,ev,elnv,ez1v,ez2v,error,method)

      if(error .ne. 0) then
            error=2
            return
      endif

      call gettau(tau,pro,loglik,n,g,error)

      if(error .ne. 0) then
            error=3
            return
      endif


      do k=1,g

        sum1=zero
        sum2=zero
        sum3=zero
        sum4=zero

        do i=1,n
          sum1=sum1+tau(i,k)
          sum2=sum2+ev(i,k)*tau(i,k)
          sum3=sum3+ez2v(i,k)*tau(i,k)
          sum4=sum4+elnv(i,k)*tau(i,k)
        enddo

        sumtau(k) =sum1
        sumvt(k)  =sum2
        sumzt(k)  =sum3
        sumlnv(k) =sum4
        
	pro(k)=sumtau(k)/dble(n)

        if(sumtau(k) .lt. two) then
	   pro(k)=zero    
        endif

      end do

c-----------------------------------------------------------------------------
      return
      end


      subroutine mstepmst(y,n,p,g,ncov,
     &tau,ev,ez1v,ez2v,sumtau,sumvt,sumzt,
     &mu,sigma,delta)

c-----------------------------------------------------------------------------

      implicit none
      integer n,p,g,ncov

      double precision y(n,*), mu(p,*), sigma(p,p,*),delta(p,*)

      double precision tau(n,*),ev(n,*),ez1v(n,*),ez2v(n,*)

      double precision sumtau(*),sumvt(*),sumzt(*)

c---------------------------------------------------------------------
c    local variables
      double precision zero, one, two
      parameter(zero = 0.d0, one = 1.d0, two = 2.d0)

      double precision sum,tmp
      integer h,i,j,k

c---------------------------------------------------------------------



c   calculate the variances
      do 1000 h=1,g

        do 200 j=1,p
          do 200 k=j,p

	    sum=zero

            do i=1,n
      sum=sum+((y(i,j)-mu(j,h)  )*( y(i,k)-mu(k,h))*ev(i,h)
     &- delta(j,h)*( y(i,k)-mu(k,h))*ez1v(i,h)
     &- delta(k,h)*( y(i,j)-mu(j,h))*ez1v(i,h)
     &+ delta(j,h)*delta(k,h)*ez2v(i,h))*tau(i,h)
            enddo

            if(sumtau(h) .lt. two) then
              sigma(k,j,h)=zero
            else
              sigma(k,j,h)=sum/sumtau(h)
            endif
	   
            sigma(j,k,h)=sigma(k,j,h)

200     continue

1000  continue

      if(ncov .ne.3) then
         call getcov(sigma,sumtau,n,p,g,ncov)
      endif

c   calculate the means
      do 10 h=1,g
       do 10 j=1,p
         sum=zero
         tmp=zero

         do i=1,n
         sum=sum+( y(i,j)*ev(i,h)
     &   - delta(j,h)*ez1v(i,h))*tau(i,h)
         tmp=tmp+(y(i,j)-mu(j,h))*(ez1v(i,h)*tau(i,h))
	 end do

         if(sumtau(h)  .lt. two) then
           mu(j,h)    = zero
           delta(j,h) = zero
         else
           mu(j,h)   =sum/sumvt(h)
           delta(j,h)=tmp/sumzt(h)
         endif

10    continue

      return
      end
* EOF *
