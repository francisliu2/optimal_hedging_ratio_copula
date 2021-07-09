c#-------------------------------------------------------------
c#
c#  Multivariate Skew Normal Mixture Models
c#
c#-------------------------------------------------------------

ccc    e-step

      subroutine estepmsn(x,n,p,g,pro,mu,sigma,delta,
     &tau,ev,vv,sumtau,sumev,loglik,error)
c----------------------------------------------------------------

      implicit NONE
      integer            n, p, g, error
      double precision   x(n,*), loglik
      double precision   pro(*),mu(p,*),sigma(p,p,*),delta(p,*)
      double precision   tau(n,*),ev(n,*), vv(n,*),sumtau(*),sumev(*)

c-----------------------------------------------------------------------------
      double precision zero, one, two
      parameter(zero = 0.d0, one = 1.d0, two = 2.d0)
      double precision sum,tmp
      integer  i, k

     
c-----------------------------------------------------------------------------

      error  = 0
      loglik = zero

      call denmsn2(x,n,p,g,mu,sigma,delta,
     &tau,ev,vv,error)
      if(error .ne. 0) then
        error=2
        return
      endif

      call gettau(tau,pro,loglik,n,g,error)
      if(error .ne. 0) then
        error=3
        return
      endif

      do 10 k=1,g
        sum = zero
        tmp = zero
        
	do i=1,n
          sum = sum + tau(i,k)
          tmp = tmp + vv(i,k)*tau(i,k)
        end do

	sumtau(k)=sum
        sumev(k) =tmp

	pro(k)=zero

        if(sumtau(k) .ge. two) then
	   pro(k)    =sumtau(k)/dble(n)
        endif
10    continue

      return
      end

      subroutine mstepmsn(y,n,p,g,ncov,tau,ev,vv,
     &sumtau,sumev,mu,sigma,delta)

      implicit none

c-----------------------------------------------------------------------------
      integer n,p,g,ncov
      double precision y(n,*),mu(p,*),sigma(p,p,*),delta(p,*)
      double precision tau(n,*),ev(n,*),vv(n,*),sumtau(*),sumev(*)
c---------------------------------------------------------------------
      double precision sum,tmp
      integer h,i,j,k
      double precision zero, one, two
      parameter(zero = 0.d0, one = 1.d0, two = 2.d0)
c---------------------------------------------------------------------

c   calculate the means
      do 100 h=1,g
       do 100 j=1,p
         sum=zero
         tmp=zero

         do i=1,n
           sum=sum+(y(i,j)-delta(j,h)*ev(i,h) )*tau(i,h)
           tmp=tmp+(y(i,j)-mu(j,h))*ev(i,h)*tau(i,h)
         enddo

         if(sumtau(h)  .lt. two) then
            mu(j,h) =zero
            delta(j,h)=zero
         else
            mu(j,h)    = sum/sumtau(h)
            delta(j,h) = tmp/sumev(h)
         endif

100   continue


c   calculate the variances
      do 1000 h=1,g


          do 200 k=1,p
            do 200 j=1,k
	      
	      sum=zero

              do i=1,n

       sum = sum
     & +((y(i,j)-mu(j,h))*(y(i,k)-mu(k,h))
     & -(delta(j,h)*ev(i,h))*(y(i,k)-mu(k,h))
     & -(delta(k,h)*ev(i,h))*(y(i,j)-mu(j,h))
     & +(delta(j,h)*delta(k,h)*vv(i,h)))*tau(i,h)
              
	      enddo

          if(sumtau(h) .gt. two) then
            sigma(j,k,h)= sum/sumtau(h)
          else
            sigma(j,k,h)= zero
          endif

          sigma(k,j,h)=sigma(j,k,h)

200     continue

1000  continue

      call getcov(sigma,sumtau,n,p,g,ncov)


      return
      end



