


      subroutine scaestepmst(y,n,p,g,
     &tau,ev,ez1v,ez2v,
     &mu,delta,ewy,ewz,ewyy)

c-----------------------------------------------------------------------------

      implicit none
      integer n,p,g

      double precision y(n,*),mu(p,*),delta(p,*)

      double precision ewyy(p,p,*),ewy(p,*),ewz(p,*)

      double precision tau(n,*),ev(n,*),ez1v(n,*),ez2v(n,*)

c---------------------------------------------------------------------
c    local variables
      double precision sum,tmp
      integer h,i,j,k
      double precision        zero, one, two
      parameter              (zero = 0.d0, one = 1.d0, two = 2.d0)

c---------------------------------------------------------------------



c   calculate the variances
      do 1000 h=1,g


        do 200 j=1,p
          do 200 k=j,p
            sum=zero;
            do i=1,n
      sum=sum+((y(i,j)-mu(j,h)  )*( y(i,k)-mu(k,h))*ev(i,h)
     &- delta(j,h)*( y(i,k)-mu(k,h))*ez1v(i,h)
     &- delta(k,h)*( y(i,j)-mu(j,h))*ez1v(i,h)
     &+ delta(j,h)*delta(k,h)*ez2v(i,h))*tau(i,h)
            enddo

        ewyy(j,k,h)=sum

        ewyy(k,j,h)=sum

200     continue

1000  continue


c   calculate the means
      do 10 h=1,g
       do 10 j=1,p

         sum=zero
         tmp=zero

         do 20 i=1,n
         sum=sum+( y(i,j)*ev(i,h)
     & - delta(j,h)*ez1v(i,h) )*tau(i,h)


         tmp=tmp+(y(i,j)-mu(j,h))*(ez1v(i,h)*tau(i,h))

20    continue

         ewy(j,h)=sum
         ewz(j,h)=tmp

10    continue

      return
      end


      subroutine scaestepmsn(y,n,p,g,
     &tau,ev,vv,
     &mu,delta,ewy,ewz,ewyy)

      implicit none

c-----------------------------------------------------------------------------
      integer n,p,g
      double precision y(n,*), mu(p,*),delta(p,*) 
      double precision ewyy(p,p,*),ewy(p,*),ewz(p,*)
      double precision tau(n,*),ev(n,*),vv(n,*)
c---------------------------------------------------------------------
      double precision sum,tmp
      integer h,i,j,k
      double precision    zero, one, two
      parameter(zero = 0.d0, one = 1.d0, two = 2.d0)
c---------------------------------------------------------------------

c   calculate the means
      do h=1,g
       do j=1,p
         sum=zero
         tmp=zero

         do i=1,n
           sum=sum+(y(i,j)- delta(j,h)*ev(i,h) ) * tau(i,h)
           tmp=tmp+(y(i,j)-mu(j,h))*ev(i,h)*tau(i,h)
         enddo

         ewy(j,h) = sum
         ewz(j,h) = tmp

       enddo
      enddo


c   calculate the variances
      do 1000 h=1,g

          do 200 k=1,p
            do 200 j=1,k
       
	sum=zero
        
	do i=1,n

      sum=sum+(y(i,j)-mu(j,h))*( y(i,k)-mu(k,h))*tau(i,h)
     &-(delta(j,h)*ev(i,h) )*( y(i,k)-mu(k,h))*tau(i,h)
     &-(delta(k,h)*ev(i,h) )*( y(i,j)-mu(j,h))*tau(i,h)
     &+delta(j,h)*delta(k,h)*vv(i,h)*tau(i,h)
      
	enddo

          ewyy(j,k,h)=sum

          ewyy(k,j,h)=sum       

200     continue

1000  continue

      return
      end



