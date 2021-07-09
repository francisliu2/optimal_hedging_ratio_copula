      subroutine skew(y,n,p,g,tau,sumtau,mu,sigma,delta)
      implicit none
*******************************
      integer p,n,g
      double precision y(n,*),mu(p,*),delta(p,*)
      double precision tau(n,*),sumtau(*),sigma(p,p,*)

      
      integer i,j,h
      double precision zero, cut, two,five
      parameter(zero = 0.d0,cut=1.d-1,two=2.d0,five=5.d0)
      double precision sum


*******************************

      do 100 h=1,g
      do 100 j=1,p

        sum=zero

        if( sumtau(h) .gt. two) then

        do i=1,n
          sum=sum+(( y(i,j)-mu(j,h))*tau(i,h))**3
        enddo

        sum=sum/sumtau(h)/(sigma(j,j,h)**(3/2))

        if(abs(sum)  .le. cut) then
            sum= zero
        elseif(sum   .gt. cut) then
            sum= five
        else
            sum=-five
        endif

        endif

	delta(j,h)=sum

100   continue

      return
      end

*****************************************************

c  fun := dnorm(x)/pnorm(x)

      double precision function dnbypn(x)
      implicit none
      double precision x
      double precision mydnorm,mvphin
      external mydnorm,mvphin

      if(x .gt. -37.0) then
        dnbypn=mydnorm(x)/mvphin(x)
      else
        dnbypn = 37.
      endif

      return
      end



*****************************************************

      subroutine denmst(x,n,p,g,mu,sigma,dof,delta,den,error)

      implicit NONE

      integer            n, p, g, error


      double precision   x(n,*), dof(*),delta(p,*)

      double precision   mu(p,*), sigma(p,p,*)

      double precision   den(n,  *  )


*****************************************************

      double precision   sigmad(p,p),sigm(p,p)
      double precision   inv1(p,p),inv2(p,p)


      integer            i, j, k

      double precision   const, logdet, temp
      double precision   tmp,tmp2


      double precision   wd1(p)
      double precision   tmpdd,tempyy,tempyd
      double precision   wx(p), wd(p), wy(p)

*****************************************************

      double precision   zero, one, two
      parameter              (zero = 0.d0, one = 1.d0, two = 2.d0)

      double precision        lnsqrtpi
      parameter   (lnsqrtpi = 0.572364942924700087071713675677d0)

      double precision        lnsqrt2pi
      parameter   (lnsqrt2pi = 0.918938533204672741780329736406d0)

      external                ddot, mygammln,mvphit
      double precision        ddot, mygammln,mvphit

*****************************************************
      integer counter,save(p),ii

      error  = 0


*  start of loop k (1, g)


      do k = 1, g
*****************************************************


         do i=1,p
         do j=i,p
         sigm(i,j)=sigma(i,j,k)
         end do
         end do


        call inverse3(sigm,inv1,logdet,p,error,counter,save)

        if(error .ne. 0) then
                error = 11
                return
        endif

        if(counter .ge. 1) then
           do j=1,counter
              i=save(j)+1
              do ii=1,p
                sigm(ii,i)=zero
                sigm(i,ii)=zero
              enddo

*             delta(i,k)=zero
              sigm(i,i)=1e-4

           enddo
        endif


         do i=1,p
         do j=i,p
         sigmad(i,j)=sigm(i,j)+delta(i,k)*delta(j,k)
         end do
         end do

        call inverse3(sigmad,inv2,logdet,p,error,counter,save)
        if(error .ne. 0) then
                error = 22
                return
        endif



*****************************************************


        const = mygammln((dof(k)+dble(p))/two)-(log(dof(k))
     &  +two*lnsqrtpi)*dble(p) /two
     &  -mygammln(dof(k)/two)-log(logdet)/two

        call dcopy( p, delta(1,k), 1, wx, 1)
* /* y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y,  */
        call dgemv("N", p, p, one, inv2, p, wx, 1, zero, wd, 1);

        call dcopy( p, delta(1,k), 1, wx, 1)
* /* y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y,  */
        call dgemv("N", p, p, one, inv1, p, wx, 1, zero, wd1, 1);
        tmpdd = ddot( p, wd1, 1, wd1, 1)



*  start of loop i

        do i = 1, n
          call dcopy( p, x(i,1), n, wy, 1)
          call daxpy( p, (-one), mu(1,k), 1, wy, 1)
          call dcopy( p, wy, 1, wx, 1)

* /* y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y,  */
          call dgemv("N", p, p, one, inv2, p, wx, 1, zero, wy, 1);

          tempyy = ddot( p, wy, 1, wy, 1)
          tempyd = ddot( p, wd, 1, wy, 1)


          tmp = tempyd*sqrt((one+tmpdd)
     &    *(dble(p)+dof(k)) /(tempyy+dof(k)))



          tmp2=mvphit(tmp,dof(k)+dble(p))

          temp = -log(one+tempyy/dof(k))*(dof(k)+dble(p))/two
     &    +log(two*tmp2)

* log density f(y_i,k)

          den(i,k) = (const+temp)


        end do
*  end of loop i (1,n)



      enddo

*  end of loop k (1,g)

      return
      end



      subroutine denmsn(x,n,p,g,mu,sigma,delta,den,error)
*****************************************************

      implicit NONE
      integer            n, p, g, error
      double precision   x(n,*), den(n,*)
      double precision   mu(p,*),sigma(p,p,*),delta(p,*)

*****************************************************

      double precision   sigmad(p,p),wx(p),wd(p),wy(p),inv(p,p)
      double precision   const, logdet, temp
      double precision   tempyy,tempdd,tempyd
      double precision   sigm(p,p),check
      integer  i, j, k

      double precision   mvphin,  ddot
      external           mvphin,  ddot

*****************************************************

      double precision        zero, one, two
      parameter(zero = 0.d0, one = 1.d0, two = 2.d0)

      double precision        log2pi
      parameter(log2pi = 1.837877066409345d0)



*****************************************************
      integer counter,save(p),ii


      error  = 0

*****************************************************
*  start of k loop

      do k = 1, G

        do i=1,p
          do j=i,p
            sigm(i,j)=sigma(i,j,k)
          enddo
        enddo


        call inverse3(sigm,inv,logdet,p,error,counter,save)




        if(error .ne. 0) then
                error = 11
                return
        endif

        if(counter .ge. 1) then
           do j=1,counter
              i=save(j)+1
              do ii=1,p
                sigm(ii,i)=zero
                sigm(i,ii)=zero
              enddo
*              delta(i,k)=zero
              sigm(i,i)=1e-4
           enddo
        endif

        do i=1,p
          do j=i,p
            sigmad(i,j)=sigm(i,j)+delta(i,k)*delta(j,k)
          enddo
        enddo

        call inverse3(sigmad,inv,logdet,p,error,counter,save)

        if(error .ne. 0) then
        error=22
                return
        endif


*****************************************************

        const =dble(p)*log2pi/two + log(logdet)/two

        call dcopy( p, delta(1,k), 1, wx, 1)

* /* y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y,  */

        call dgemv("N", p, p, one, inv, p, wx, 1, zero, wd, 1);

        tempdd = ddot( p, wd, 1, wd, 1)


*  start of i loop
        do i = 1, n
          call dcopy( p, x(i,1), n, wx, 1)
          call daxpy( p, (-one), mu(1,k), 1, wx, 1)

* /* y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y,  */

          call dgemv("N", p, p, one, inv, p, wx, 1, zero, wy, 1);

          tempyy = ddot( p, wy, 1, wy, 1)
          tempyd = ddot( p, wd, 1, wy, 1)

c          temp = tempyy/two
c     &    -log(two*mvphin(tempyd/sqrt(one-tempdd)))

	  check = tempyd/sqrt(one-tempdd)

          if(check .lt. -10.0) check=-10.0
	   
          temp = tempyy/two-log(two*mvphin(check))


* log density f(y_i,k)

          den(i,k) = -(const+temp)


        end do
*  end of i loop (1,n)

      end do
*  end of k loop (1,g)
*****************************************************


       return
       end



      subroutine denmsn2(x,n,p,g,mu,sigma,delta,
     &tau,ev,vv,error)
c----------------------------------------------------------------

      implicit NONE
      integer            n, p, G, error
      double precision   x(n,*), tau(n,  *  ),delta(p,*)
      double precision   mu(p,*), Sigma(p,p,*)
      double precision   ev(n,*), vv(n,*)

c-----------------------------------------------------------------------------

      double precision   sigmad(p,p), wx(p),wd(p), wy(p),inv(p,p)
      double precision   const, logdet, temp
      double precision   tempyy,tempdd,tempyd
      double precision   value,sigm(p,p),check
      integer            i, j, k

      double precision   mvphin,  dnbypn,ddot
      external           mvphin,  dnbypn,ddot

c-----------------------------------------------------------------------------

      double precision        zero, one, two
      parameter              (zero = 0.d0, one = 1.d0, two = 2.d0)

      double precision        log2pi
      parameter              (log2pi = 1.837877066409345d0)



c-----------------------------------------------------------------------------
      integer counter,save(p),ii


      error  = 0

c-----------------------------------------------------------------------------
*  start of k loop

      do k = 1, G

        do i=1,p
          do j=i,p
            sigm(i,j)=sigma(i,j,k)
          enddo
        enddo


        call inverse3(sigm,inv,logdet,p,error,counter,save)


        if(error .ne. 0) then
                error = 11
                return
        endif

        if(counter .ge. 1) then
           do j=1,counter
              i=save(j)+1
              do ii=1,p
                sigm(ii,i)=zero
                sigm(i,ii)=zero
              enddo
*              delta(i,k)=zero
              sigm(i,i)=1e-4
           enddo
        endif

        do i=1,p
          do j=i,p
            sigmad(i,j)=sigm(i,j)+delta(i,k)*delta(j,k)
          enddo
        enddo

        call inverse3(sigmad,inv,logdet,p,error,counter,save)

        if(error .ne. 0) then
        error=22
                return
        endif


c-----------------------------------------------------------------------------

        const =dble(p)*log2pi/two + log(logdet)/two

        call dcopy( p, delta(1,k), 1, wx, 1)

        call dgemv("N", p, p, one, inv, p, wx, 1, zero, wd, 1);

        tempdd = ddot( p, wd, 1, wd, 1)


*----------------------------------
*  start of i loop
*----------------------------------

        do i = 1, n
          call dcopy( p, x(i,1), n, wx, 1)
          call daxpy( p, (-one), mu(1,k), 1, wx, 1)

* /* y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y,  */

          call dgemv("N", p, p, one, inv, p, wx, 1, zero, wy, 1)

          tempyy = ddot( p, wy, 1, wy, 1)
          tempyd = ddot( p, wd, 1, wy, 1)



          check = tempyd/sqrt(one-tempdd)

          if(check .lt. -10.0) check=-10.0
	   
          temp = tempyy/two-log(two*mvphin(check))

          tau(i,k) = -(const+temp)

          value=sqrt(one-tempdd)*dnbypn(check)

          ev(i,k)=tempyd+ value
          vv(i,k)=tempyd**2+(one-tempdd)+value*tempyd

        end do

*----------------------------------
*  end of i loop
*----------------------------------



*  tau(i,k) = log f(y_i;k)*P(y_i is in the band of h)


      end do


*  end of k loop
c-----------------------------------------------------------------------------



      return
      end


      subroutine denmst2(x,n,p,g,mu,sigma,dof,delta,
     &tau,ev,elogv,ez1v,ez2v,error,method)

      implicit NONE

      integer            n, p, g, error, method(g)


      double precision   x(n,*), dof(*),delta(p,*)

      double precision   mu(p,*), Sigma(p,p,*)

      double precision   tau(n,  *  ),ev(n,*), elogv(n,*)

      double precision   ez1v(n,*), ez2v(n,*)

c-----------------------------------------------------------------------------

      double precision   sigmad(p,p), wd(p), wy(p),sigm(p,p)
      double precision   inv1(p,p),inv2(p,p)


      integer                  i, j, k

      double precision        const, logdet, temp
      double precision        tempyy,tempyd


      double precision   wd1(p),wy1(p),value,value2,tmp2,tmp3
      double precision   tmpyy,tmpyd,tmpdd,tmp,wx(p)

c-----------------------------------------------------------------------------

      double precision        zero, one, two
      parameter              (zero = 0.d0, one = 1.d0, two = 2.d0)

      double precision        lnsqrtpi
      parameter   (lnsqrtpi = 0.572364942924700087071713675677d0)

      double precision        lnsqrt2pi
      parameter   (lnsqrt2pi = 0.918938533204672741780329736406d0)

      double precision        sqrt2pi
      parameter              (sqrt2pi=0.39894228040143d0)


      external                ddot, mygammln,mvphit,mydigamma
      double precision        ddot, mygammln,mvphit,mydigamma

c-----------------------------------------------------------------------------
      integer counter,save(p),ii

      error  = 0



*  start of loop k (g)


      do k = 1, G
c-----------------------------------------------------------------------------


         do i=1,p
         do j=i,p
         sigm(i,j)=sigma(i,j,k)
         end do
         end do


        call inverse3(sigm,inv1,logdet,p,error,counter,save)

        if(error .ne. 0) then
                error = 11
                return
        endif

        if(counter .ge. 1) then
           do j=1,counter
              i=save(j)+1
              do ii=1,p
                sigm(ii,i)=zero
                sigm(i,ii)=zero
              enddo
*              delta(i,k)=zero
              sigm(i,i)=1e-4
           enddo
        endif


         do i=1,p
         do j=i,p
         sigmad(i,j)=sigm(i,j)+delta(i,k)*delta(j,k)
         end do
         end do

        call inverse3(sigmad,inv2,logdet,p,error,counter,save)
        if(error .ne. 0) then
                error = 22
                return
        endif



c-----------------------------------------------------------------------------


        const = mygammln((dof(k)+dble(p))/two)-(log(dof(k))
     &  +two*lnsqrtpi)*dble(p) /two
     &  -mygammln(dof(k)/two)-log(logdet)/two

        call dcopy( p, delta(1,k), 1, wx, 1)
        call dgemv("N", p, p, one, inv2, p, wx, 1, zero, wd, 1);

        call dcopy( p, delta(1,k), 1, wx, 1)
        call dgemv("N", p, p, one, inv1, p, wx, 1, zero, wd1, 1);
        tmpdd = ddot( p, wd1, 1, wd1, 1)



*  start of loop i

        do i = 1, n
          call dcopy( p, x(i,1), n, wy, 1)
          call daxpy( p, (-one), mu(1,k), 1, wy, 1)
          call dcopy( p, wy, 1, wx, 1)

          call dgemv("N", p, p, one, inv2, p, wx, 1, zero, wy, 1);

          tempyy = ddot( p, wy, 1, wy, 1)
          tempyd = ddot( p, wd, 1, wy, 1)


          tmp = tempyd*sqrt((one+tmpdd)
     &    *(dble(p)+dof(k)) /(tempyy+dof(k)))


          tmp2=mvphit(tmp,dof(k)+dble(p))

          temp = -log(one+tempyy/dof(k))*(dof(k)+dble(p))/two
     &    +log(two*tmp2)


          tau(i,k) = (const+temp)

c-----------------------------------------------------------------------------
          call dgemv("N", p, p, one, inv1, p, wx, 1, zero, wy1, 1);

          tmpyy = ddot( p, wy1, 1, wy1, 1)
          tmpyd = ddot( p, wd1, 1, wy1, 1)

          value=tmp*sqrt(one+two/(dble(p)+dof(k)))

*  E(lamda|y)

          ev(i,k)=mvphit(value,(dof(k)+dble(p)+two))
     &    *(dof(k)+dble(p))/(dof(k)+tempyy)/tmp2

c-----------------------------------------------------------------------------


      tmp3=((dof(k)+tempyy)/(dof(k)+tmpyy))**((dof(k)+dble(p))/two)
     &          /sqrt( (dof(k)+tmpyy)/two )

          value2=exp(mygammln((dof(k)+dble(p)+one)/two)
     &     -mygammln((dof(k)+dble(p))/two))*tmp3/tmp2*sqrt2pi

*  sqrt2pi= 1/sqrt(2*pi)
c-----------------------------------------------------------------------------

      ez1v(i,k)=tmpyd*ev(i,k)/(one+tmpdd)+value2/sqrt(one+tmpdd)
      ez2v(i,k)=(tmpyd/(one+tmpdd))**2*ev(i,k)+one/(one+tmpdd)
     &+tmpyd*value2/(one+tmpdd)/sqrt(one+tmpdd)
c-----------------------------------------------------------------------------

*    three methods to estimate degrees of freedom (dof)
*    method one: approximate
*    method two: intergral
*    method three:  fixed



          tmp3=zero
          elogv(i,k)= zero

          if(method(k) .eq. 2) then
c          call intsum(tmp, tempyy, dof(k), tmp3, p, 30)
          endif

c-----------------------------------------------------------------------------

          if(method(k) .ne. 3) then

          elogv(i,k)=-log((tempyy+dof(k))/two)-
     &      (dof(k)+dble(p))/(dof(k)+tempyy)
     &      +mydigamma((dof(k)+dble(p))/two) +tmp3

          endif

c-----------------------------------------------------------------------------
c   the above elongv-ev is combined to elogv
c          elogv(i,k)=ev(i,k)-log((tempyy+dof(k))/two)-
c     &      (dof(k)+dble(p))/(dof(k)+tempyy)
c     &      +mydigamma((dof(k)+dble(p))/two)

c void intsum_(double *pux, double *pdist, double *pdof, double *ret, int *pp, int *pL)
c-----------------------------------------------------------------------------

        end do

*  end of loop i


      end do

*  end of loop k


      return
      end



* for joint clustering and alignment model
* Dec 18, 2008




      subroutine denmst3(x,n,p,g,pro,mu,sigma,dof,delta,
     &tau,ev,elogv,ez1v,ez2v,ewy,loglik,error,method)

      implicit NONE

      integer            n, p, g, error, method(*)


      double precision   x(n,*), tau(n,  *  ),dof(*),delta(p,*)

      double precision   mu(p,*), Sigma(p,p,*),pro(*)

      double precision   ev(n,*), elogv(n,*),loglik

      double precision   ez1v(n,*), ez2v(n,*),ewy(n,*)

c-----------------------------------------------------------------------------

      double precision   sigmad(p,p), wd(p), wy(p),sigm(p,p)
      double precision   inv1(p,p),inv2(p,p)


      integer                  i, j, k

      double precision        const, logdet, temp
      double precision        tempyy,tempyd,value3


      double precision   wd1(p),wy1(p),value,value2,tmp2,tmp3
      double precision   tmpyy,tmpyd,tmpdd,tmp,wx(p)

c-----------------------------------------------------------------------------

      double precision        zero, one, two
      parameter              (zero = 0.d0, one = 1.d0, two = 2.d0)

      double precision        lnsqrtpi
      parameter   (lnsqrtpi = 0.572364942924700087071713675677d0)

      double precision        lnsqrt2pi
      parameter   (lnsqrt2pi = 0.918938533204672741780329736406d0)

      double precision        sqrt2pi
      parameter              (sqrt2pi=0.39894228040143d0)


      external                ddot, mygammln,mvphit,mydigamma
      double precision        ddot, mygammln,mvphit,mydigamma

c-----------------------------------------------------------------------------
      integer counter,save(p),ii

      error  = 0



*  start of loop k (g)


      do k = 1, G
c-----------------------------------------------------------------------------


         do i=1,p
         do j=i,p
         sigm(i,j)=sigma(i,j,k)
         end do
         end do


        call inverse3(sigm,inv1,logdet,p,error,counter,save)

        if(error .ne. 0) then
                error = 11
                return
        endif

        if(counter .ge. 1) then
           do j=1,counter
              i=save(j)+1
              do ii=1,p
                sigm(ii,i)=zero
                sigm(i,ii)=zero
              enddo
*              delta(i,k)=zero
              sigm(i,i)=1e-4
           enddo
        endif


         do i=1,p
         do j=i,p
         sigmad(i,j)=sigm(i,j)+delta(i,k)*delta(j,k)
         end do
         end do

        call inverse3(sigmad,inv2,logdet,p,error,counter,save)
        if(error .ne. 0) then
                error = 22
                return
        endif



c-----------------------------------------------------------------------------


        const = mygammln((dof(k)+dble(p))/two)-(log(dof(k))
     &  +two*lnsqrtpi)*dble(p) /two
     &  -mygammln(dof(k)/two)-log(logdet)/two

        call dcopy( p, delta(1,k), 1, wx, 1)
        call dgemv("N", p, p, one, inv2, p, wx, 1, zero, wd, 1);

        call dcopy( p, delta(1,k), 1, wx, 1)
        call dgemv("N", p, p, one, inv1, p, wx, 1, zero, wd1, 1);
        tmpdd = ddot( p, wd1, 1, wd1, 1)



*  start of loop i

        do i = 1, n
          call dcopy( p, x(i,1), n, wy, 1)
          call daxpy( p, (-one), mu(1,k), 1, wy, 1)
          call dcopy( p, wy, 1, wx, 1)

          call dgemv("N", p, p, one, inv2, p, wx, 1, zero, wy, 1);

          tempyy = ddot( p, wy, 1, wy, 1)
          tempyd = ddot( p, wd, 1, wy, 1)


          tmp = tempyd*sqrt((one+tmpdd)
     &    *(dble(p)+dof(k)) /(tempyy+dof(k)))


          tmp2=mvphit(tmp,dof(k)+dble(p))

          temp = -log(one+tempyy/dof(k))*(dof(k)+dble(p))/two
     &    +log(two*tmp2)


          tau(i,k) = (const+temp)

c-----------------------------------------------------------------------------
          call dgemv("N", p, p, one, inv1, p, wx, 1, zero, wy1, 1);

          tmpyy = ddot( p, wy1, 1, wy1, 1)
          tmpyd = ddot( p, wd1, 1, wy1, 1)

            value=tmp*sqrt(one+two/(dble(p)+dof(k)))

*  E(lamda|y)

          ev(i,k)=mvphit(value,(dof(k)+dble(p)+two))
     &    *(dof(k)+dble(p))/(dof(k)+tempyy)/tmp2

c-----------------------------------------------------------------------------


c          tmp3=((dof(k)+tempyy)/(dof(k)+tmpyy))**((dof(k)+dble(p))/two)
c     &          /sqrt( (dof(k)+tmpyy)/two )

c          value2=exp(mygammln((dof(k)+dble(p)+one)/two)
c     &     -mygammln((dof(k)+dble(p))/two))*tmp3/tmp2*sqrt2pi


      tmp3=((dof(k)+tempyy)/(dof(k)+tmpyy))**((dof(k)+dble(p))/two)
     & /tmp2*sqrt2pi

      value2=(exp(mygammln((dof(k)+dble(p)+one)/two)
     &-mygammln((dof(k)+dble(p))/two))
     &*tmp3/sqrt((dof(k)+tmpyy)/two ))

                value3=(exp(mygammln((dof(k)+dble(p)-one)/two)
     &     -mygammln((dof(k)+dble(p))/two))
     &*tmp3*sqrt((dof(k)+tmpyy)/two))


*  note:  sqrt2pi= 1/sqrt(2*pi) !
c-----------------------------------------------------------------------------

          ez1v(i,k)=tmpyd/(one+tmpdd)*ev(i,k)+value2/sqrt(one+tmpdd)
          
	  ez2v(i,k)=(tmpyd/(one+tmpdd))**2*ev(i,k)+one/(one+tmpdd)
     &      +tmpyd*value2/(one+tmpdd)/sqrt(one+tmpdd)

          ewy(i,k)=tmpyd/(one+tmpdd)+value3/sqrt(one+tmpdd)


c-----------------------------------------------------------------------------

*    three methods to estimate degrees of freedom (dof)
*    method one:
*    method two:
*    method three:  fixed



          tmp3=zero
          elogv(i,k)= zero

          if(method(k) .eq. 2) then
c          call intsum(tmp, tempyy, dof(k), tmp3, p, 30)
          endif

c-----------------------------------------------------------------------------

          if(method(k) .ne. 3) then



          elogv(i,k)=-log((tempyy+dof(k))/two)-
     &      (dof(k)+dble(p))/(dof(k)+tempyy)
     &      +mydigamma((dof(k)+dble(p))/two) +tmp3

          endif

c-----------------------------------------------------------------------------
c   the above elongv-ev is combined to elogv
c          elogv(i,k)=ev(i,k)-log((tempyy+dof(k))/two)-
c     &      (dof(k)+dble(p))/(dof(k)+tempyy)
c     &      +mydigamma((dof(k)+dble(p))/two)

c void intsum_(double *pux, double *pdist, double *pdof, double *ret, int *pp, int *pL)
c-----------------------------------------------------------------------------

        end do

*  end of loop i



      end do

*  end of loop k


      loglik = zero

      call gettau(tau,pro,loglik,n,g,error)

      if(error .ne. 0) then
            error=23
      endif



      return
      end

