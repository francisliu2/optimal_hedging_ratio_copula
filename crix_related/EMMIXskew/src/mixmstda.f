
c#-------------------------------------------------------------
c#
c#  Multivariate Skew T Mixture Models
c
c   Discriminant Analysis
c#
c#-------------------------------------------------------------

      subroutine emmstda(y,n,p,g,ncov,
     &pro,mu,sigma,dof,delta,
     &tau,ev,elnv,ez1v,ez2v,
     &sumtau,sumvt,sumzt,sumlnv,ewy,ewz,ewyy, 
     &loglik,lk,clust,itmax,epsilon,error)

      implicit none
c---------------------------------------------------------------------
c   parameter declarations

      integer n,p,g,ncov,itmax,error,clust(*)

      double precision y(n,*),pro(*),mu(p,*),sigma(p,p,*)

      double precision dof(*),delta(p,*)

      double precision tau(n,*),ev(n,*),elnv(n,*)

      double precision ez1v(n,*),ez2v(n,*)

      double precision sumtau(*),sumvt(*),sumzt(*),sumlnv(*)

      double precision loglik,lk(*),epsilon
      double precision ewy(p,*),ewz(p,*),ewyy(p,p,*)
c---------------------------------------------------------------------
c   local variables

      integer it,maxloop,method(g)

      double precision zero,one,two,bx

c---------------------------------------------------------------------
      parameter  (zero=0.d0,one =1.d0,two=2.d0,bx=50.0d0)
      

c---------------------------------------------------------------------

      maxloop=20

      error=0

      loglik=zero

      call initmst(y,n,p,g,ncov,pro,mu,sigma,dof,delta, 
     &tau,ev,elnv,ez1v,ez2v,sumtau,sumvt,sumzt,sumlnv,
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


      do it=1,g
         
	 method(it)=1
      
      end do


      do 1000 it=1,itmax
      
c  E-Step
      call estepmstda(y,n,p,g,pro,mu,sigma,dof,delta,
     &tau,ev,elnv,ez1v,ez2v,
     &sumtau,sumvt,sumzt,sumlnv,
     &loglik,clust,error,method)

      if(error.ne.0) then
          return
      endif

      lk(it)=loglik

c  M-step
      call mstepmst(y,n,p,g,ncov,
     &tau,ev,ez1v,ez2v,sumtau,sumvt,sumzt,
     &mu,sigma,delta)

c  calculate the dof
      call getdof(n, g, sumtau, sumlnv, dof, bx)

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
      call scaestepmst(y,n,p,g,tau,ev,ez1v,ez2v,
     &mu,delta,ewy,ewz,ewyy)


      return
      end


      subroutine estepmstda(x,n,p,g,pro,mu,sigma,dof,delta,
     &tau,ev,elnv,ez1v,ez2v,sumtau,sumvt,sumzt,sumlnv,
     &loglik,clust,error,method)

      implicit NONE

      integer n,p,g,error,method(*),clust(*)

      double precision loglik

      double precision x(n,*), tau(n,*)

      double precision pro(*),mu(p,*),sigma(p,p,*),dof(*),delta(p,*)

      double precision ev(n,*),elnv(n,*),ez1v(n,*),ez2v(n,*)

      double precision sumtau(*),sumlnv(*),sumvt(*),sumzt(*)

c-----------------------------------------------------------------------------


      integer i, k
      double precision sum1,sum2,sum3,sum4
c-----------------------------------------------------------------------------

      double precision zero,one,two
      parameter (zero = 0.d0,one=1.0d0,two=2.0d0)
c-----------------------------------------------------------------------------
      error=0
      loglik=zero

      call denmst2(x,n,p,g,mu,sigma,dof,delta,
     &tau,ev,elnv,ez1v,ez2v,error,method)

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



        do k=1,g

        sum1=zero
        sum2=zero
        sum3=zero
        sum4=zero

        do i=1,n
          tau(i,k)=zero
          if(clust(i)==k) tau(i,k)=one
          sum1=sum1+tau(i,k)
          sum2=sum2+ez2v(i,k)*tau(i,k)
          sum3=sum3+ev(i,k)*tau(i,k)
          sum4=sum4+elnv(i,k)*tau(i,k)
        enddo

	sumtau(k)=sum1
        sumzt(k)=sum2
        sumvt(k)=sum3
        sumlnv(k)=sum4

	pro(k)=sumtau(k)/dble(n)

        if(sumtau(k) .lt. two) then
	   pro(k)=zero    
        endif

      end do

c-----------------------------------------------------------------------------
      return
      end





