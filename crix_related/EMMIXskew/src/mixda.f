
*******************************************************
*
* Discriminant Analysis (DA)
*
*******************************************************


      subroutine emskewda(y,n,p,g,ncov,dist,
     &pro,mu,sigma,dof,delta,tau,ev,elnv,ez1v,ez2v,
     &sumtau,sumvt,sumzt,sumlnv,ewy,ewz,ewyy, 
     &loglik,lk,clust,itmax,epsilon,error)

      implicit none
*---------------------------------------------------------------------
      integer n,p,g,ncov,dist
      integer clust(*),itmax,error

      double precision y(n,*),mu(p,*),sigma(p,p,*),dof(*)
      double precision delta(p,*),pro(*)
      double precision tau(n,*),ev(n,*),elnv(n,*)
      double precision ez1v(n,*),ez2v(n,*)

      double precision sumtau(*),sumzt(*)
      double precision sumlnv(*), sumvt(*)
      double precision ewy(p,*),ewz(p,*),ewyy(p,p,*)
      double precision loglik,lk(*),epsilon

*---------------------------------------------------------------------
      integer h
       error = 0

      do h=1,g

         sumtau(h)=0.0
         sumzt(h)=0.0
         sumlnv(h)=0.0
         sumvt(h)=0.0

      enddo


      if    (dist .eq. 1) then

        call emmvnda(y,n,p,g,ncov,pro,
     &mu,sigma,tau,sumtau,ewy,ewz,ewyy, 
     &loglik,lk,clust,itmax,epsilon,error)

      elseif(dist .eq. 2) then
        call emmvtda(y,n,p,g,ncov,pro,
     &mu,sigma,dof,
     &tau,ev,
     &sumtau,sumvt,sumlnv,
     &ewy,ewz,ewyy, 
     &loglik,lk,clust,itmax,epsilon,error)

      elseif(dist .eq. 3) then
        call emmsnda(y,n,p,g,ncov,pro,
     &mu,sigma,delta,
     &tau,ev,elnv,
     &sumtau,sumvt,
     &ewy,ewz,ewyy, 
     &loglik,lk,clust,itmax,epsilon,error)

      elseif(dist .eq. 4) then
        call emmstda(y,n,p,g,ncov,pro,
     &mu,sigma,dof,delta,
     &tau,ev,elnv,ez1v,ez2v,     
     &sumtau,sumvt,sumzt,sumlnv, 
     &ewy,ewz,ewyy, 
     &loglik,lk,clust,itmax,epsilon,error)

      endif

      return
      end


      subroutine emskewpred(x,n,p,g,dist,pro,mu,sigma,dof,delta,
     &tau,clust,error)

      implicit none
*-----------------------------------------------------------------------------
      integer            n, p, g, error,dist, clust(*)
      double precision   x(n,*), tau(n,  *  ),dof(*),delta(p,*)
      double precision   mu(p,*), sigma(p,p,*), pro(  *  )
      integer            j, h
*-----------------------------------------------------------------------------
      if(dist .lt. 3) then
         do j=1,p
           do h =1,g
            delta(j,h)=dble(0)
           enddo
         enddo
      endif

      error = 0


      if    (dist .eq. 1) then
         call predmixdamsn(x,n,p,g,pro,mu,sigma,    delta,tau,error)
      elseif(dist .eq. 2) then
         call predmixdamst(x,n,p,g,pro,mu,sigma,dof,delta,tau,error)
      elseif(dist .eq. 3) then
         call predmixdamsn(x,n,p,g,pro,mu,sigma,    delta,tau,error)
      elseif(dist .eq. 4) then
         call predmixdamst(x,n,p,g,pro,mu,sigma,dof,delta,tau,error)
      endif

      call tau2clust(tau, n, g, clust)


      return
      end

