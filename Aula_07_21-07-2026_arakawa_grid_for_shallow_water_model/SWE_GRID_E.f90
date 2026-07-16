MODULE ModSWE
   IMPLICIT NONE
   PRIVATE
   REAL(KIND=8), PUBLIC   :: DeltaX      !metros
   REAL(KIND=8), PUBLIC   :: DeltaY      !metros
   REAL(KIND=8), PUBLIC   :: DeltaLat    !degree
   REAL(KIND=8), PUBLIC   :: DeltaLon    !degree
   REAL(KIND=8), PUBLIC   :: DeltaLambda !degree
   REAL(KIND=8), PUBLIC   :: DeltaPhi    !degree

   ! time
   ! start and end point for time
   ! number of time steps
   INTEGER,PUBLIC, PARAMETER :: nt = 3001
   REAL(KIND=8) , PARAMETER  :: tstart = 0.0, tend = 1000.0
   REAL(KIND=8) , PARAMETER  :: Lt = tend - tstart
   REAL(KIND=8), PUBLIC   ,parameter        :: DeltaT =1200 ! Lt/float(nt-1)  !seg.   
   REAL(KIND=8), PUBLIC   ,parameter        :: DeltaTOUT =3600 ! Lt/float(nt-1)  !seg.   
   REAL(KIND=8), PUBLIC   ,parameter        :: H0     =  1.0   !metros
   REAL(KIND=8), PUBLIC   ,parameter        :: A      =  1.0   !metros
   REAL(KIND=8), PUBLIC   ,parameter        :: r      =  0.0e-6 !1e-3   !1/86400/10 > Reyleigh dissipation for momentum (1/s)
   REAL(KIND=8), PUBLIC   ,parameter        :: vd     =  1e-3   !1/86400/10 > Reyleigh dissipation for momentum (1/s)
   REAL(KIND=8), PUBLIC   ,parameter        :: vis    = 5.e-3! 1.5e-10 !1.5e-5        !  viscosity
   REAL(KIND=8), PUBLIC   ,parameter        :: nu     = vis        !  viscosity

   REAL(KIND=8), PUBLIC   ,parameter        :: Beta     =  2.0e-11        !  
   REAL(KIND=8), PUBLIC   ,parameter        :: r_earth     =  6371000.0        !  meters

   ! start and end point for x axis
   REAL(KIND=8), PUBLIC   ,parameter        :: xLonStart = -180.0, xLonEnd = 180.0!degree
    
   ! start and end point for y axis
   REAL(KIND=8), PUBLIC   ,parameter        :: yLatStart = -90.0, yLatEnd = 90.0!degree
    
    real*8, parameter :: pi = 4.0*atan(1.0)

    ! rotation parameter, omega
   REAL(KIND=8)  , parameter :: g =  9.8065     ! metros/seg.^2
   REAL(KIND=8)  , parameter :: omega = 7.27e-5
   ! perturbation
   REAL(KIND=8)  , parameter :: perturb = 0.01
   REAL(KIND=8)  , parameter :: lambda0 =300.0
   REAL(KIND=8)  , parameter :: phi0    =-10.0
   REAL(KIND=8)  , parameter :: d1      = 10
   REAL(KIND=8)  , parameter :: d2      = 10



   REAL(KIND=8)  , PUBLIC, ALLOCATABLE           :: f(:,:) 
   INTEGER, PUBLIC :: Idim
   INTEGER, PUBLIC :: Jdim
   REAL(KIND=8)   , PUBLIC, ALLOCATABLE :: u(:,:) 
   REAL(KIND=8)   , PUBLIC, ALLOCATABLE :: v(:,:) 
   REAL(KIND=8)   , PUBLIC, ALLOCATABLE :: h(:,:) 
   REAL(KIND=8)   , PUBLIC, ALLOCATABLE :: hs(:,:) 
   REAL(KIND=8)   , PUBLIC, ALLOCATABLE :: xLon(:) 
   REAL(KIND=8)   , PUBLIC, ALLOCATABLE :: yLat(:) 
   REAL(KIND=8)   , PUBLIC, ALLOCATABLE :: x(:) 
   REAL(KIND=8)   , PUBLIC, ALLOCATABLE :: y(:) 
   REAL(KIND=8)   , PUBLIC, ALLOCATABLE :: xx(:,:) 
   REAL(KIND=8)   , PUBLIC, ALLOCATABLE :: yy(:,:) 
   REAL(KIND=8)   , PUBLIC, ALLOCATABLE :: lambda(:,:) 
   REAL(KIND=8)   , PUBLIC, ALLOCATABLE :: phi(:,:) 

   REAL(KIND=8)   , PUBLIC, ALLOCATABLE :: coordY(:,:) 

  PUBLIC :: DefineCI
  PUBLIC :: Solve,Solve_Forward_Beta_plane,Solve_Forward_f_plane,&
            Solve_Forward_Beta_plane_Sphera,&
	    Solve_Forward_Beta_plane_Grid_E

CONTAINS

   SUBROUTINE DefineCI()
      INTEGER :: i,j
      Idim=360   !(xLonEnd-xLonStart+DeltaLon)/DeltaLon
      Jdim=180   !ABS(yLatEnd-yLatStart+DeltaLat)/DeltaLat
      ALLOCATE (u(Idim,Jdim)) ; u=0.0
      ALLOCATE (v(Idim,Jdim)) ; v=0.0
      ALLOCATE (h(Idim,Jdim)) ; h=0.0
      ALLOCATE (hs(Idim,Jdim)) ; h=0.0
      ALLOCATE (f(Idim,Jdim)) ; f=0.0
      ALLOCATE (xLon(Idim)) ; xLon=0.0
      ALLOCATE (yLat(Jdim)) ; yLat=0.0
      ALLOCATE (x(Idim)) ; x=0.0
      ALLOCATE (y(Jdim)) ; y=0.0
      ALLOCATE (xx(Idim,Jdim)) ; xx=0.0
      ALLOCATE (yy(Idim,Jdim)) ; yy=0.0
      ALLOCATE (coordY(Idim,Jdim)) ; coordY=0.0
      ALLOCATE (lambda(Idim,Jdim)) ; lambda=0.0
      ALLOCATE (phi(Idim,Jdim)) ; phi=0.0

      DeltaLon =abs(xLonEnd-xLonStart)/float(Idim-1)
      DeltaLat =abs(yLatEnd-yLatStart)/float(Jdim-1)

      DeltaLambda =  (pi * DeltaLon/180.0)
      DeltaPhi    =  (pi * DeltaLat/180.0)

      DO i=1,Idim
         x(i) = xLonStart + DeltaLon*float(i-1)
      END DO  
       DO j=1,Jdim
         y(j) = yLatStart + DeltaLat*float(j-1)
      END DO  

      DeltaX= (x(2)- x(1))*100000.0
      DeltaY= (y(2)- y(1))*100000.0

      PRINT*,DeltaLon,DeltaLat,DeltaX,DeltaY,DeltaT

      !
      !  --   --  2          --   --       --       --        --   -- 
      ! |   ni  |         2 |   kd  |     |   lambda  |    2 |   kd  |
      ! |  ---- |    = cos  |  ---- |  + 4|  -------- | sin  |  ---- | 
      ! |   f   |           |   2   |     |      d    |      |   2   |
      !  --   --             --   --       --       --        --   --
      !
      !      2*pi
      ! k = -------- =
      !      lambda
      !
      !  k*d                  k*(pi * d/180.0)
      ! ------< 0.5 ===>     -------------------<0.5 =====> 0  < k*(pi * d/180.0) < pi
      !   pi                          pi
      !                ______ 
      !              \/ gH
      !lambda =     -----------
      !                  f
      !

      DO j=1,Jdim
         DO i=1,Idim
             coordY(i,j) = (y(j)*100000.0)
             lambda(i,j) =  (pi * x(i)/180.0)
             phi   (i,j) =  (pi * y(j)/180.0)
             !f(i,j) = 2.0*omega*sin(pi * 45.0/180.0)
             !f(i,j) =0.0
             f(i,j) = 2.0*omega*sin(pi*y(j)/((xLonEnd-xLonStart)))
             h(i,j) = H0 - perturb*exp (-((i - 0.5*Idim)/5.0)**2 - ((j - 0.5*Jdim)/5.0)**2)

             !h(i,j) = H0 - perturb*exp (-(( lambda0 - x(i))/d1)**2 - ((phi0 - y(j))/d2)**2)
         END DO
      END DO
      
   END SUBROUTINE DefineCI

   SUBROUTINE  Solve_Forward_Beta_plane_Sphera(TermEqMomU, TermEqMomV, TermEqConH, u, v, h)

      REAL(KIND=8), dimension(1:Idim, 1:Jdim), intent( in) ::  u,  v,  h
      REAL(KIND=8), dimension(1:Idim, 1:Jdim), intent(out) :: TermEqMomU, TermEqMomV, TermEqConH
      
      INTEGER :: xb,xc,xf,i,j,yb,yc,yf
      REAL(KIND=8)    :: udux,vduy,vbar,lambdabar,fcov,gdhx,ru,vis2dudx,vis2dudy
      REAL(KIND=8)    :: udvx,vdvy,ubar,phibar,fcou,gdhy,rv,vis2dvdx,vis2dvdy
      REAL(KIND=8)    :: hdudx,hdvdy,dhudx ,dhvdy,vis2dhdx,vis2dhdy
      REAL(KIND=8)    :: betau,betav
      DO j=1,Jdim
         CALL index(j,Jdim,yb,yc,yf)
         DO i=1,Idim
            CALL index(i,Idim,xb,xc,xf)
            !
            !                     
            ! d(u)         d(u)   
            ! -----  + u * ------   = 0
            ! dt           dx     
            !                     
            udux  = u(xc,yc)*((u(xf,yc) - u(xc,yc))/(r_earth*cos(phi(xc,yc))*DeltaLambda))
            !
            !
            ! d(u)          d(u) 
            ! -----  + v * -------   = 0
            ! dt            dy  
            !
            vbar  = 0.25*( v(xc,yf)*cos(phi(xc,yf)) + v(xb,yf)*cos(phi(xb,yf)) +   v(xb,yc)*cos(phi(xb,yc))+  v(xc,yc)*cos(phi(xc,yc)) )

            vduy  = vbar*((u(xc,yf) - u(xc,yc))/(r_earth*DeltaPhi))
            !
            !
            ! d(u)
            ! -----   - f*v   = 0
            ! dt
            !
            fcov  = - f(xc,yc)*vbar 
            !
            !
            ! d(u)
            ! -----   - beta*v  =0
            ! dt
            !
            betav = -Beta*coordY(xc,yc)*vbar
            !
            ! d(u)       d(Eta)  
            ! -----   +g -------- 
            ! dt          dx
            !
            !
            gdhx  = g*((h(xc,yc) - h (xb,yc))/(r_earth*cos(phi(xc,yc))*DeltaLambda))
            !
            !       
            !       
            ! + b*u 
            !       
            !       
            ru    =  r*u(xc,yc)
            !
            !                  -         -
            ! d(u)            |d(d(u))    |
            ! -----  - Neta * |--------   | = 0
            ! dt              |dxdx       |
            !                  -         -

            vis2dudx= - vis*((u(xf,yc) - 2.0*u(xc,yc) + u(xb,yc))/((r_earth*cos(phi(xc,yc))*DeltaLambda)**2))
            !
            !                  -       -
            ! d(u)             |d(d(u)) |
            ! -----   - Neta * |--------| = 0
            ! dt               |dydy    |
            !                  -       -
            vis2dudy= - vis*((u(xc,yf) - 2.0*u(xc,yc) + u(xc,yb))/((r_earth*DeltaPhi)**2)) 
            !
            !                    
            ! d(u)         d(u)  
            ! -----  + u * ------ = 0
            ! dt           dx    
            !                    
            !TermEqMomU(i,j) = udux +  vduy + fcov + gdhx + ru + vis2dudx + vis2dudy
            TermEqMomU(i,j) = -(udux +  vduy + betav + gdhx + ru + vis2dudx + vis2dudy)
            !
            ! d(v)         d(v)   
            ! -----  + u * ------- =0
            ! dt            dx    
            !
            !
            ubar=0.25*(u(xf,yc) + u(xc,yc) + u(xc,yb) + u(xf,yb) )
            udvx = ubar * ((v(xf,yc)*cos(phi(xf,yc)) - v(xc,yc)*cos(phi(xc,yc)))/(r_earth*cos(phi(xc,yc))*DeltaLambda))
            !
            !
            ! d(v)         d(v) 
            ! -----  + v * -------   =0
            ! dt           dy  
            !
            vdvy = v(xc,yc)*cos(phi(xc,yc)) * ((v(xc,yf)*cos(phi(xc,yf)) - v(xc,yc)*cos(phi(xc,yc)))/(r_earth*DeltaPhi))
            !
            !-
            ! d(v)
            ! -----   + f*u  =0
            ! dt
            fcou = f(xc,yc) * ubar
            !
            ! d(v)
            ! -----    + beta*u  =0
            ! dt
            !
            betau = Beta*coordY(xc,yc) *  ubar
            !
            !
            ! d(v)        d(Eta)  
            ! -----   +g -------- =0
            ! dt          dy
            !
            gdhy = g * ((h(xc,yc) - h(xc,yb))/(r_earth*DeltaPhi))
            !
            !
            ! d(v)
            ! -----  + b*v =0
            ! dt
            !
            rv = r * v(xc,yc)*cos(phi(xc,yc))
            !                       -       -
            ! d(v)                 |d(d(v))  |
            ! -----       - Neta * |-------- | =0
            ! dt                   |dxdx     |
            !                       -       -
            
            vis2dvdx = -vis*((v(xf,yc)*cos(phi(xf,yc)) - 2.0*v(xc,yc)*cos(phi(xc,yc)) + v(xb,yc)*cos(phi(xb,yc)))/((r_earth*cos(phi(xc,yc))*DeltaLambda)**2) ) 
            !                  -             -
            ! d(v)            |    d(d(v))   |
            ! -----  - Neta * | ------------ | =0
            ! dt              |    dydy      |
            !                  -            -
            
            vis2dvdy = -vis* ((v(xc,yf)*cos(phi(xc,yf)) -2.0*v(xc,yc)*cos(phi(xc,yc)) + v(xc,yb)*cos(phi(xc,yb)))/((r_earth*DeltaPhi)**2) )
            !                                                                          -                      -
            ! d(v)         d(v)         d(v)                 d(Eta)                   |d(d(v))       d(d(v))   |
            ! -----  + u * ------ + v * -------  + beta*u  +g -------- + b*v - Neta * |-------- + ------------ | =0
            ! dt           dx            dy                   dy                      |dxdx          dydy      |
            !                                                                          -                      -
            
            !TermEqMomV(i,j) = udvx + vdvy + fcou + gdhy + rv + vis2dvdx + vis2dvdy
            
            TermEqMomV(i,j) = -(udvx + vdvy + betau + gdhy + rv + vis2dvdx + vis2dvdy)
            !										    	      - 		     -
            ! d(Eta)           d(u)              d(v)               d(Eta)         d(Eta)   	     |d(d(h))	    d(d(h))   |
            ! -----  + Eta * --------- + Eta * -----------  + u * --------- + v * --------  - Neta * |-------- + ------------ | =0
            ! dt                dx                dy                dx               dy     	     |dxdx	    dydy      |
            !										    	      - 		     -

            hdudx = H0 * ((u(xf,yc)  - u(xc,yc))/(r_earth*cos(phi(xc,yc))*DeltaLambda))
            
            !
            ! d(Eta)           d(u)    
            ! -----  + Eta * ---------  =0
            ! dt                dx     
            !

            hdvdy = H0 * ((v(xc,yf)*cos(phi(xc,yf))  - v(xc,yc)*cos(phi(xc,yc)))/(r_earth*DeltaPhi))

            !
            ! d(Eta)           d(v)
            ! -----  + Eta * -----------
            ! dt               dy
            !

            dhudx = ((u(xf,yc)+u(xc,yc))/2.0)*(h(xf,yc)  - h(xc,yc))/(r_earth*cos(phi(xc,yc))*DeltaLambda)
            
            !
            ! d(Eta)        d(Eta)
            ! -----  + u * ---------  =0
            ! dt             dx 
            !

            dhvdy =  ((v(xc,yf)*cos(phi(xc,yf)) + v(xc,yc)*cos(phi(xc,yc)))/2.0) * (h(xc,yf)  - h(xc,yc))/(r_earth*DeltaPhi)
            !
            ! d(Eta)          d(Eta)
            ! -----   + v * ----------- =0
            ! dt              dy    
            !
            !                       -       -
            ! d(h)                 |d(d(h))  |
            ! -----       - Neta * |-------- | =0
            ! dt                   |dxdx     |
            !                       -       -
            
            vis2dhdx = -nu*(h(xf,yc) - 2.0*h(xc,yc) + h(xb,yc))/((r_earth*cos(phi(xc,yc))*DeltaLambda)**2)  
            !                  -             -
            ! d(h)            |    d(d(h))   |
            ! -----  - Neta * | ------------ | =0
            ! dt              |    dydy      |
            !                  -            -
            
            vis2dhdy = -nu* (h(xc,yf) -2.0*h(xc,yc) + h(xc,yb))/((r_earth*DeltaPhi)**2) 
            !										    	      - 		     -
            ! d(Eta)           d(u)              d(v)               d(Eta)         d(Eta)   	     |d(d(h))	    d(d(h))   |
            ! -----  + Eta * --------- + Eta * -----------  + u * --------- + v * --------  - Neta * |-------- + ------------ | =0
            ! dt                dx                dy                dx               dy     	     |dxdx	    dydy      |
            !										    	      - 		     -
            

            TermEqConH(i,j) = -(hdudx + hdvdy + dhudx + dhvdy + vis2dhdx +vis2dhdy)
            
         END DO
      END DO
   END SUBROUTINE Solve_Forward_Beta_plane_Sphera






   
   

   SUBROUTINE  Solve_Forward_Beta_plane_Grid_E(TermEqMomU, TermEqMomV, TermEqConH, u, v, h)

      REAL(KIND=8), dimension(1:Idim, 1:Jdim), intent( in) ::  u,  v,  h
      REAL(KIND=8), dimension(1:Idim, 1:Jdim), intent(out) :: TermEqMomU, TermEqMomV, TermEqConH
      
      INTEGER :: xb,xc,xf,i,j,yb,yc,yf
      REAL(KIND=8)    :: udux,vduy,vbar,fcov,gdhx,ru,vis2dudx,vis2dudy,	vbar2, hbar  ,vis2dhdx
      REAL(KIND=8)    :: udvx,vdvy,ubar,fcou,gdhy,rv,vis2dvdx,vis2dvdy,	ubar2, hbar2, vis2dhdy
      REAL(KIND=8)    :: hdudx,hdvdy,dhudx ,dhvdy
      REAL(KIND=8)    :: betau,betav
      DO j=1,Jdim
         CALL index(j,Jdim,yb,yc,yf)
         DO i=1,Idim
            CALL index(i,Idim,xb,xc,xf)
            !
            !                     
            ! d(u)         d(u)   
            ! -----  + u * ------   = 0
            ! dt           dx     
            !                     
            udux  = u(xc,yc)*((u(xf,yc) - u(xb,yc))/(2*DeltaX))
            !
            !
            ! d(u)          d(u) 
            ! -----  + v * -------   = 0
            ! dt            dy  
            !
            vduy  = v(xc,yc) *((u(xc,yf) - u(xc,yb))/(2*DeltaY))
            !
            !
            ! d(u)
            ! -----   - f*v   = 0
            ! dt
            !
            fcov  = - f(xc,yc)*v(xc,yc) 
            !
            ! d(u)
            ! -----   - beta*v  =0
            ! dt
            !
	    !
	    betav = -Beta*coordY(xc,yc)*v(xc,yc)
            !
            ! d(u)       d(Eta)  
            ! -----   +g -------- 
            ! dt          dx
            !
            !
	    hbar =0.25*(h(xc,yf)+h(xb,yf)+h(xb,yc)+h(xc,yc))
	    hbar2=0.25*(h(xf,yf)+h(xc,yf)+h(xf,yc)+h(xc,yc))
            gdhx  = g* ( ( hbar2 - hbar)/(DeltaX))
            !
            !
            !       
            !       
            ! + b*u 
            !       
            !       
            ru    =  r*u(xc,yc)
            !
            !                  -         -
            ! d(u)            |d(d(u))    |
            ! -----  - Neta * |--------   | = 0
            ! dt              |dxdx       |
            !                  -         -
            vis2dudx= - vis*((u(xf,yc) - 2.0*u(xc,yc) + u(xb,yc))/(DeltaX*DeltaX))
            !
            !                  -       -
            ! d(u)             |d(d(u)) |
            ! -----   - Neta * |--------| = 0
            ! dt               |dxdx    |
            !                  -       -
            vis2dudy= - vis*((u(xc,yf) - 2.0*u(xc,yc) + u(xc,yb))/(DeltaY*DeltaY)) 
            !
            !                                                                          -                      -
            ! d(u)         d(u)         d(u)                 d(Eta)                   |d(d(u))       d(d(u))   |
            ! -----  + u * ------ + v * -------  - beta*v  +g -------- + b*u - Neta * |-------- + ------------ | = 0
            ! dt           dx            dy                   dx                      |dxdx          dydy      |
            !                                                                          -                      -
            !TermEqMomU(i,j) = udux +  vduy + fcov + gdhx + ru + vis2dudx + vis2dudy

            TermEqMomU(i,j) = -1*(udux + vduy + betav + gdhx + ru + vis2dudx + vis2dudy)
            !
            !
            ! d(v)         d(v) 
            ! -----  + u * -------   =0
            ! dt           dx  
            !
            !
            udvx = u(xc,yc) * ((v(xf,yc) - v(xb,yc))/(2*DeltaX))
            !
            !
            ! d(v)         d(v) 
            ! -----  + v * -------   =0
            ! dt           dy  
            !
            !
            vdvy = v(xc,yc) * ((v(xc,yf) - v(xc,yb))/(2*DeltaY))
            !
            !
            ! d(v)
            ! -----   + f*u  =0
            ! dt
            !
            fcou = f(xc,yc) * u(xc,yc)
            !
            ! d(v)
            ! -----    + beta*u  =0
            ! dt
            !
            betau = Beta*coordY(xc,yc) *  u(xc,yc)
            !
            !
            ! d(v)        d(Eta)  
            ! -----   +g -------- =0
            ! dt          dy
            !
            !
            gdhy = g * ((   h(xc,yf) - h(xc,yc))/(DeltaY ))
            !
            !
            ! d(v)
            ! -----  + b*v =0
            ! dt
            !
            !
            rv = r * v(xc,yc)
            !                       -       -
            ! d(v)                 |d(d(v))  |
            ! -----       - Neta * |-------- | =0
            ! dt                   |dxdx     |
            !                       -       -
            vis2dvdx = -vis*((v(xf,yc) - 2.0*v(xc,yc) + v(xb,yc))/(DeltaX*DeltaX) ) 
            !                  -             -
            ! d(v)            |    d(d(v))   |
            ! -----  - Neta * | ------------ | =0
            ! dt              |    dydy      |
            !                  -            -
            vis2dvdy = -vis* ((v(xc,yf) -2.0*v(xc,yc) + v(xc,yb))/(DeltaY*DeltaY) )
            !                                                                          -                      -
            ! d(v)         d(v)         d(v)                 d(Eta)                   |d(d(v))       d(d(v))   |
            ! -----  + u * ------ + v * -------  + beta*u  +g -------- + b*v - Neta * |-------- + ------------ | =0
            ! dt           dx            dy                   dy                      |dxdx          dydy      |
            !                                                                          -                      -
            
            !TermEqMomV(i,j) = udvx + vdvy + fcou + gdhy + rv + vis2dvdx + vis2dvdy
            
            TermEqMomV(i,j) = -(udvx + vdvy + betau + gdhy + rv + vis2dvdx + vis2dvdy)
            !
            ! d(Eta)           d(u)    
            ! -----  + Eta * ---------  =0
            ! dt                dx     
            !
            ubar2 = 0.25*(u(xf,yc) + u(xc,yc) + u(xc,yb) + u(xf,yb))
            ubar  = 0.25*(u(xb,yc) + u(xc,yc) + u(xb,yb) + u(xc,yb))
            hdudx = H0 * (( ubar2  - ubar)/(DeltaX))
            !
            ! d(Eta)           d(v)
            ! -----  + Eta * -----------
            ! dt               dy
            !            
            hdvdy = H0 * ( ( v(xc,yc)  - v(xc,yb))/(DeltaY))
            !
            ! d(Eta)        d(Eta)
            ! -----  + u * ---------  =0
            ! dt             dx 
            !
            !
            ubar= (u(xc,yc)+u(xc,yb))/2.0
            dhudx = ubar*(h(xf,yc)  - h(xb,yc))/(2.0*DeltaX)
            !
            ! d(Eta)          d(Eta)
            ! -----   + v * ----------- =0
            ! dt              dy    
            !
            vbar= (v(xc,yc)+v(xc,yb))/2.0
            dhvdy = vbar*(h(xc,yf)  - h(xc,yb))/(2.0*DeltaY)

            !
            !                  -         -
            ! d(u)            |d(d(h))    |
            ! -----  - Neta * |--------   | = 0
            ! dt              |dxdx       |
            !                  -         -
            vis2dhdx= - vis*((h(xf,yc) - 2.0*h(xc,yc) + h(xb,yc))/(DeltaX*DeltaX))
            !
            !                  -       -
            ! d(u)             |d(d(h)) |
            ! -----   - Neta * |--------| = 0
            ! dt               |dxdx    |
            !                  -       -
            vis2dhdy= - vis*((h(xc,yf) - 2.0*h(xc,yc) + h(xc,yb))/(DeltaY*DeltaY)) 

            !                                                                                         -         -          -         -
            ! d(Eta)           d(u)              d(v)               d(Eta)         d(Eta)            |d(d(h))    |        |d(d(h))    |
            ! -----  + Eta * --------- + Eta * -----------  + u * --------- + v * -----------  -vis* |--------   |  -vis* |--------   | =0
            ! dt                dx                dy                dx               dy              |dxdx       |        |dxdx       |
            !                                                                                         -         -          -         -
            !
            TermEqConH(i,j) = -(hdudx + hdvdy + dhudx + dhvdy + vis2dhdx +vis2dhdy )
            
         END DO
      END DO
   END SUBROUTINE Solve_Forward_Beta_plane_Grid_E
   
      
   SUBROUTINE  Solve_Forward_Beta_plane(TermEqMomU, TermEqMomV, TermEqConH, u, v, h)

      REAL(KIND=8), dimension(1:Idim, 1:Jdim), intent( in) ::  u,  v,  h
      REAL(KIND=8), dimension(1:Idim, 1:Jdim), intent(out) :: TermEqMomU, TermEqMomV, TermEqConH
      
      INTEGER :: xb,xc,xf,i,j,yb,yc,yf
      REAL(KIND=8)    :: udux,vduy,vbar,fcov,gdhx,ru,vis2dudx,vis2dudy
      REAL(KIND=8)    :: udvx,vdvy,ubar,fcou,gdhy,rv,vis2dvdx,vis2dvdy
      REAL(KIND=8)    :: hdudx,hdvdy,dhudx ,dhvdy
      REAL(KIND=8)    :: betau,betav
      DO j=1,Jdim
         CALL index(j,Jdim,yb,yc,yf)
         DO i=1,Idim
            CALL index(i,Idim,xb,xc,xf)
            !
            !                     
            ! d(u)         d(u)   
            ! -----  + u * ------   = 0
            ! dt           dx     
            !                     
            udux  = u(xc,yc)*((u(xf,yc) - u(xc,yc))/DeltaX)
            !
            !
            ! d(u)          d(u) 
            ! -----  + v * -------   = 0
            ! dt            dy  
            !

            vbar  = 0.25*(v(xc,yf) + v(xb,yf) +   v(xb,yc)+   v(xc,yc))
            vduy  = vbar*((u(xc,yf) - u(xc,yc))/DeltaY)
            !
            !
            ! d(u)
            ! -----   - f*v   = 0
            ! dt
            !
            fcov  = - f(xc,yc)*vbar 
            !
            ! d(u)
            ! -----   - beta*v  =0
            ! dt
            !
	    !
	    betav = -Beta*coordY(xc,yc)*vbar
            !
            ! d(u)       d(Eta)  
            ! -----   +g -------- 
            ! dt          dx
            !
            !
            gdhx  = g*((h(xc,yc) - h (xb,yc))/DeltaX)
            !
            !
            !       
            !       
            ! + b*u 
            !       
            !       
            ru    =  r*u(xc,yc)
            !
            !                  -         -
            ! d(u)            |d(d(u))    |
            ! -----  - Neta * |--------   | = 0
            ! dt              |dxdx       |
            !                  -         -
            vis2dudx= - vis*((u(xf,yc) - 2.0*u(xc,yc) + u(xb,yc))/(DeltaX*DeltaX))
            !
            !                  -       -
            ! d(u)             |d(d(u)) |
            ! -----   - Neta * |--------| = 0
            ! dt               |dxdx    |
            !                  -       -
            vis2dudy= - vis*((u(xc,yf) - 2.0*u(xc,yc) + u(xc,yb))/(DeltaY*DeltaY)) 
            !
            !                                                                          -                      -
            ! d(u)         d(u)         d(u)                 d(Eta)                   |d(d(u))       d(d(u))   |
            ! -----  + u * ------ + v * -------  - beta*v  +g -------- + b*u - Neta * |-------- + ------------ | = 0
            ! dt           dx            dy                   dx                      |dxdx          dydy      |
            !                                                                          -                      -
            !TermEqMomU(i,j) = udux +  vduy + fcov + gdhx + ru + vis2dudx + vis2dudy

            TermEqMomU(i,j) = -1*(udux + vduy + betav + gdhx + ru + vis2dudx + vis2dudy)
            !
            !
            ! d(v)         d(v) 
            ! -----  + u * -------   =0
            ! dt           dx  
            !
            !
            ubar=0.25*(u(xf,yc) + u(xc,yc) + u(xc,yb) + u(xf,yb) )
            udvx = ubar * ((v(xf,yc) - v(xc,yc))/DeltaX)
            !
            !
            ! d(v)         d(v) 
            ! -----  + v * -------   =0
            ! dt           dy  
            !
            !
            vdvy = v(xc,yc) * ((v(xc,yf) - v(xc,yc))/DeltaY)
            !
            !
            ! d(v)
            ! -----   + f*u  =0
            ! dt
            !
            fcou = f(xc,yc) * ubar
            !
            ! d(v)
            ! -----    + beta*u  =0
            ! dt
            !
            betau = Beta*coordY(xc,yc) *  ubar
            !
            !
            ! d(v)        d(Eta)  
            ! -----   +g -------- =0
            ! dt          dy
            !
            !
            gdhy = g * ((h(xc,yc) - h(xc,yb))/DeltaY)
            !
            !
            ! d(v)
            ! -----  + b*v =0
            ! dt
            !
            !
            rv = r * v(xc,yc)
            !                       -       -
            ! d(v)                 |d(d(v))  |
            ! -----       - Neta * |-------- | =0
            ! dt                   |dxdx     |
            !                       -       -
            vis2dvdx = -vis*((v(xf,yc) - 2.0*v(xc,yc) + v(xb,yc))/(DeltaX*DeltaX) ) 
            !                  -             -
            ! d(v)            |    d(d(v))   |
            ! -----  - Neta * | ------------ | =0
            ! dt              |    dydy      |
            !                  -            -
            vis2dvdy = -vis* ((v(xc,yf) -2.0*v(xc,yc) + v(xc,yb))/(DeltaY*DeltaY) )
            !                                                                       -                      -
            ! d(v)         d(v)         d(v)                 d(Eta)                |d(d(v))       d(d(v))   |
            ! -----  + u * ------ + v * -------  + beta*u  +g -------- + b*v - Neta * |-------- + ------------ | =0
            ! dt           dx            dy                   dy                   |dxdx          dydy      |
            !                                                                       -                      -
            
            !TermEqMomV(i,j) = udvx + vdvy + fcou + gdhy + rv + vis2dvdx + vis2dvdy
            
            TermEqMomV(i,j) = -(udvx + vdvy + betau + gdhy + rv + vis2dvdx + vis2dvdy)
            !
            ! d(Eta)           d(u)    
            ! -----  + Eta * ---------  =0
            ! dt                dx     
            !
            hdudx = H0 * ((u(xf,yc)  - u(xc,yc))/DeltaX)
            !
            ! d(Eta)           d(v)
            ! -----  + Eta * -----------
            ! dt               dy
            !            
            hdvdy = H0 * ((v(xc,yf)  - v(xc,yc))/DeltaY)
            !
            ! d(Eta)        d(Eta)
            ! -----  + u * ---------  =0
            ! dt             dx 
            !
            dhudx = (h(xf,yc)*u(xf,yc)  - h(xc,yc)*u(xc,yc))/DeltaX
            !
            ! d(Eta)          d(Eta)
            ! -----   + v * ----------- =0
            ! dt              dy    
            !
            dhvdy = (h(xc,yf)*v(xc,yf)  - h(xc,yc)*v(xc,yc))/DeltaY
            !
            ! d(Eta)           d(u)              d(v)               d(Eta)         d(Eta)
            ! -----  + Eta * --------- + Eta * -----------  + u * --------- + v * ----------- =0
            ! dt                dx                dy                dx               dy    
            TermEqConH(i,j) = -(hdudx + hdvdy + dhudx + dhvdy)
            
         END DO
      END DO
   END SUBROUTINE Solve_Forward_Beta_plane
   
 
   SUBROUTINE  Solve_Forward_f_plane(TermEqMomU, TermEqMomV, TermEqConH, u, v, h)

      REAL(KIND=8), dimension(1:Idim, 1:Jdim), intent( in) ::  u,  v,  h
      REAL(KIND=8), dimension(1:Idim, 1:Jdim), intent(out) :: TermEqMomU, TermEqMomV, TermEqConH
      
      INTEGER :: xb,xc,xf,i,j,yb,yc,yf
      REAL(KIND=8)    :: udux,vduy,vbar,fcov,gdhx,ru,vis2dudx,vis2dudy
      REAL(KIND=8)    :: udvx,vdvy,ubar,fcou,gdhy,rv,vis2dvdx,vis2dvdy
      REAL(KIND=8)    :: hdudx,hdvdy,dhudx ,dhvdy
      REAL(KIND=8)    :: betau,betav
      DO j=1,Jdim
         CALL index(j,Jdim,yb,yc,yf)
         DO i=1,Idim
            CALL index(i,Idim,xb,xc,xf)
            !
            udux  = u(xc,yc)*((u(xf,yc) - u(xc,yc))/DeltaX)
            !
            vbar  = 0.25*(v(xc,yf) + v(xb,yf) +   v(xb,yc)+   v(xc,yc))
            vduy  = vbar*((u(xc,yf) - u(xc,yc))/DeltaY)
            !
            fcov  = - f(xc,yc)*vbar 
	    !
	    betav = -Beta*coordY(xc,yc)*vbar
            !
            gdhx  = g*((h(xc,yc) - h (xb,yc))/DeltaX)
            !
            ru    =  r*u(xc,yc)
            !
            !        -        -
            !       |d(d(u))   |
            ! Neta *|--------  |
            !       |dxdx      |
            !        -        -
            !uxx = (u(xf,yc) - 2.0*u(xc,yc) + u(xb,yc))*nudxx

            vis2dudx= - vis*((u(xf,yc) - 2.0*u(xc,yc) + u(xb,yc))/(DeltaX*DeltaX))
            !
            !        -        -
            !       |d(d(u))   |
            ! Neta *|--------  |
            !       |dydy      |
            !        -        -
            !uxx = (u(xf,yc) - 2.0*u(xc,yc) + u(xb,yc))*nudxx
            ! 
            vis2dudy= - vis*((u(xc,yf) - 2.0*u(xc,yc) + u(xc,yb))/(DeltaY*DeltaY)) 
            !
            !                                                                          -                      -
            ! d(u)         d(u)         d(u)                 d(Eta)                   |d(d(u))       d(d(u))   |
            ! -----  + u * ------ + v * -------  - beta*v  +g -------- + b*u - Neta * |-------- + ------------ | = 0
            ! dt           dx            dy                   dx                      |dxdx          dydy      |
            !                                                                          -                      -
            !TermEqMomU(i,j) = udux +  vduy + fcov + gdhx + ru + vis2dudx + vis2dudy

            TermEqMomU(i,j) = -(udux +  vduy + fcov + gdhx + ru + vis2dudx + vis2dudy)

            !
            ubar=0.25*(u(xf,yc) + u(xc,yc) + u(xc,yb) + u(xf,yb) )
            udvx = ubar * ((v(xf,yc) - v(xc,yc))/DeltaX)
            !
            vdvy = v(xc,yc) * ((v(xc,yf) - v(xc,yc))/DeltaY)
            !
            fcou = f(xc,yc) * ubar
            betau = Beta*coordY(xc,yc) *  ubar
            !
            gdhy = g * ((h(xc,yc) - h(xc,yb))/DeltaY)
            !
            rv = r * v(xc,yc)
            !
            !        -        -
            !       |d(d(u))   |
            ! Neta *|--------  |
            !       |dxdx      |
            !        -        -
            !uxx = (u(xf,yc) - 2.0*u(xc,yc) + u(xb,yc))*nudxx
            
            vis2dvdx = -vis*((v(xf,yc) - 2.0*v(xc,yc) + v(xb,yc))/(DeltaX*DeltaX) ) 
            !        -        -
            !       |d(d(u))   |
            ! Neta *|--------  |
            !       |dydy      |
            !        -        -
            !uxx = (u(xf,yc) - 2.0*u(xc,yc) + u(xb,yc))*nudxx
            
            vis2dvdy = -vis* ((v(xc,yf) -2.0*v(xc,yc) + v(xc,yb))/(DeltaY*DeltaY) )
            !                                                                       -                      -
            ! d(v)         d(v)         d(v)                 d(Eta)                |d(d(v))       d(d(v))   |
            ! -----  + u * ------ + v * -------  + beta*u  +g -------- + b*v - Neta * |-------- + ------------ | =0
            ! dt           dx            dy                   dy                   |dxdx          dydy      |
            !                                                                       -                      -
            
            !TermEqMomV(i,j) = udvx + vdvy + fcou + gdhy + rv + vis2dvdx + vis2dvdy
            
            TermEqMomV(i,j) = -(udvx + vdvy + fcou + gdhy + rv + vis2dvdx + vis2dvdy)

            hdudx = H0 * ((u(xf,yc)  - u(xc,yc))/DeltaX)
            
            hdvdy = H0 * ((v(xc,yf)  - v(xc,yc))/DeltaY)

            dhudx = (h(xf,yc)*u(xf,yc)  - h(xc,yc)*u(xc,yc))/DeltaX
            
            dhvdy = (h(xc,yf)*v(xc,yf)  - h(xc,yc)*v(xc,yc))/DeltaY
            !
            ! d(Eta)           d(u)              d(v)               d(Eta)         d(Eta)
            ! -----  + Eta * --------- + Eta * -----------  + u * --------- + v * ----------- =0
            ! dt                dx                dy                dx               dy    
            TermEqConH(i,j) = -(hdudx + hdvdy + dhudx + dhvdy)
            
         END DO
      END DO
   END SUBROUTINE Solve_Forward_f_plane
 
   
   SUBROUTINE Solve(TermEqMomU, TermEqMomV, TermEqConH, u, v, h)

      REAL(KIND=8), dimension(1:Idim, 1:Jdim), intent( in) ::  u,  v,  h
      REAL(KIND=8), dimension(1:Idim, 1:Jdim), intent(out) :: TermEqMomU, TermEqMomV, TermEqConH
      
      INTEGER :: xb,xc,xf,i,j,yb,yc,yf
      REAL(KIND=8)    :: udux,vduy,vbar,fcov,gdhx,ru,vis2dudx,vis2dudy
      REAL(KIND=8)    :: udvx,vdvy,ubar,fcou,gdhy,rv,vis2dvdx,vis2dvdy
      REAL(KIND=8)    :: hdudx,hdvdy,dhudx ,dhvdy
      REAL(KIND=8)    :: betau,betav
      DO j=1,Jdim
         CALL index(j,Jdim,yb,yc,yf)
         DO i=1,Idim
            CALL index(i,Idim,xb,xc,xf)
            !
            udux  = u(xc,yc)*((u(xf,yc) - u(xb,yc))/(2.0*DeltaX))
            !
            vbar  = 0.25*(v(xc,yf) + v(xb,yf) +   v(xb,yc)+   v(xc,yc))
            vduy  = vbar*   ((u(xc,yf) - u(xc,yb))/(2.0*DeltaY))
            !
            vbar  = 0.25*(v(xc,yf) + v(xb,yf) +   v(xb,yc)+   v(xc,yc))
            fcov  =  f(xc,yc)*vbar 
	    betav = Beta*coordY(xc,yc)*vbar
            !
            gdhx  = g*((h(xf,yc) - h (xb,yc))/(2.0*DeltaX))
            !
            ru    =  r*u(xc,yc)
            !
            !        -        -
            !       |d(d(u))   |
            ! Neta *|--------  |
            !       |dxdx      |
            !        -        -
            !uxx = (u(xf,yc) - 2.0*u(xc,yc) + u(xb,yc))*nudxx

            vis2dudx=  vis*((u(xf,yc) - 2.0*u(xc,yc) + u(xb,yc))/(DeltaX*DeltaX))
            !
            !        -        -
            !       |d(d(u))   |
            ! Neta *|--------  |
            !       |dydy      |
            !        -        -
            !uxx = (u(xf,yc) - 2.0*u(xc,yc) + u(xb,yc))*nudxx
            ! 
            vis2dudy=  vis*((u(xc,yf) - 2.0*u(xc,yc) + u(xc,yb))/(DeltaY*DeltaY)) 
            !
            !                                                                       -                      -
            ! d(u)         d(u)         d(u)                 d(Eta)                |d(d(u))       d(d(u))   |
            ! -----  + u * ------ + v * -------  - f*v  +g -------- + b*u - Neta * |-------- + ------------ | =0
            ! dt           dx            dy                   dx                   |dxdx          dydy      |
            !                                                                       -                      -
            !TermEqMomU(i,j) = -udux -vduy + fcov - gdhx - ru + vis2dudx + vis2dudy
            TermEqMomU(i,j) = -udux -vduy + betav - gdhx - ru + vis2dudx + vis2dudy
	    
            !
            ubar= 0.25*(u(xf,yc) + u(xc,yc) + u(xc,yb) + u(xf,yb) )
            udvx = ubar * ((v(xf,yc) - v(xb,yc))/(2.0*DeltaX))
            !
            vdvy = v(xc,yc) * ((v(xc,yf) - v(xc,yb))/(2.0*DeltaY))
            !
            ubar= 0.25*(u(xf,yc) + u(xc,yc) + u(xc,yb) + u(xf,yb) )
            fcou = f(xc,yc) * ubar
            betau = Beta*coordY(xc,yc) *  ubar
            !
            gdhy = g * ((h(xc,yf) - h(xc,yb))/(2*DeltaY))
            !
            rv = r * v(xc,yc)
            !
            !        -        -
            !       |d(d(u))   |
            ! Neta *|--------  |
            !       |dxdx      |
            !        -        -
            !uxx = (u(xf,yc) - 2.0*u(xc,yc) + u(xb,yc))*nudxx
            
            vis2dvdx = vis*((v(xf,yc) - 2.0*v(xc,yc) + v(xb,yc))/(DeltaX*DeltaX) ) 
            !        -        -
            !       |d(d(u))   |
            ! Neta *|--------  |
            !       |dydy      |
            !        -        -
            !uxx = (u(xf,yc) - 2.0*u(xc,yc) + u(xb,yc))*nudxx
            
            vis2dvdy = vis* ((v(xc,yf) -2.0*v(xc,yc) + v(xc,yb))/(DeltaY*DeltaY) )
            !                                                                       -                      -
            ! d(v)         d(v)         d(v)                 d(Eta)                |d(d(v))       d(d(v))   |
            ! -----  + u * ------ + v * -------  + f*u  +g -------- + b*v - Neta * |-------- + ------------ | =0
            ! dt           dx            dy                   dy                   |dxdx          dydy      |
            !                                                                       -                      -
            
            !TermEqMomV(i,j) = -udvx - vdvy - fcou - gdhy - rv + vis2dvdx + vis2dvdy
            TermEqMomV(i,j) = -udvx - vdvy - betau - gdhy - rv + vis2dvdx + vis2dvdy
            
            hdudx = H0 * ((u(xf,yc)  - u(xb,yc))/(2.0*DeltaX))
            
            hdvdy = H0 * ((v(xc,yf)  - v(xc,yb))/(2.0*DeltaY))

            dhudx = (h(xf,yc)*u(xf,yc)  - h(xb,yc)*u(xb,yc))/(2.0*DeltaX)
            
            dhvdy = (h(xc,yf)*v(xc,yf)  - h(xc,yb)*v(xc,yb))/(2.0*DeltaY)
            !
            ! d(Eta)           d(u)              d(v)               d(Eta)         d(Eta)
            ! -----  + Eta * --------- + Eta * -----------  + u * --------- + v * ----------- =0
            ! dt                dx                dy                dx               dy    
            TermEqConH(i,j) = -(hdudx +hdvdy+ dhudx + dhvdy)
            
         END DO
      END DO
   END SUBROUTINE Solve



   
   SUBROUTINE index(i,Idim,xb,xc,xf)
      IMPLICIT NONE
      INTEGER, INTENT(IN   ) :: i
      INTEGER, INTENT(IN   ) :: Idim
      INTEGER, INTENT(OUT  ) :: xb,xc,xf
      IF(i==1) THEN
        xb=Idim
        xc=i
        xf=i+1
      ELSE IF(i==Idim)THEN
        xb=Idim-1
        xc=Idim
        xf=1
      ELSE
        xb=i-1
        xc=i
        xf=i+1
      END IF
   END SUBROUTINE index

   SUBROUTINE index_block(i,Idim,xb,xc,xf)
      IMPLICIT NONE
      INTEGER, INTENT(IN   ) :: i
      INTEGER, INTENT(IN   ) :: Idim
      INTEGER, INTENT(OUT  ) :: xb,xc,xf
      IF(i==1) THEN
        xb=i
        xc=i
        xf=i
      ELSE IF(i==Idim)THEN
        xb=Idim
        xc=Idim
        xf=Idim
      ELSE
        xb=i-1
        xc=i
        xf=i+1
      END IF
   END SUBROUTINE index_block

END MODULE ModSWE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PROGRAM  Main
  USE ModSWE, Only :    DefineCI,Solve,u,v,h,Idim,Jdim,DeltaT,&
                        f,nt,Solve_Forward_Beta_plane,coordY,Solve_Forward_f_plane,&
			Solve_Forward_Beta_plane_Sphera,&
			Solve_Forward_Beta_plane_Grid_E
   IMPLICIT NONE
   INTEGER :: it,irec,lrec
   REAL(KIND=4),ALLOCATABLE :: aux(:, :)
   REAL(KIND=8),ALLOCATABLE :: uc(:, :), un(:, :), ub(:, :)
   REAL(KIND=8),ALLOCATABLE :: vc(:, :), vn(:, :), vb(:, :)
   REAL(KIND=8),ALLOCATABLE :: hc(:, :), hn(:, :), hb(:, :)

  CALL Init()
  ALLOCATE(aux(1:Idim, 1:Jdim))
  ALLOCATE(uc(1:Idim, 1:Jdim), un(1:Idim, 1:Jdim), ub(1:Idim, 1:Jdim))
  ALLOCATE(vc(1:Idim, 1:Jdim), vn(1:Idim, 1:Jdim), vb(1:Idim, 1:Jdim))
  ALLOCATE(hc(1:Idim, 1:Jdim), hn(1:Idim, 1:Jdim), hb(1:Idim, 1:Jdim))
  INQUIRE(IOLENGTH=lrec)aux
  OPEN(1,FILE='saidaE.bin',ACCESS='DIRECT',FORM='UNFORMATTED', RECL=lrec,&
       STATUS='UNKNOWN',ACTION='WRITE')
  irec=0
  irec=irec+1
  aux=u
  WRITE(1,rec=irec)REAL(aux,KIND=4)
  irec=irec+1
  aux=v
  WRITE(1,rec=irec)REAL(aux,KIND=4)
  irec=irec+1
  aux=h
  WRITE(1,rec=irec)REAL(aux,KIND=4)
  irec=irec+1
  aux=f
  WRITE(1,rec=irec)REAL(aux,KIND=4)
  irec=irec+1
  aux=coordY
  WRITE(1,rec=irec)REAL(aux,KIND=4)

  DO it=1,nt
      CALL Run(it)
      
      irec=irec+1
      aux=u
      WRITE(1,rec=irec)aux
      irec=irec+1
      aux=v
      WRITE(1,rec=irec)aux
      irec=irec+1
      aux=h
      WRITE(1,rec=irec)aux
      irec=irec+1
      aux=f
      WRITE(1,rec=irec)REAL(aux,KIND=4)
      irec=irec+1
      aux=coordY
      WRITE(1,rec=irec)REAL(aux,KIND=4)

  END DO
  CALL  Finalize()
CONTAINS
  SUBROUTINE Init()
      CALL DefineCI()
  END SUBROUTINE Init



   FUNCTION Filter_RAW(var_b,var_c,var_f)  RESULT (ok)
      IMPLICIT NONE
      REAL(KIND=8),  INTENT(IN   ) :: var_b (1:Idim, 1:Jdim)
      REAL(KIND=8),  INTENT(INOUT) :: var_c (1:Idim, 1:Jdim)
      REAL(KIND=8),  INTENT(INOUT) :: var_f (1:Idim, 1:Jdim)
      REAL(KIND=8) :: Deslc (1:Idim, 1:Jdim)
      REAL(KIND=8), PARAMETER :: alfa=0.05
      REAL(KIND=8), PARAMETER :: beta=0.05    !0.5 < beta <= 1
      INTEGER :: i,j
      INTEGER :: ok
      DO j=1,Jdim
         DO i=1,Idim
            Deslc(i,j) = alfa*(var_b(i,j) - 2.0*var_c(i,j) + var_f(i,j) )
            var_c(i,j) = var_c(i,j) + Deslc(i,j)
            var_f(i,j) = var_f(i,j) + Deslc(i,j)*(beta-1.0)
         END DO
      END DO
      ok=0
  END FUNCTION Filter_RAW
  
  SUBROUTINE Run(it)
      INTEGER, INTENT(IN  ) :: it
      REAL(KIND=8), dimension(1:Idim, 1:Jdim) :: ku1, uo, un
      REAL(KIND=8), dimension(1:Idim, 1:Jdim) :: ku2, ku3, ku4
      REAL(KIND=8), dimension(1:Idim, 1:Jdim) :: kv1, vo, vn
      REAL(KIND=8), dimension(1:Idim, 1:Jdim) :: kv2, kv3, kv4
      REAL(KIND=8), dimension(1:Idim, 1:Jdim) :: kh1, ho, hn
      REAL(KIND=8), dimension(1:Idim, 1:Jdim) :: kh2, kh3, kh4
      REAL(KIND=8) :: dt6
      REAL(KIND=8) :: dt2
      REAL(KIND=8) :: dt


        dt=DeltaT;  dt6 = DeltaT/6.0; dt2=DeltaT*0.5
       
        uo =  u(:,:)
        vo =  v(:,:)
        ho =  h(:,:)
        PRINT*,'h eta',MAXVAL(h),MINVAL(h),'u eta',MAXVAL(u),MINVAL(u),'v eta',MAXVAL(v),MINVAL(v)

        call Solve_Forward_Beta_plane_Grid_E(ku1, kv1, kh1, uo        , vo        , ho        )
        call Solve_Forward_Beta_plane_Grid_E(ku2, kv2, kh2, uo+ku1*dt2, vo+kv1*dt2, ho+kh1*dt2)
        call Solve_Forward_Beta_plane_Grid_E(ku3, kv3, kh3, uo+ku2*dt2, vo+kv2*dt2, ho+kh2*dt2)
        call Solve_Forward_Beta_plane_Grid_E(ku4, kv4, kh4, uo+ku3*dt , vo+kv3*dt , ho+kh3*dt )

    ! final step and time marching / new values for RK4
        un = uo + (ku1 + 2.0*ku2 + 2.0*ku3 + ku4)*dt6
        vn = vo + (kv1 + 2.0*kv2 + 2.0*kv3 + kv4)*dt6
        hn = ho + (kh1 + 2.0*kh2 + 2.0*kh3 + kh4)*dt6

        ! updating the data
        u = un
        v = vn
        h = hn

   END SUBROUTINE Run

     
  SUBROUTINE Finalize()
      
  END SUBROUTINE Finalize
END PROGRAM  Main
