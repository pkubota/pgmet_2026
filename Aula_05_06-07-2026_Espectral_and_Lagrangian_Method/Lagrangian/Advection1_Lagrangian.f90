MODULE Advection1D
  IMPLICIT NONE

! ------------------------------------------------------------------------------------
!
! Default Setup
!
 INTEGER   :: N         = 99!99 Number of points
 REAL      :: xmin      = 1.!Minimum X valueLF
 REAL      :: xmax      = 10.!Maximum X value
 REAL      :: tend      = 100.!End time of simulation
 REAL      :: dt        = 0.01!Time step
 REAL      :: a         = 1.6!Advection velocity
 CHARACTER(LEN=10) :: init        = 'gaussian' ! Initial condition. Possible values: gaussian, sin, heavy
 CHARACTER(LEN=10) :: scheme      = 'SL2M'  ! Advection scheme. Possible values: upstream, SL1,SL1M, SL3, SL3M
 CHARACTER(LEN=10) :: output_file = 'out'
 LOGICAL   :: final_plot   = .True.
 LOGICAL   :: write_output = .True.


CONTAINS

 SUBROUTINE InitAD(arg)
   IMPLICIT NONE
   INTEGER :: arg
   PRINT*,'InitAD', arg

 END SUBROUTINE InitAD

 SUBROUTINE Savef(unitout,irec, Q )
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: unitout
   INTEGER, INTENT(INOUT) :: irec
   REAL   , INTENT(IN) :: Q(:)
   ! """
   ! Save variable to file
   ! """
   if( write_output )THEN
    irec=irec+1
    WRITE(unitout,rec=irec)Q
   END IF
 END SUBROUTINE Savef
 

 FUNCTION SemiLagrangian(Q,X,t,dt,dx,order,monotone)  result(Qnp1)
   ! """
   ! Lax-Wendroff advection scheme
   ! """
   IMPLICIT NONE
   REAL   , INTENT(IN) :: Q(:)
   REAL   , INTENT(IN) :: X(:)
   REAL   , INTENT(IN) :: t
   REAL   , INTENT(IN) :: dt
   REAL   , INTENT(IN) :: dx
   INTEGER, INTENT(IN) :: order
   LOGICAL, INTENT(IN),OPTIONAL :: monotone
   
   REAL                :: Qnp1 (SIZE(Q))
   INTEGER             :: N0
   INTEGER             :: N1
   REAL                :: xp
   INTEGER             :: j
   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)

   DO j=N0,N1!for j in range(N):
       ! Find departure point
        xp = X(j) - a*dt
       DO WHILE  ( xp <= X(N0) .or. xp >= X(N1) )
            if (xp <=  X(N0))xp = X(N1) - (X(N0)-xp)
            if( xp >=  X(N1))xp = X(N0) + (xp-X(N1))
       END DO
       ! # Interpolate
        if( order == 5 )THEN
             Qnp1(j) = order5_interpolation(xp,X,dx,Q,monotone)
        else if( order == 3 )THEN
            Qnp1(j) = cubic_interpolation(xp,X,dx,Q,monotone)
        else if( order == 2 )THEN
            Qnp1(j) = Quadratica_interpolation(xp,X,dx,Q,monotone)
        else if( order == 1 )THEN
            Qnp1(j) = linear_interpolation(xp,X,dx,Q,monotone)
        endif  
   END DO
 END FUNCTION SemiLagrangian  



 FUNCTION linear_interpolation(xp,X,dx,Q,monotone) result(qq)
    !"""
    !Linear interpolation (1st order)
    !"""
   IMPLICIT NONE
   REAL   , INTENT(IN) :: xp
   REAL   , INTENT(IN) :: X(:)
   REAL   , INTENT(IN) :: dx
   REAL   , INTENT(IN) :: Q(:)
   LOGICAL, INTENT(IN),OPTIONAL :: monotone
   INTEGER   :: p
   INTEGER   :: N0
   INTEGER   :: N1
   INTEGER   :: i,k
   REAL      :: Xs(1:2)
   REAL      :: Qs(1:2)
   REAL      :: alfa
   REAL      :: c2
   REAL      :: qq
   REAL      :: qmax
   REAL      :: qmin
   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)
    !# Find point below x
    p = int(floor(xp/dx))
    IF(p>=N1)p=p-N1+1
    IF(p<=N0)p=N1+p

    !# Stencil that makes interpolation periodic
    if(a < 0)THEN
       if( p>=N1)THEN
          Xs(1:2) = (/ X(N1), X(N1-1  ) /)
          Qs(1:2) = (/ Q(N1), Q(N1-1  ) /)
          alfa      = ABS(xp-Xs(2))/ABS(Xs(1)-Xs(2))
       else if( p<=N0)THEN
          Xs(1:2) = (/ X(N0), X(N1) /)
          Qs(1:2) = (/ Q(N0), Q(N1) /)
          alfa      = ABS(xp-Xs(2))/ABS(Xs(1)-Xs(2))
       else
          Xs(1:2) = (/ X(p), X(p-1) /)
          Qs(1:2) = (/ Q(p), Q(p-1) /)
          alfa      = ABS(xp-Xs(2))/ABS(Xs(1)-Xs(2))
       end if    
    ELSE
       if( p>=N1)THEN
          Xs(1:2) = (/ X(N1), X(N1-1  ) /)
          Qs(1:2) = (/ Q(N1), Q(N1-1  ) /)
          alfa      = ABS(xp-Xs(2))/ABS(Xs(1)-Xs(2))
       else if( p<=N0)THEN
          Xs(1:2) = (/ X(N0), X(N1) /)
          Qs(1:2) = (/ Q(N0), Q(N1) /)
          alfa      = ABS(xp-Xs(2))/ABS(Xs(1)-Xs(2))
       else
          Xs(1:2) = (/ X(p), X(p-1) /)
          Qs(1:2) = (/ Q(p), Q(p-1) /)
          alfa      = ABS(xp-Xs(2))/ABS(Xs(1)-Xs(2))
       end if    
    END IF
    IF(Qs(1)> 1e12)return

    !### Add interpolation code
    c2=1-alfa
    qq = alfa*Qs(1) + c2*Qs(2)
    IF(present(monotone))THEN
    if( monotone )THEN
        qmax = max(Qs(1),Qs(2))
        qmin = min(Qs(1),Qs(2))
        qq = max(qmin,min(qmax,qq))
    END IF
    END IF

 END FUNCTION linear_interpolation



 FUNCTION Quadratica_interpolation(xp,X,dx,Q,monotone) result(qq)
    !"""
    !Linear interpolation (1st order)
    !"""
   IMPLICIT NONE
   REAL   , INTENT(IN) :: xp
   REAL   , INTENT(IN) :: X(:)
   REAL   , INTENT(IN) :: dx
   REAL   , INTENT(IN) :: Q(:)
   LOGICAL, INTENT(IN),OPTIONAL :: monotone
   INTEGER   :: p
   INTEGER   :: N0
   INTEGER   :: N1,opt
   INTEGER   :: i,k
   REAL      :: Xs(0:2)
   REAL      :: Qs(0:2)
   REAL      :: alfa
   REAL      :: c2
   REAL      :: qq
   REAL      :: qmax,qmax1
   REAL      :: qmin,qmin1
   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)
    !# Find point below x
    p = int(floor(xp/dx))
    IF(p>=N1)p=p-N1+1
    IF(p<=N0)p=N1+p
    IF(p>=N1)p=p-N1+1

    qq = 0
    opt=2
    IF(opt==1)THEN
    !# Stencil that makes interpolation periodic
    if(a < 0)THEN
       IF(p>=N1)p=p-N1+1
       IF(p<=N0)p=N1+p

       if( p>=N1)THEN
          Xs(0:2) = (/ X(N0), X(N1  ), X(N1-1  )/)
          Qs(0:2) = (/ Q(N0), Q(N1  ), Q(N1-1  )/)
          alfa      = ABS(xp-Xs(2))/ABS(Xs(1)-Xs(2))
       else if( p<=N0)THEN
          Xs(0:2) = (/ X(N0+1), X(N0), X(N1)/)
          Qs(0:2) = (/ Q(N0+1), Q(N0), Q(N1)/)
          alfa      = ABS(xp-Xs(2))/ABS(Xs(1)-Xs(2))
       else
          Xs(0:2) = (/ X(p+1), X(p), X(p-1)/)
          Qs(0:2) = (/ Q(p+1), Q(p), Q(p-1)/)
          alfa      = ABS(xp-Xs(2))/ABS(Xs(1)-Xs(2))
       end if    
    ELSE
       if( p==N1)THEN
          Xs(0:2) = (/ X(N1), X(N1-1  ), X(N1-2  )/)
          Qs(0:2) = (/ Q(N1), Q(N1-1  ), Q(N1-2  )/)
          alfa      = (xp-Xs(1))/(Xs(1)-Xs(2))
       else if( p==N0)THEN
          Xs(0:2) = (/ X(N0), X(N1), X(N1-1)/)
          Qs(0:2) = (/ Q(N0), Q(N1), Q(N1-1)/)
          alfa      = (xp-Xs(1))/(Xs(1)-Xs(2))
       else if( p==N0-1)THEN
          Xs(0:2) = (/ X(N1), X(N1-1), X(N1-2)/)
          Qs(0:2) = (/ Q(N1), Q(N1-1), Q(N1-2)/)
          alfa      = (xp-Xs(1))/(Xs(1)-Xs(2))
       else if( p-1==0)THEN
          Xs(0:2) = (/ X(p), X(N1), X(N1-1)/)
          Qs(0:2) = (/ Q(p), Q(N1), Q(N1-1)/)
          alfa      = (xp-Xs(1))/(Xs(1)-Xs(2))
       else if( p-2==0)THEN
          Xs(0:2) = (/ X(p), X(p-1), X(N1)/)
          Qs(0:2) = (/ Q(p), Q(p-1), Q(N1)/)
          alfa      = (xp-Xs(1))/(Xs(1)-Xs(2))
       else
          Xs(0:2) = (/ X(p), X(p-1), X(p-2)/)
          Qs(0:2) = (/ Q(p), Q(p-1), Q(p-2)/)
          alfa      = (xp-Xs(1))/(Xs(1)-Xs(2))
       end if    
    END IF
    ELSE
    !# Stencil that makes interpolation periodic
    if(a < 0)THEN
       IF(p>=N1)p=p-N1+1
       IF(p<=N0)p=N1+p

       if( p>=N1)THEN
          Xs(0:2) = (/ X(N0), X(N1  ), X(N1-1  )/)
          Qs(0:2) = (/ Q(N0), Q(N1  ), Q(N1-1  )/)
          alfa      = ABS(xp-Xs(2))/ABS(Xs(1)-Xs(2))
       else if( p<=N0)THEN
          Xs(0:2) = (/ X(N0+1), X(N0), X(N1)/)
          Qs(0:2) = (/ Q(N0+1), Q(N0), Q(N1)/)
          alfa      = ABS(xp-Xs(2))/ABS(Xs(1)-Xs(2))
       else
          Xs(0:2) = (/ X(p+1), X(p), X(p-1)/)
          Qs(0:2) = (/ Q(p+1), Q(p), Q(p-1)/)
          alfa      = ABS(xp-Xs(2))/ABS(Xs(1)-Xs(2))
       end if    
    ELSE
       if( p==N1)THEN
          Xs(0:2) = (/ X(N0), X(N1  ), X(N1-1  )/)
          Qs(0:2) = (/ Q(N0), Q(N1  ), Q(N1-1  )/)
          alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
       else if( p==N0)THEN
          Xs(0:2) = (/ X(N0+1), X(N0), X(N1)/)
          Qs(0:2) = (/ Q(N0+1), Q(N0), Q(N1)/)
          alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
       else
          Xs(0:2) = (/ X(p+1), X(p), X(p-1)/)
          Qs(0:2) = (/ Q(p+1), Q(p), Q(p-1)/)
          alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
       end if    
    END IF
    END IF

    IF(Qs(0)> 1e12)return
    IF(Qs(2)> 1e12)return

    !### Add interpolation code
    qq = alfa*((alfa-1)/2.0)*Qs(0) - &
         (((alfa-1)*(alfa+1))/(1.0))*Qs(1) + &
         alfa*((alfa+1)/2.0)*Qs(2)
    IF(present(monotone))THEN
    if( monotone )THEN
        qmax = max(Qs(1),Qs(2))
        qmin = min(Qs(1),Qs(2))
        qq = max(qmin,min(qmax,qq))

!        qmax = max(Qs(0),Q(p),Qs(1))
!        qmax1= max(Qs(1),Q(p),Qs(2))
!        qmin = min(Qs(0),Q(p),Qs(1))
!        qmin1= min(Qs(1),Q(p),Qs(2))
!        qq = min(qq,max(qmax,qmax1)) 
!        qq = max(qq,min(qmin,qmin1)) 
    END IF
    END IF

 END FUNCTION Quadratica_interpolation
 
 
 
 
  FUNCTION cubic_interpolation(xp,X,dx,Q,monotone) result(qq)
   ! """
   ! Cubic interpolation (3rd order)
   ! """
   IMPLICIT NONE
   REAL   , INTENT(IN) :: xp
   REAL   , INTENT(IN) :: X(:)
   REAL   , INTENT(IN) :: dx
   REAL   , INTENT(IN) :: Q(:)
   LOGICAL, INTENT(IN),OPTIONAL :: monotone
   INTEGER   :: p
   REAL      :: c
   REAL      :: qq
   REAL      :: qmax,qmax1
   REAL      :: qmin,qmin1
   REAL      :: Xs(0:3)
   REAL      :: Qs(0:3)
   REAL      :: alfa
   INTEGER   :: N0
   INTEGER   :: N1
   INTEGER   :: i,k
   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)

   !  Find grid-point below x
   ! FLOOR(A) returns the greatest integer less than or equal to X.

    p = int(floor(xp/dx))
    IF(p>=N1)p=p-N1+1
    IF(p<=N0)p=N1+p
    IF(p>=N1)p=p-N1+1
   !  Stencil that makes interpolation periodic
    if(a < 0)THEN
    IF( p==N1 )THEN
        Qs(0:3) = (/ Q (N1-2  ) ,Q (N1-1  )   , Q (N1)      , Q (N0)      /)
        Xs(0:3) = (/ X (N1-2  ) ,X (N1-1  )   , X (N1)      , X (N0)      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    ELSE IF( p==N1+1)THEN
        Qs(0:3) = (/ Q (N1-1  ) ,Q (N1  )     , Q (N0)    , Q (N0+1)      /)
        Xs(0:3) = (/ X (N1-1  ) ,X (N1  )     , X (N0)    , X (N0+1)      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    elseif( p==N1-1 )THEN
        Qs(0:3) = (/ Q (N1-3)   ,Q (N1-2)     , Q (N1-1)      , Q (N1  )      /)
        Xs(0:3) = (/ X (N1-3)   ,X (N1-2)     , X (N1-1)      , X (N1  )      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    elseif( p==N0 )THEN
        Qs(0:3) = (/ Q (N1-1)   ,Q (N1)       , Q (N0)    , Q (N0+1)      /)
        Xs(0:3) = (/ X (N1-1)   ,X (N1)       , X (N0)    , X (N0+1)      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    elseif( p==N0+1 )THEN
        Qs(0:3) = (/ Q (N1  )   ,Q (N0  )     , Q (N0+1)    , Q (N0+2)      /)
        Xs(0:3) = (/ X (N1  )   ,X (N0  )     , X (N0+1)    , X (N0+2)      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    elseif( p==N0-1 )THEN
        Qs(0:3) = (/ Q (N1-2)   ,Q (N1-1)      , Q (N1)      , Q (N0)      /)
        Xs(0:3) = (/ X (N1-2)   ,X (N1-1)      , X (N1)      , X (N0)      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    else
        Qs(0:3) = (/ Q (p-2)    ,Q (p-1)       , Q (p)     , Q (p+1 )      /)
        Xs(0:3) = (/ X (p-2)    ,X (p-1)       , X (p)     , X (p+1 )      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    endif
    ELSE
    IF( p==N1 )THEN
        Qs(0:3) = (/ Q (N1-2  ) ,Q (N1-1  )   , Q (N1)      , Q (N0)      /)
        Xs(0:3) = (/ X (N1-2  ) ,X (N1-1  )   , X (N1)      , X (N0)      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    elseif( p==N1+1 )THEN
        Qs(0:3) = (/ Q (N1)   ,Q (N0)      , Q (N0+1)      , Q (N0+2)      /)
        Xs(0:3) = (/ X (N1)   ,X (N0)      , X (N0+1)      , X (N0+2)      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    elseif( p==N1-1 )THEN
        Qs(0:3) = (/ Q (N1-2)   ,Q (N1-1)      , Q (N1)      , Q (N0)      /)
        Xs(0:3) = (/ X (N1-2)   ,X (N1-1)      , X (N1)      , X (N0)      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    elseif( p==N1-2 )THEN
        Qs(0:3) = (/ Q (N1-3)   ,Q (N1-2)      , Q (N1-1)      , Q (N1)      /)
        Xs(0:3) = (/ X (N1-3)   ,X (N1-2)      , X (N1-1)      , X (N1)      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    elseif( p==N1+2 )THEN
        Qs(0:3) = (/ Q (N0)   ,Q (N0+1)      , Q (N0+2)      , Q (N0+3)      /)
        Xs(0:3) = (/ X (N0)   ,X (N0+1)      , X (N0+2)      , X (N0+3)      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    elseif( p>=100 )THEN
        Qs(0:3) = (/ Q (N0)   ,Q (N1)      , Q (N0+1)      , Q (N0+2)      /)
        Xs(0:3) = (/ X (N0)   ,X (N1)      , X (N0+1)      , X (N0+2)      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    elseif( p==N0 )THEN
        Qs(0:3) = (/ Q (N1)   ,Q (N0)      , Q (N0+1)      , Q (N0+2)      /)
        Xs(0:3) = (/ X (N1)   ,X (N0)      , X (N0+1)      , X (N0+2)      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    elseif( p==N0-1 )THEN
        Qs(0:3) = (/ Q (N1-1)   ,Q (N1)      , Q (N0)      , Q (N0+1)      /)
        Xs(0:3) = (/ X (N1-1)   ,X (N1)      , X (N0)      , X (N0+1)      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    elseif( p==N0-2 )THEN
        Qs(0:3) = (/ Q (N1-2)   ,Q (N1-1)      , Q (N1)      , Q (N0)      /)
        Xs(0:3) = (/ X (N1-2)   ,X (N1-1)      , X (N1)      , X (N0)      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    else
        Qs(0:3) = (/ Q (p-1)    ,Q (p  )       , Q (p+1)     , Q (p+2 )      /)
        Xs(0:3) = (/ X (p-1)    ,X (p  )       , X (p+1)     , X (p+2 )      /)
        alfa      = ABS(xp-Xs(1))/ABS(Xs(1)-Xs(2))
    endif
    END IF
    IF(Qs(1)> 1e12)return


    !### Add interpolation code
    qq = -(((1-alfa)*alfa*(1+alfa))/(6.0))*Qs(0) + &
         (((2-alfa)*alfa*(1+alfa))/(2.0))*Qs(1) + &
         (((2-alfa)*(1-alfa)*(1+alfa))/(2.0))*Qs(2) - &
         (((2-alfa)*(1-alfa)*alfa)/(6.0))*Qs(3) 

    IF(present(monotone))THEN
    if( monotone )THEN
        qmax = max(Qs(2),Qs(3) )
        qmin = min(Qs(2),Qs(3) )
        qq = max(qmin,min(qmax,qq))

!        qmax = max(Qs(0),Q(p),Qs(1))
!        qmax1= max(Qs(1),Q(p),Qs(2))
!        qmin = min(Qs(0),Q(p),Qs(1))
!        qmin1= min(Qs(1),Q(p),Qs(2))
!        qq = min(qq,max(qmax,qmax1)) 
!        qq = max(qq,min(qmin,qmin1)) 
    END IF
    END IF



 END FUNCTION cubic_interpolation
 
 
 
 
 FUNCTION order5_interpolation(xp,X,dx,Q,monotone) result(qq)
   ! """
   ! Cubic interpolation (3rd order)
   ! """
   IMPLICIT NONE
   REAL   , INTENT(IN) :: xp
   REAL   , INTENT(IN) :: X(:)
   REAL   , INTENT(IN) :: dx
   REAL   , INTENT(IN) :: Q(:)
   LOGICAL, INTENT(IN),OPTIONAL :: monotone
   INTEGER   :: p
   REAL      :: c
   REAL      :: qq
   REAL      :: qmax,qmax1
   REAL      :: qmin,qmin1
   REAL      :: Xs(1:5)
   REAL      :: Qs(1:5)
   INTEGER   :: N0
   INTEGER   :: N1
   INTEGER   :: i,k
   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)

   !  Find grid-point below x
   ! FLOOR(A) returns the greatest integer less than or equal to X.

    p = int(floor(xp/dx))
    IF(p>=N1)p=p-N1+1
    IF(p<=N0)p=N1+p
   !  Stencil that makes interpolation periodic

    IF( p==N1 )THEN
        Qs(1:5) = (/ Q (N1-2  ) ,Q (N1-1  ) , Q (N1  )  , Q (N0)      , Q (N0+1)      /)
        Xs(1:5) = (/ X (N1-2  ) ,X (N1-1  ) , X (N1  )  , X (N0)      , X (N0+1)      /)
    ELSE IF( p==N1+1)THEN
        Qs(1:5) = (/ Q (N1-1  ) ,Q (N1  )   , Q(N0   )  , Q (N0+1)    , Q (N0+2)      /)
        Xs(1:5) = (/ X (N1-2  ) ,X (N1  )   , X(N0   )  , X (N0+1)    , X (N0+2)      /)
    elseif( p==N1-1 )THEN
        Qs(1:5) = (/ Q (N1-3)   ,Q (N1-2)   , Q (N1-1)  , Q (N1)      , Q (N0  )      /)
        Xs(1:5) = (/ X (N1-3)   ,X (N1-2)   , X (N1-1)  , X (N1)      , X (N0  )      /)
    elseif( p==N0 )THEN
        Qs(1:5) = (/ Q (N1-1)   ,Q (N1)     , Q (N0  )  , Q (N0+1)    , Q (N0+2)      /)
        Xs(1:5) = (/ X (N1-1)   ,X (N1)     , X (N0  )  , X (N0+1)    , X (N0+2)      /)
    elseif( p==N0+1 )THEN
        Qs(1:5) = (/ Q (N1  )   ,Q (N0  )   , Q (N0+1)  , Q (N0+2)    , Q (N0+3)      /)
        Xs(1:5) = (/ X (N1  )   ,X (N0  )   , X (N0+1)  , X (N0+2)    , X (N0+3)      /)
    elseif( p==N0-1 )THEN
        Qs(1:5) = (/ Q (N1-2)   ,Q (N1-1)   , Q (N1)    , Q (N0)      , Q (N0+1)      /)
        Xs(1:5) = (/ X (N1-2)   ,X (N1-1)   , X (N1)    , X (N0)      , X (N0+1)      /)
    else
        Qs(1:5) = (/ Q (p-2)    ,Q (p-1)    , Q (p)     , Q (p+1)     , Q (p+2 )      /)
        Xs(1:5) = (/ X (p-2)    ,X (p-1)    , X (p)     , X (p+1)     , X (p+2 )      /)
    endif
    IF(Qs(1)> 1e12)return

    qq = 0
    !### Add interpolation code
    DO i=1,5!for i in range(4):
        c = 1
        DO k=1,5 !for k in range(4):
            if( k /= i )THEN

!       
!                 xp-Xs(2)        xp-Xs(3)         xp-Xs(4)         xp-Xs(5)
!   i=1      c= -------------*  ------------- *  ------------- *  -------------
!                Xs(1)-Xs(2)     Xs(1)-Xs(3)      Xs(1)-Xs(4)      Xs(1)-Xs(5)
!
                c = c * ((xp-Xs(k))/(Xs(i)-Xs(k)))
!       
!                 xp-Xs(2)              xp-Xs(3)               xp-Xs(4)                xp-Xs(5)
!   i=1     qq= -------------*Qs(1) * -------------*Qs(1) *  ------------- *Qs(1)*  -------------*Qs(1)
!                Xs(1)-Xs(2)           Xs(1)-Xs(3)            Xs(1)-Xs(4)             Xs(1)-Xs(5)
!

!       
!                 xp-Xs(1)              xp-Xs(3)               xp-Xs(4)                xp-Xs(5)
!   i=2     qq= -------------*Qs(2) * -------------*Qs(2) *  ------------- *Qs(2)*  -------------*Qs(2)
!                Xs(2)-Xs(1)           Xs(2)-Xs(3)            Xs(2)-Xs(4)             Xs(2)-Xs(5)
!

!       
!                 xp-Xs(1)              xp-Xs(2)               xp-Xs(4)                xp-Xs(5)
!   i=3     qq= -------------*Qs(3) * -------------*Qs(3) *  ------------- *Qs(3)*  -------------*Qs(3)
!                Xs(3)-Xs(1)           Xs(3)-Xs(2)            Xs(3)-Xs(4)             Xs(3)-Xs(5)
!

!       
!                 xp-Xs(1)              xp-Xs(2)               xp-Xs(3)                xp-Xs(5)
!   i=4     qq= -------------*Qs(4) * -------------*Qs(4) *  ------------- *Qs(3)*  -------------*Qs(4)
!                Xs(4)-Xs(1)           Xs(4)-Xs(2)            Xs(4)-Xs(3)             Xs(4)-Xs(5)
!

!       
!                 xp-Xs(1)              xp-Xs(2)               xp-Xs(4)                xp-Xs(4)
!   i=5     qq= -------------*Qs(5) * -------------*Qs(5) *  ------------- *Qs(5)*  -------------*Qs(5)
!                Xs(5)-Xs(1)           Xs(5)-Xs(2)            Xs(5)-Xs(4)             Xs(5)-Xs(4)
!

                qq = qq + c*Qs(i)
            END IF
        END DO
    END DO
    !### Add monotonicity code
    IF(present(monotone))THEN
    if( monotone )THEN
        qmax = max(Qs(1),Qs(2))
        qmin = min(Qs(1),Qs(2))
        qq = max(qmin,min(qmax,qq))
    END IF
    END IF
 END FUNCTION order5_interpolation
END MODULE Advection1D



PROGRAM Main

 USE Advection1D,Only:InitAD,xmin,xmax,N,dt,a,write_output,output_file,init,Savef,tend,&
                      scheme,SemiLagrangian
 IMPLICIT NONE
 REAL, ALLOCATABLE :: X(:)
 REAL, ALLOCATABLE :: Q(:)
 REAL, ALLOCATABLE :: Q0(:)
 REAL :: mu 
 REAL :: order
 REAL ::sig
 REAL :: dx
 REAL :: cfl
 REAL :: PI
 REAL :: startTime,EndTime,elapsed
 REAL :: t  = 0.
 REAL :: nn = 0
 INTEGER :: i
 INTEGER :: irec,lrec
 LOGICAL :: test
 CALL InitAD(1)
 ALLOCATE(X(0:N-1));X=0
 ALLOCATE(Q(0:N-1));Q=0
 ALLOCATE(Q0(0:N-1));Q0=0
 PI=4.0*ATAN(1.0)
 X = linspace( xmin, xmax, num=N, endpoint=.False. )
 PRINT*,X
 INQUIRE(IOLENGTH=lrec)Q
 dx = (xmax-xmin)/N
 cfl = dt/dx*abs(a)
 if(write_output )test=FileOpen(1,output_file,lrec)
 !if(write_output )test=FileClose(1)
   
  ! Initial condition
 print*, 'Set initial condition'
   
   
  if (TRIM(init) == 'sin')THEN !:         # Sin wave
    Q = sin(2.0*pi * X/(xmax-xmin))
  else if (TRIM(init) == 'gaussian')THEN!:  # Gaussian wave
    mu  = 0.5*(xmin+xmax)
    sig = 0.1*(xmax-xmin)
    Q = exp( -(X-mu)**2 / (2.*sig**2) )
  else if (TRIM(init) == 'heavy')THEN!:  # Heavy
    mu  = 0.5*(xmin+xmax)
    sig = 0.1*(xmax-xmin)
    Q = 0*X
    DO i=LBOUND(Q,DIM=1),UBOUND(Q,DIM=1)! 0,N !for n in range(N):
        if ( abs(mu-X(i)) < sig )THEN!:
            Q(i) = 1.0
        endif
       PRINT*,i
    END DO
  else
    Q = 0.0*X  
  END IF
  PRINT*,Q
 print*, 'copy initial condition'

  Q0 =copy(Q)
  PRINT*,Q0
  
! Time stepping
  t = 0.
  nn = 0
  irec=0
  print*, 'Start time stepping'
  
  CALL Savef(1,irec, Q )!save(Q,t,n)
  call cpu_time ( startTime )!start = clock()

   DO WHILE  (t < tend )      
    ! To make simulation end exactly at tend
    DT=min(dt,tend-t)
    ! Propagate one time step
    if( scheme == 'SL1')THEN
        Q = SemiLagrangian(Q,X,t,DT,dx,order=1,monotone=.False.)
    else if( scheme == 'SL1M')THEN
        Q = SemiLagrangian(Q,X,t,DT,dx,order=1,monotone=.True.)
    else if( scheme == 'SL2')THEN
        Q = SemiLagrangian(Q,X,t,DT,dx,order=2,monotone=.False.)
    else if( scheme == 'SL2M')THEN
        Q = SemiLagrangian(Q,X,t,DT,dx,order=2,monotone=.True.)
    else if( scheme == 'SL3')THEN
        Q = SemiLagrangian(Q,X,t,DT,dx,order=3,monotone=.False.)
    else if( scheme == 'SL3M')THEN
        Q = SemiLagrangian(Q,X,t,DT,dx,order=3,monotone=.True.)
    else
        scheme = 'default'
        STOP 
    endif

    t =t+DT
    nn=nn+1
    print*, 'iter [',nn,']  time [',t,']'
    CALL Savef(1,irec, Q )!save(Q,t,n)

   END DO
  call cpu_time ( EndTime )!start = clock()
  elapsed = EndTime - startTime
  print*, 'Time stepping took',elapsed,'seconds'
  
  print*, 'Turn off output for accurate timing'

CONTAINS
  LOGICAL FUNCTION FileOpen(unitout,output_file,lrec)
   INTEGER, INTENT(IN) :: unitout
   CHARACTER(LEN=*),INTENT(IN) :: output_file
   INTEGER, INTENT(IN) :: lrec
   INTEGER :: IERR
   FileOpen=.FALSE.
   OPEN(unitout,FILE=TRIM(output_file),ACCESS='DIRECT',FORM='UNFORMATTED'&
       ,STATUS='UNKNOWN',&
        ACTION='WRITE',RECl=lrec,IOSTAT=IERR)
   IF(IERR==0)FileOpen=.TRUE.
  END FUNCTION FileOpen
  
 FUNCTION copy(Q)  result(array)
   IMPLICIT NONE
   REAL   , INTENT(IN) :: Q(:)
   REAL                :: array(SIZE(Q))
   INTEGER             :: i
    DO i=LBOUND(Q,DIM=1),UBOUND(Q,DIM=1)! 0,N !for n in range(N):
          array(i)=Q(i)
    END DO
 END FUNCTION copy  


  LOGICAL FUNCTION FileClose(unitout)
   INTEGER, INTENT(IN) :: unitout
   CLOSE(unitout,STATUS='KEEP')
   FileClose=.TRUE.
  END FUNCTION FileClose

 FUNCTION linspace(xmin,xmax,num,endpoint)  result(array)
   IMPLICIT NONE
   REAL   , INTENT(IN)           :: xmin
   REAL   , INTENT(IN)           :: xmax
   INTEGER, INTENT(IN)           :: num
   LOGICAL, INTENT(IN), OPTIONAL :: endpoint
   REAL                :: array(1:num)
   REAL                :: delta
   INTEGER             :: i  
   LOGICAL opt 
   opt=PRESENT(endpoint)
   IF(.not.opt)THEN
      delta=(xmax-xmin)/(num-1)
      array(1)=xmin
      DO i=2,num
        array(i)=array(i-1)+delta
      END DO  
   ELSE
     IF(endpoint)THEN
       delta=(xmax-xmin)/(num-1)
       array(1)=xmin
       DO i=2,num
         array(i)=array(i-1)+delta
       END DO  
     ELSE
       delta=(xmax-xmin)/(num)
       array(1)=xmin
       DO i=2,num
         array(i)=array(i-1)+delta
       END DO  
     END IF
   END IF
 END FUNCTION linspace  


END PROGRAM Main
