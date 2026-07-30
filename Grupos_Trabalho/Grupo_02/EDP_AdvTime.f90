MODULE Class_Fields
  IMPLICIT NONE
  PRIVATE
  INTEGER, PUBLIC      , PARAMETER  :: r8=8
  INTEGER, PUBLIC      , PARAMETER  :: r4=4
 REAL (KIND=r8) :: xMax=1
 REAL (KIND=r8) :: xMin=0
 REAL (KIND=r8),PUBLIC :: DeltaX=1.0
 REAL (KIND=r8),PUBLIC :: DeltaT=0.25  
 REAL (KIND=r8), PUBLIC :: C =2.0
 !                     'CFL=',C*DeltaT/DeltaX  <==> 0.2  ou 0.7
 INTEGER, PUBLIC :: Idim
 REAL (KIND=r8), PUBLIC :: xb0=100.0
 REAL (KIND=r8), PUBLIC :: xf0=400.0
 REAL (KIND=r8), PUBLIC :: tb0 =0
 REAL (KIND=r8), PUBLIC :: tf0 =0
 REAL (KIND=r8), PUBLIC :: xxb
 REAL (KIND=r8), PUBLIC :: yyf
 REAL (KIND=r8), PUBLIC :: Area

  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: xa(:) 
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: ua(:) 

  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: u(:) 
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: um(:) 
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: up(:) 

  PUBLIC :: Init_Class_Fields

CONTAINS  
!-----------------------------------------------------------------------------------------
  SUBROUTINE Init_Class_Fields()
    IMPLICIT NONE
      INTEGER :: i,xb(1),xf(1)
      REAL (KIND=r8):: t
      REAL (KIND=r8),ALLOCATABLE    :: diff(:)
      PRINT*,'DeltaX=',DeltaX,'DeltaT=',DeltaT,'CFL=',C*DeltaT/DeltaX
      Idim=1000
      !Idim=  (xMax-Xmin)/DeltaX
      if (.not. allocated(u))  ALLOCATE (u(Idim))
      u=0.0
      if (.not. allocated(um))  ALLOCATE (um(Idim))
      um=0.0
      if (.not. allocated(up))  ALLOCATE (up(Idim))
      up=0.0

      if (.not. allocated(ua)) ALLOCATE (ua(Idim))
      ua=0.0
      if (.not. allocated(xa)) ALLOCATE (xa(Idim))
      if (.not. allocated(diff)) ALLOCATE (diff(Idim))

      DO i=1,Idim
         xa(i)=(i-1)*DeltaX
      END DO
      xb0=xa(Idim)/4.0
      xf0=xa(Idim)/2.0
      tb0 =0
      tf0 =0
      t=0
      xxb= xb0 + C*(t-tb0)
      yyf= xf0 + C*(t-tf0)
      DO i=1,Idim
         IF(xa(i) >xxb .and. xa(i) <yyf)THEN
            u(i)=1.0
         ELSE
            u(i)=0.0
         END IF
      END DO
      diff=ABS(xa-xxb)
      xb=MINLOC(diff) 
      diff=ABS(xa-yyf)
      xf=MINLOC(diff) 
      Area=( u(xf(1))-u(xb(1)-1))*(xa(xf(1))-xa(xb(1)))/(xf(1)-xb(1)+1)
      DO i=1,Idim
         IF(u(i) ==1.0)THEN
            u(i)=Area
         END IF
      END DO
      ua=u
      um=u
      up=u
  END SUBROUTINE Init_Class_Fields
!------------------------------------------------------------------------------------------
END MODULE Class_Fields


 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODULE Class_WritetoGrads
 USE Class_Fields, Only: Idim,xa
 IMPLICIT NONE
 PRIVATE
 INTEGER, PUBLIC      , PARAMETER  :: r8=8
 INTEGER, PUBLIC      , PARAMETER  :: r4=4
 INTEGER                    , PARAMETER :: UnitData=1
 INTEGER                    , PARAMETER :: UnitCtl=2
 CHARACTER (LEN=400)                   :: FileName
 LOGICAL                                            :: CtrlWriteDataFile
 PUBLIC :: SchemeWriteCtl
 PUBLIC :: SchemeWriteData
 PUBLIC :: InitClass_WritetoGrads
CONTAINS
 SUBROUTINE InitClass_WritetoGrads()
    IMPLICIT NONE
   FileName=''
   FileName='AdvecLinearConceitual1D'
   CtrlWriteDataFile=.TRUE.
 END SUBROUTINE InitClass_WritetoGrads

 FUNCTION SchemeWriteData(vars,irec)  RESULT (ok)
    IMPLICIT NONE
    REAL (KIND=r8), INTENT (INOUT) :: vars(Idim)
    INTEGER       , INTENT (INOUT) :: irec
    INTEGER        :: ok
    INTEGER        :: lrec
    REAL (KIND=r4) :: Yout(Idim)
    IF(CtrlWriteDataFile)INQUIRE (IOLENGTH=lrec) Yout
    IF(CtrlWriteDataFile)OPEN(UnitData,FILE=TRIM(FileName)//'.bin',&
    FORM='UNFORMATTED', ACCESS='DIRECT', STATUS='UNKNOWN', &
    ACTION='WRITE',RECL=lrec)
    ok=1
    CtrlWriteDataFile=.FALSE.
    Yout=REAL(vars(1:Idim),KIND=r4)
    irec=irec+1
    WRITE(UnitData,rec=irec)Yout
     ok=0
 END FUNCTION SchemeWriteData

 FUNCTION SchemeWriteCtl(nrec)  RESULT (ok)
    IMPLICIT NONE
    INTEGER, INTENT (IN) :: nrec
    INTEGER             :: ok,i
    ok=1
   OPEN(UnitCtl,FILE=TRIM(FileName)//'.ctl',FORM='FORMATTED', &
   ACCESS='SEQUENTIAL',STATUS='UNKNOWN',ACTION='WRITE')
    WRITE (UnitCtl,'(A6,A           )')'dset ^',TRIM(FileName)//'.bin'
    WRITE (UnitCtl,'(A                 )')'title  EDO'
    WRITE (UnitCtl,'(A                 )')'undef  -9999.9'
    WRITE (UnitCtl,'(A6,I8,A8   )')'xdef  ',Idim,' levels '
    WRITE (UnitCtl,'(10F16.10   )')(xa(i),i=1,Idim)
    WRITE (UnitCtl,'(A                  )')'ydef  1 linear  -1.27 1'
    WRITE (UnitCtl,'(A6,I6,A25   )')'tdef  ',nrec,' linear  00z01jan0001 1hr'
    WRITE (UnitCtl,'(A20             )')'zdef  1 levels 1000 '
    WRITE (UnitCtl,'(A           )')'vars 2'
    WRITE (UnitCtl,'(A           )')'uc 0 99 resultado da edol yc'
    WRITE (UnitCtl,'(A           )')'ua 0 99 solucao analitica ya'
    WRITE (UnitCtl,'(A           )')'endvars'
    CLOSE (UnitCtl,STATUS='KEEP')
    CLOSE (UnitData,STATUS='KEEP')
    ok=0
 END FUNCTION SchemeWriteCtl
END MODULE Class_WritetoGrads




 MODULE ModAdvection
  USE Class_Fields, Only: DeltaT,DeltaX,Idim,r8,xa,tf0,tb0,yyf,xxb,xb0,xf0,Area,C
   IMPLICIT NONE
   PRIVATE

  PUBLIC :: AnaliticFunction,RungeKutta6,RungeKutta4,RungeKutta3,RungeKutta2,Upstream

CONTAINS

!   
  FUNCTION AnaliticFunction(termX,ua,it)  RESULT (ok)

      REAL(KIND=r8), INTENT(INOUT) :: termX(Idim)
      REAL(KIND=r8), INTENT(IN   ) :: ua(Idim)
      INTEGER, INTENT(IN   ) :: it
      INTEGER          :: i2,xb,xc,xf,i
      INTEGER         :: ok
      REAL(KIND=r8)    :: t
      t=(it)*DeltaT

      yyf= xf0 + C*(t-tf0)
      IF(yyf >= xa(Idim))THEN
         xf0=0.0
         yyf=xf0
         tf0=t
      END IF 

      xxb= xb0 + C*(t-tb0)
      IF(xxb >= xa(Idim))THEN
         xb0=0.0
         xxb=xb0
         tb0=t
      END IF 
      IF(xf0 <= xb0 .and. yyf <= xxb) THEN  
         DO i=1,Idim
            IF(xa(i) > xxb )THEN
               termX(i)=Area
            ELSE IF( xa(i) < yyf )THEN
               termX(i)=Area
            ELSE
               termX(i)=0.0
            END IF
         END DO
      ELSE
         DO i=1,Idim
            IF(xa(i) > xxb .and. xa(i) < yyf)THEN
               termX(i)=Area
            ELSE
               termX(i)=0.0
            END IF
         END DO
      END IF
    ok=0
   END FUNCTION AnaliticFunction

 FUNCTION Solve_4thCS(Q,t,dt,dx)  result(rhs)
   ! """
   ! Centred in space right-hand-side computation (4nd order)
   ! """
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   INTEGER                   :: xb3,xb2,xb,xc,xf,xf2,xf3
   REAL(KIND=r8)             :: dudx (SIZE(Q))
   REAL(KIND=r8)             :: rhs  (SIZE(Q))
   INTEGER                   :: N0
   INTEGER                   :: N1
   INTEGER                   :: i
   REAL(KIND=r8)               :: Kdif   = 2.0e-1_r8

   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)

   DO i=N0,N1
      CALL index2(i,xb3,xb2,xb,xc,xf,xf2,xf3)
      dudx(xc)  = (-Q(xf2) + 8.0*Q(xf) - 8.0*Q(xb) + Q(xb2))/((12*dx)) 

      rhs(i) =  - (C)*( dudx(xc)) + Kdif*( Q(xf) - 2*Q(xc) + Q(xb) )/(dx*dx)
   END DO

 END FUNCTION Solve_4thCS 

 FUNCTION UpwindSpace(Q,t,dt,dx)  result(rhs)
   ! """
   ! Forward in space right-hand-side computation (1st order)
   ! """
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   INTEGER             ::   xb,xc,xf      
   REAL(KIND=r8)       :: rhs (SIZE(Q))
   INTEGER             :: N0
   INTEGER             :: N1
   INTEGER             :: i
   REAL(KIND=r8)               :: Kdif   = 2.0e-1_r8

   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)
   IF( C > 0 )THEN
      DO i=N0,N1
          CALL index(i,xb,xc,xf)
          rhs(i) = - ( C*Q(xc) - C*Q(xb) )/dx + Kdif*( Q(xf) - 2*Q(xc) + Q(xb) )/(dx*dx)
      END DO 
    ELSE
      DO i =N0,N1
          CALL index(i,xb,xc,xf)
          rhs(i) = -( C*Q(xf) - C*Q(xc))/dx  + Kdif*( Q(xf) - 2*Q(xc) + Q(xb) )/(dx*dx)
      END DO
    END IF

 END FUNCTION UpwindSpace  

 FUNCTION CentredSpace(Q,t,dt,dx)  result(rhs)
   ! """
   ! Centred in space right-hand-side computation (2nd order)
   ! """
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   INTEGER                   ::   xb,xc,xf   
   REAL(KIND=r8)             :: rhs (SIZE(Q))
   INTEGER                   :: N0
   INTEGER                   :: N1
   INTEGER                   :: i
   REAL(KIND=r8)               :: Kdif   = 2.0e-1_r8

   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)
   !
   !  du          du
   ! --- = - C * ---
   !  dt          dx
   !
   DO i=N0,N1
      CALL index(i,xb,xc,xf)
      rhs(i) = -( C*Q(xf) - C*Q(xb) )/(2*dx) + Kdif*( Q(xf) - 2*Q(xc) + Q(xb) )/(dx*dx)
   END DO

 END FUNCTION CentredSpace  

 FUNCTION RungeKutta1(EDO,Q,t,dt,dx)  result(Qnp1)
   ! """
   ! Runge-Kutta 4 time integration (4th order)
   ! """
   IMPLICIT NONE
   CHARACTER(LEN=*), INTENT(IN) :: EDO
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   REAL(KIND=r8):: k1 (SIZE(Q))
   REAL(KIND=r8):: Qnp1(SIZE(Q))
   IF(EDO=='CentredSpace')THEN
      k1 = CentredSpace(Q            , t          , dt,dx)
   ELSE IF(EDO=='UpwindSpace')THEN
       k1 = UpwindSpace(Q            , t          , dt,dx)
   ELSE IF(EDO=='Centred4thCS')THEN
       k1 = Solve_4thCS(Q            , t          , dt,dx)
   END IF
    Qnp1=0.0  
    Qnp1 = Q + 1./1.*dt*( k1)

!************************************************************************************

 END FUNCTION RungeKutta1


 FUNCTION RungeKutta2(EDO,Q,t,dt,dx)  result(Qnp1)
   ! """
   ! Runge-Kutta 4 time integration (4th order)
   ! """
   IMPLICIT NONE
   CHARACTER(LEN=*), INTENT(IN) :: EDO
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   REAL(KIND=r8):: k1 (SIZE(Q))
   REAL(KIND=r8):: k2 (SIZE(Q))
   REAL(KIND=r8):: Qnp1(SIZE(Q))
   IF(EDO=='CentredSpace')THEN
      k1 = CentredSpace(Q             , t            , dt,dx)
      k2 = CentredSpace(Q+  1.0*dt*k1 , t +   1.0*dt , dt,dx)
   ELSE IF(EDO=='UpwindSpace')THEN
       k1 = UpwindSpace(Q             , t            , dt,dx)
       k2 = UpwindSpace(Q+  1.0*dt*k1 , t +   1.0*dt , dt,dx)   
   ELSE IF(EDO=='Centred4thCS')THEN
       k1 = Solve_4thCS(Q             , t            , dt,dx)
       k2 = Solve_4thCS(Q+  1.0*dt*k1 , t +   1.0*dt , dt,dx)   
   END IF
    Qnp1=0.0  
    Qnp1 = Q + 1./2.*dt*( k1 + k2)! 0.5 + 0.5

!************************************************************************************

 END FUNCTION RungeKutta2

 FUNCTION RungeKutta3(EDO,Q,t,dt,dx)  result(Qnp1)
   ! """
   ! Runge-Kutta 4 time integration (4th order)
   ! """
   IMPLICIT NONE
   CHARACTER(LEN=*), INTENT(IN) :: EDO
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   REAL(KIND=r8):: k1 (SIZE(Q))
   REAL(KIND=r8):: k2 (SIZE(Q))
   REAL(KIND=r8):: k3 (SIZE(Q))
   REAL(KIND=r8):: Qnp1(SIZE(Q))
   IF(EDO=='CentredSpace')THEN
      k1 = CentredSpace(Q            , t          , dt,dx)
      k2 = CentredSpace(Q+ 0.5*dt*k1 , t + 0.5*dt , dt,dx)
      k3 = CentredSpace(Q+ 1.0*dt*k2 , t + 1.0*dt , dt,dx)
   ELSE IF(EDO=='UpwindSpace')THEN
       k1 = UpwindSpace(Q            , t          , dt,dx)
       k2 = UpwindSpace(Q+ 0.5*dt*k1 , t + 0.5*dt , dt,dx)
       k3 = UpwindSpace(Q+ 1.0*dt*k2 , t + 1.0*dt , dt,dx)   
   ELSE IF(EDO=='Centred4thCS')THEN
       k1 = Solve_4thCS(Q            , t          , dt,dx)
       k2 = Solve_4thCS(Q+ 0.5*dt*k1 , t + 0.5*dt , dt,dx)   
       k3 = Solve_4thCS(Q+ 1.0*dt*k2 , t + 1.0*dt , dt,dx)   
   END IF
    Qnp1=0.0  
    Qnp1 = Q + 1./4.*dt*( k1 + 2.*k2 + k3  )    != ! 0.25 + 0.5 + 0.25

!************************************************************************************

 END FUNCTION RungeKutta3


 
 FUNCTION RungeKutta4(EDO,Q,t,dt,dx)  result(Qnp1)
   ! """
   ! Runge-Kutta 4 time integration (4th order)
   ! """
   IMPLICIT NONE
   CHARACTER(LEN=*), INTENT(IN) :: EDO
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   REAL(KIND=r8):: k1 (SIZE(Q))
   REAL(KIND=r8):: k2 (SIZE(Q))
   REAL(KIND=r8):: k3 (SIZE(Q))
   REAL(KIND=r8):: k4 (SIZE(Q))
   REAL(KIND=r8):: Qnp1(SIZE(Q))
   IF(EDO=='CentredSpace')THEN
      k1 = CentredSpace(Q             , t           , dt,dx)
      k2 = CentredSpace(Q+ 0.5*dt*k1 , t + 0.5*dt , dt,dx)
      k3 = CentredSpace(Q+ 0.5*dt*k2 , t + 0.5*dt , dt,dx)
      k4 = CentredSpace(Q+ 1.00*dt*k3 , t + 1.00*dt , dt,dx)
   ELSE IF(EDO=='UpwindSpace')THEN
       k1 = UpwindSpace(Q             , t           , dt,dx)
       k2 = UpwindSpace(Q+ 0.35*dt*k1 , t +  0.35*dt, dt,dx)
       k3 = UpwindSpace(Q+ 0.75*dt*k2 , t +  0.75*dt, dt,dx)
       k4 = UpwindSpace(Q+ 1.00*dt*k3 , t +  1.00*dt, dt,dx)   
   ELSE IF(EDO=='Centred4thCS')THEN
       k1 = Solve_4thCS(Q            , t            , dt,dx)
       k2 = Solve_4thCS(Q+ 0.35*dt*k1, t +  0.35*dt , dt,dx)	
       k3 = Solve_4thCS(Q+ 0.75*dt*k2, t +  0.75*dt , dt,dx)	
       k4 = Solve_4thCS(Q+ 1.00*dt*k3, t +  1.00*dt , dt,dx)   
   END IF
    Qnp1=0.0  
                                                    !  1    2	 2    1     1+2+2+1
    Qnp1 = Q + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 ) ! ---+ ---+ ---+ --- = --------
                                                    !  6    6	 6    6        6
!************************************************************************************

 END FUNCTION RungeKutta4   

 FUNCTION RungeKutta6(EDO,Q,t,dt,dx)  result(Qnp1)
   ! """
   ! Runge-Kutta 4 time integration (4th order)
   ! """
   IMPLICIT NONE
   CHARACTER(LEN=*), INTENT(IN) :: EDO
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   REAL(KIND=r8):: k1 (SIZE(Q))
   REAL(KIND=r8):: k2 (SIZE(Q))
   REAL(KIND=r8):: k3 (SIZE(Q))
   REAL(KIND=r8):: k4 (SIZE(Q))
   REAL(KIND=r8):: k5 (SIZE(Q))
   REAL(KIND=r8):: k6 (SIZE(Q))
   REAL(KIND=r8):: Qnp1(SIZE(Q))
   IF(EDO=='CentredSpace')THEN
      k1 = CentredSpace(Q              , t            , dt,dx)
      k2 = CentredSpace(Q+ 0.200*dt*k1 , t + 0.200*dt , dt,dx)
      k3 = CentredSpace(Q+ 0.400*dt*k2 , t + 0.400*dt , dt,dx)
      k4 = CentredSpace(Q+ 0.600*dt*k3 , t + 0.600*dt , dt,dx)
      k5 = CentredSpace(Q+ 0.800*dt*k4 , t + 0.800*dt , dt,dx)
      k6 = CentredSpace(Q+ 1.000*dt*k5 , t + 1.000*dt , dt,dx)
   ELSE IF(EDO=='UpwindSpace')THEN
       k1 = UpwindSpace(Q              , t            , dt,dx)
       k2 = UpwindSpace(Q+ 0.200*dt*k1 , t + 0.200*dt , dt,dx)
       k3 = UpwindSpace(Q+ 0.400*dt*k2 , t + 0.400*dt , dt,dx)
       k4 = UpwindSpace(Q+ 0.600*dt*k3 , t + 0.600*dt , dt,dx)
       k5 = UpwindSpace(Q+ 0.800*dt*k4 , t + 0.800*dt , dt,dx)
       k6 = UpwindSpace(Q+ 1.000*dt*k5 , t + 1.000*dt , dt,dx)
   ELSE IF(EDO=='Centred4thCS')THEN
       k1 = Solve_4thCS(Q              , t            , dt,dx)
       k2 = Solve_4thCS(Q+ 0.200*dt*k1 , t + 0.200*dt , dt,dx)
       k3 = Solve_4thCS(Q+ 0.400*dt*k2 , t + 0.400*dt , dt,dx)
       k4 = Solve_4thCS(Q+ 0.600*dt*k3 , t + 0.600*dt , dt,dx)
       k5 = Solve_4thCS(Q+ 0.800*dt*k4 , t + 0.800*dt , dt,dx)
       k6 = Solve_4thCS(Q+ 1.000*dt*k5 , t + 1.000*dt , dt,dx)

   END IF
    Qnp1=0.0  
    Qnp1 = Q + (1./14.)*dt*( k1 + 4.*k2 + 2.*k3 + 2.*k4 + 4.*k5 + k6 )
!
                           !  1    2    4    4     2    1       1+2+4+4+2+1
                           ! ---+ ---+ ---+ --- + ---+ --- =   ---------------
                           !  14   14  14    14   14   14         14
!
!************************************************************************************

 END FUNCTION RungeKutta6   



 FUNCTION Upstream(Q,t,dt,dx)  result(Qnp1)
   !"""
   !Forward in time, Forward in space advection scheme (1st order)
   !"""
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   INTEGER             :: xb,xc,xf   
   REAL(KIND=r8)       :: Qnp1 (SIZE(Q))
   INTEGER             :: N0
   INTEGER             :: N1
   INTEGER             :: i
   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)
    IF( C>0 )THEN
        DO i=N0,N1
           CALL index(i,xb,xc,xf)
           Qnp1(i) = Q(xc) - dt*( C*Q(xc) - C*Q(xb) )/dx
        END DO
    ELSE
        DO i=N0,N1
           CALL index(i,xb,xc,xf)
           Qnp1(i) = Q(xc) - dt*( C*Q(xf) - C*Q(xc) )/dx
        END DO
    END IF

 END FUNCTION Upstream
!

 
!   
   SUBROUTINE index(i,xb,xc,xf)
      IMPLICIT NONE
      INTEGER, INTENT(IN   ) :: i
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

   SUBROUTINE index2(i,xb3,xb2,xb,xc,xf,xf2,xf3)
      IMPLICIT NONE
      INTEGER, INTENT(IN   ) :: i
      INTEGER, INTENT(OUT  ) :: xb3,xb2,xb,xc,xf,xf2,xf3
      IF(i==1) THEN
        xb3=Idim-2
        xb2=Idim-1
        xb=Idim
        xc=i
        xf=i+1
        xf2=i+2
        xf3=i+3
      ELSE IF(i==2)THEN
        xb3=Idim-1
        xb2=Idim
        xb=i-1
        xc=i
        xf=i+1
        xf2=i+2
        xf3=i+3
      ELSE IF(i==3)THEN
        xb3=Idim
        xb2=i-2
        xb=i-1
        xc=i
        xf=i+1
        xf2=i+2
        xf3=i+3
      ELSE IF(i==Idim)THEN
        xb3=Idim-3
        xb2=Idim-2
        xb=Idim-1
        xc=i
        xf=1
        xf2=2
        xf3=3
      ELSE IF(i==Idim-1)THEN
        xb3=Idim-4
        xb2=Idim-3
        xb=Idim-2
        xc=i
        xf=Idim
        xf2=1
        xf3=2
      ELSE IF(i==Idim-2)THEN
        xb3=Idim-5
        xb2=Idim-4
        xb=Idim-3
        xc=i
        xf=Idim-1
        xf2=Idim
        xf3=1
      ELSE
        xb3=i-3
        xb2=i-2
        xb=i-1
        xc=i
        xf=i+1
        xf2=i+2
        xf3=i+3
      END IF
   END SUBROUTINE index2

END MODULE ModAdvection
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PROGRAM  Main
  USE Class_Fields, Only : Init_Class_Fields,DeltaT,DeltaX,ua,u,um,up,r8,Idim,C
  USE ModAdvection, Only : AnaliticFunction,Upstream,RungeKutta6,RungeKutta4,RungeKutta3,RungeKutta2
  USE Class_WritetoGrads, Only : SchemeWriteCtl,SchemeWriteData,InitClass_WritetoGrads
   IMPLICIT NONE
   REAL               :: tend = 2000.!End time of simulation
   CHARACTER(LEN=10) :: scheme      = 'RK4CS4'!'RK6CS4'  ! Advection scheme. Possible values:
                                               !scheme      =upstream 
                                               !scheme      =EULFS
                                               !scheme      =RK4CS
                                               !scheme      =RK3CS
                                               !scheme      =RK6CS 
                                               !scheme      =RK6FS
                                               !scheme      =RK6CS4 
                                               !scheme      =RK4CS4
                                               !scheme      =RK3CS4
                                               !scheme      =RK3CS4
                                               !scheme      =RK1CS4

   INTEGER :: irec_err,unit2

      CALL Init()
      CALL Run(irec_err,unit2)
      CALL Finalize()

  CONTAINS

  SUBROUTINE Init()
      CALL Init_Class_Fields()
      CALL InitClass_WritetoGrads
  END SUBROUTINE Init

  SUBROUTINE Run(irec_err,unit)
      INTEGER, INTENT(INOUT) :: irec_err
      INTEGER, INTENT(IN   ) :: unit
      REAL (KIND=r8) :: termX(Idim)
      REAL (KIND=r8) :: termXa(Idim)
      REAL (KIND=r8) :: err,DT,dx,t
      INTEGER :: i,nn
      INTEGER :: it,lrec,irec,test
      irec=0
      err=0
      test=SchemeWriteData(u ,irec)
      test=SchemeWriteData(ua,irec)
      ! Time stepping
       t = 0.
       nn = 0
       DO WHILE  (t < tend )      
          nn=nn+1
          DO i=1,Idim
             termXa(i)=0.0
          END DO 
          ! To make simulation end exactly at tend
          DT=min(DeltaT,tend-t)
          ! Propagate one time step 
            !4th order
          if( scheme == 'RK6CS')THEN
             up = RungeKutta6('CentredSpace',u,t,DT,DeltaX)
          else if( scheme == 'RK6CS4')THEN
             up = RungeKutta6('Centred4thCS' ,u,t,DT,DeltaX)
          else if( scheme == 'RK6FS')THEN
             up = RungeKutta6('UpwindSpace' ,u,t,DT,DeltaX)
            !4th order
          else if( scheme == 'RK4CS')THEN
             up = RungeKutta4('CentredSpace',u,t,DT,DeltaX)
          else if( scheme == 'RK4CS4')THEN
             up = RungeKutta4('Centred4thCS' ,u,t,DT,DeltaX)
          else if( scheme == 'RK4FS')THEN
             up = RungeKutta4('UpwindSpace' ,u,t,DT,DeltaX)
              !3th order
          else if( scheme == 'RK3CS')THEN
             up = RungeKutta3('CentredSpace',u,t,DT,DeltaX)
          else if( scheme == 'RK3CS4')THEN
             up = RungeKutta3('Centred4thCS',u,t,DT,DeltaX)
          else if( scheme == 'RK3FS')THEN
             up = RungeKutta3('UpwindSpace' ,u,t,DT,DeltaX)
          else if( scheme == 'RK2CS')THEN
            !2th order
             up = RungeKutta2('CentredSpace',u,t,DT,DeltaX)
          else if( scheme == 'RK2CS4')THEN
             up = RungeKutta2('Centred4thCS',u,t,DT,DeltaX)
          else if( scheme == 'RK2FS')THEN
             up = RungeKutta2('UpwindSpace' ,u,t,DT,DeltaX)
          else
             scheme = 'default'
             !### Comment/Uncomment desired scheme
             up = Upstream(u,t,DT,DeltaX)
          endif

          test=AnaliticFunction(termXa,ua,nn)
          ua=termXa

          test=SchemeWriteData(up ,irec)
          test=SchemeWriteData(ua ,irec)

          err=err+SUM((u-ua)**2)

          t =t+DT
	  u=up
          print*, 'iter [',nn,']  time [',t,']'
        END DO
        test=SchemeWriteCtl(nn)

   PRINT*,'err=',err/nn,'DeltaX=',DeltaX,'DeltaT=',DeltaT,'CFL=',C*DeltaT/DeltaX

  END SUBROUTINE Run
!     
  SUBROUTINE Finalize()
      
  END SUBROUTINE Finalize
END PROGRAM  Main
