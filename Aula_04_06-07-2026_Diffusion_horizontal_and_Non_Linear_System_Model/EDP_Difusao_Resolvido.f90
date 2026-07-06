MODULE Class_Fields
  IMPLICIT NONE
  PRIVATE
  INTEGER, PUBLIC      , PARAMETER  :: r8=8
  INTEGER, PUBLIC      , PARAMETER  :: r4=4
 REAL (KIND=r8) :: xMax=1
 REAL (KIND=r8) :: xMin=0
 REAL (KIND=r8),PUBLIC :: DeltaX=1.0
 REAL (KIND=r8),PUBLIC :: DeltaT=0.65  
 REAL (KIND=r8), PUBLIC :: C =2.0
 REAL (KIND=r8), PUBLIC :: Kdif =2.0e-5

 INTEGER, PUBLIC :: Idim
 REAL (KIND=r8), PUBLIC :: xb0=100.0
 REAL (KIND=r8), PUBLIC :: xf0=400.0
 REAL (KIND=r8), PUBLIC :: tb0 =0
 REAL (KIND=r8), PUBLIC :: tf0 =0
 REAL (KIND=r8), PUBLIC :: xxb
 REAL (KIND=r8), PUBLIC :: yyf
 REAL (KIND=r8), PUBLIC :: Area
 REAL (KIND=r8), PUBLIC :: alfa 
 REAL (KIND=r8), PUBLIC :: beta

  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: xa(:) 
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: ua(:) 
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: Deslc(:) 
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: u (:) 
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: um(:) 
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: up(:) 

  PUBLIC :: Init_Class_Fields

CONTAINS  
!-----------------------------------------------------------------------------------------
  SUBROUTINE Init_Class_Fields(alfa_in,beta_in,Kdif_in)
    IMPLICIT NONE
      REAL (KIND=r8), INTENT(in) :: alfa_in
      REAL (KIND=r8), INTENT(in) :: beta_in
      REAL (KIND=r8), INTENT(in) :: Kdif_in
      INTEGER :: i,xb(1),xf(1)
      REAL (KIND=r8):: t
      REAL (KIND=r8),ALLOCATABLE    :: diff(:)
      PRINT*,'DeltaX=',DeltaX,'DeltaT=',DeltaT,'CFL=',C*DeltaT/DeltaX
      Idim=1000
      alfa=alfa_in
      beta=beta_in
      Kdif=Kdif_in
      !Idim=  (xMax-Xmin)/DeltaX
      if (.not. allocated(u))  ALLOCATE (u(Idim))
      u=0.0
      if (.not. allocated(um))  ALLOCATE (um(Idim))
      um=0.0
      if (.not. allocated(up))  ALLOCATE (up(Idim))
      up=0.0
      if (.not. allocated(Deslc))   ALLOCATE (Deslc(Idim)) 
      Deslc=0.0
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
         IF(.TRUE.)THEN
            u(i) =273*MAX( COS((xa(i) - xa(Idim/2))/xa(Idim))-0.95,0.0)
         ELSE 
            IF(u(i) ==1.0)THEN
               u(i)=Area
            END IF
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
   FileName='DiffusionLinearConceitual1D'
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
    WRITE (UnitCtl,'(A           )')'vars 1'
    WRITE (UnitCtl,'(A           )')'uc 0 99 resultado da edol yc'
    WRITE (UnitCtl,'(A           )')'endvars'
    CLOSE (UnitCtl,STATUS='KEEP')
    CLOSE (UnitData,STATUS='KEEP')
    ok=0
 END FUNCTION SchemeWriteCtl
END MODULE Class_WritetoGrads




 MODULE ModAdvection
  USE Class_Fields, Only: DeltaT,DeltaX,Idim,r8,xa,tf0,tb0,yyf,&
                          xxb,xb0,xf0,Area,C,Deslc,alfa,beta,Kdif
   IMPLICIT NONE
   PRIVATE

  PUBLIC :: AnaliticFunction,Upstream,CentredTimeSpace,RungeKutta4

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



 FUNCTION CentredTimeSpace(Qm,Qc,Qp,t,dt,dx)  result(Qnp1)
   !"""
   !central in time, Forward in space advection scheme (1st order)
   !"""
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(IN) :: Qm(Idim)
   REAL(KIND=r8), INTENT(IN) :: Qc(Idim)
   REAL(KIND=r8), INTENT(IN) :: Qp(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   INTEGER             :: xb,xc,xf   
   REAL(KIND=r8)       :: Qnp1 (SIZE(Qc))
   INTEGER             :: N0
   INTEGER             :: N1
   INTEGER             :: i
   N0=LBOUND(Qc,DIM=1)
   N1=UBOUND(Qc,DIM=1)
   DO i=N0,N1
      CALL index(i,xb,xc,xf)
      !Qnp1(i) = Qc(xc) + 2*dt*Kdif*( Qc(xf) - 2*Qc(xc) + Qc(xb) )/(dx*dx)
      Qnp1(i) = Qc(xc) + 2*dt*Kdif*( Qm(xf) - 2*Qm(xc) + Qm(xb) )/(dx*dx)

      !Qnp1(i) = Qm(xc) + 2*dt*Kdif*( Qc(xf) - 2*Qc(xc) + Qc(xb) )/(dx*dx)
   END DO

 END FUNCTION CentredTimeSpace
!   


 FUNCTION Upstream(Qm,Qc,Qp,t,dt,dx)  result(Qnp1)
   !"""
   !Forward in time, Forward in space advection scheme (1st order)
   !"""
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(IN) :: Qm(Idim)
   REAL(KIND=r8), INTENT(IN) :: Qc(Idim)
   REAL(KIND=r8), INTENT(IN) :: Qp(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   INTEGER             :: xb,xc,xf   
   REAL(KIND=r8)       :: Qnp1 (SIZE(Qc))
   INTEGER             :: N0
   INTEGER             :: N1
   INTEGER             :: i
   N0=LBOUND(Qc,DIM=1)
   N1=UBOUND(Qc,DIM=1)
    IF( Kdif>0 )THEN
        DO i=N0,N1
           CALL index(i,xb,xc,xf)
           Qnp1(i) = Qc(xc) + dt*Kdif*( Qc(xf) - 2*Qc(xc) + Qc(xb) )/(dx*dx)
        END DO
    ELSE
        DO i=N0,N1
           CALL index(i,xb,xc,xf)
           Qnp1(i) = Qc(xc) + dt*Kdif*( Qc(xf) - 2*Qc(xc) + Qc(xb) )/(dx*dx)
        END DO
    END IF

 END FUNCTION Upstream

!   !************************************************************************************

 FUNCTION RungeKutta4(EDO,Qp,Q,Qm,t,dt,dx,filter)  result(Qnp1)
   ! """
   ! Runge-Kutta 4 time integration (4th order)
   ! """
   IMPLICIT NONE
   CHARACTER(LEN=*), INTENT(IN) :: EDO
   REAL(KIND=r8), INTENT(INOUT) :: Qp(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Q (Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Qm(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx   
   CHARACTER(LEN=*), INTENT(IN) :: filter
   REAL(KIND=r8):: k1 (SIZE(Q))
   REAL(KIND=r8):: k2 (SIZE(Q))
   REAL(KIND=r8):: k3 (SIZE(Q))
   REAL(KIND=r8):: k4 (SIZE(Q))
   REAL(KIND=r8):: Qnp1(SIZE(Q))
   INTEGER :: test


   IF(EDO=='CentredSpace')THEN
      k1 = CentredSpace(Q            , t          , dt,dx)
      k2 = CentredSpace(Q+ 0.5*dt*k1 , t + 0.5*dt , dt,dx)
      k3 = CentredSpace(Q+ 0.5*dt*k2 , t + 0.5*dt , dt,dx)
      k4 = CentredSpace(Q+     dt*k3 , t +     dt , dt,dx)
   ELSE IF(EDO=='UpwindSpace')THEN
       k1 = UpwindSpace(Q            , t          , dt,dx)
       k2 = UpwindSpace(Q+ 0.5*dt*k1 , t + 0.5*dt , dt,dx)
       k3 = UpwindSpace(Q+ 0.5*dt*k2 , t + 0.5*dt , dt,dx)
       k4 = UpwindSpace(Q+     dt*k3 , t +     dt , dt,dx)   
   END IF
   IF(TRIM(filter) =='noFilter')THEN
      Qnp1=0.0  
      Qnp1 = Q + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )
   ELSE IF(TRIM(filter) =='FilterRA')THEN
      Qnp1=0.0  
      Qnp1 = Q + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )
      test=Filter_RA(Qnp1,Q,Qm)
      Qnp1 = Q + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )
   ELSE IF(TRIM(filter) =='FilterRAW')THEN
      Qnp1=0.0  
      Qnp1 = Q + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )
      test=Filter_RAW(Qnp1,Q,Qm)
      Qp = Qnp1
      Qnp1 = Q + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )
      Qnp1=0.5*(Qp+Qnp1)
   ELSE
      Qnp1=0.0  
      Qnp1 = Q + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )
   END IF
 END FUNCTION RungeKutta4   

!************************************************************************************


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

   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)

   DO i=N0,N1
      CALL index(i,xb,xc,xf)
      rhs(i) = Kdif*(Q(xf) -2*Q(xc) + Q(xb) )/(dx*dx)
   END DO

 END FUNCTION CentredSpace  

!   !************************************************************************************
 
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
   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)
   IF( Kdif > 0.0 )THEN
      DO i=N0,N1
          CALL index(i,xb,xc,xf)
           rhs(i) = Kdif*(Q(xf) -2*Q(xc) + Q(xb) )/(dx*dx)
      END DO 
    ELSE
      DO i =N0,N1
          CALL index(i,xb,xc,xf)
          rhs(i) = Kdif*(Q(xf) -2*Q(xc) + Q(xb) )/(dx*dx)
      END DO
    END IF

 END FUNCTION UpwindSpace  
 !------------------------------------------------------------------------------------------
 FUNCTION Filter_RA(Qf,Qc,Qb)  RESULT (ok)
    IMPLICIT NONE
   REAL(KIND=r8), INTENT(INOUT) :: Qf(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Qc(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Qb(Idim)
   INTEGER             :: N0
   INTEGER             :: N1
   INTEGER             ::   xb,xc,xf      
   INTEGER :: i
   INTEGER :: ok
   N0=LBOUND(Qc,DIM=1)
   N1=UBOUND(Qc,DIM=1)
   DO i=N0,N1
      CALL index(i,xb,xc,xf)
      Deslc(xc) = alfa*(Qb(xc) - 2.0*Qc(xc) + Qf(xc) )
      Qc(xc) = Qc(xc) + Deslc(xc)
   END DO
    ok=0
 END FUNCTION Filter_RA
 
!   !************************************************************************************

 FUNCTION Filter_RAW(Qf,Qc,Qb)  RESULT (ok)
    IMPLICIT NONE
   REAL(KIND=r8), INTENT(INOUT) :: Qf(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Qc(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Qb(Idim)
   INTEGER             :: N0
   INTEGER             :: N1
   INTEGER             ::   xb,xc,xf      
   INTEGER :: i
   INTEGER :: ok
   N0=LBOUND(Qc,DIM=1)
   N1=UBOUND(Qc,DIM=1)
   DO i=N0,N1
       CALL index(i,xb,xc,xf)
       Deslc(xc) = alfa*(Qb(xc) - 2.0*Qc(xc) + Qf(xc) )
       Qc(xc) = Qc(xc) + Deslc(xc)
       Qf(xc) = Qf(xc) + Deslc(xc)*(beta-1.0)
    END DO
    ok=0
 END FUNCTION Filter_RAW

!   !************************************************************************************

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


END MODULE ModAdvection
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PROGRAM  Main
  USE Class_Fields, Only : Init_Class_Fields,DeltaT,DeltaX,ua,u,um,up,r8,Idim,C
  USE ModAdvection, Only : AnaliticFunction,Upstream,CentredTimeSpace,RungeKutta4
  USE Class_WritetoGrads, Only : SchemeWriteCtl,SchemeWriteData,InitClass_WritetoGrads
   IMPLICIT NONE
   REAL(KIND=r8)               :: tend = 800.!End time of simulation
   CHARACTER(LEN=10) :: scheme      = 'CTCS'  ! Advection scheme. Possible values: UPSTREAM, CTCS,RK4CS,RK4CS_RA,RK4CS_RAW
   REAL(KIND=r8)               :: alfa = 0.3
   REAL(KIND=r8)               :: beta = 0.6 ! 0.5<  beta  <=1
   REAL(KIND=r8)               :: Kdif =5.0e-5
   INTEGER :: irec_err,unit2

      CALL Init()
      CALL Run(irec_err,unit2)
      CALL Finalize()

  CONTAINS

  SUBROUTINE Init()
      CALL Init_Class_Fields(alfa,beta,Kdif)
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
          if( TRIM(scheme) == 'RK4CS')THEN
             up = RungeKutta4('CentredSpace',up,u,um,t,DT,DeltaX,'noFilter')
          else if( TRIM(scheme) == 'RK4CS_RA')THEN
             up = RungeKutta4('CentredSpace',up,u,um,t,DT,DeltaX,'FilterRA')
          else if( TRIM(scheme) == 'RK4CS_RAW')THEN
             up = RungeKutta4('CentredSpace',up,u,um,t,DT,DeltaX,'FilterRAW')
          else if(TRIM(scheme)=='UPSTREAM')THEN
             up = Upstream(um,u,up,t,DT,DeltaX)
          else if(TRIM(scheme)=='CTCS')THEN
             up = CentredTimeSpace(um,u,up,t,DT,DeltaX)
          else
             scheme = 'default'
             !### Comment/Uncomment desired scheme
             up = Upstream(um,u,up,t,DT,DeltaX)
          endif


          err=err+SUM((u-um)**2)

          t =t+DT
          um=u
	  u=up
          up=up
          test=SchemeWriteData(u  ,irec)
          print*, 'iter [',nn,']  time [',t,']'
        END DO
        test=SchemeWriteCtl(nn)

   PRINT*,'scheme=',TRIM(scheme),' err=',err/nn,' DeltaX=',DeltaX,' DeltaT=',DeltaT,' CFL=',Kdif*DeltaT/DeltaX

  END SUBROUTINE Run
!     
  SUBROUTINE Finalize()
      
  END SUBROUTINE Finalize
END PROGRAM  Main
