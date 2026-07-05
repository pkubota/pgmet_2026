!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODULE Class_Fields
  IMPLICIT NONE
  PRIVATE
  INTEGER, PUBLIC            , PARAMETER  :: r8=8
  INTEGER, PUBLIC            , PARAMETER  :: r4=4
  REAL (KIND=r8),PUBLIC ,ALLOCATABLE :: PHI_P(:)
  REAL (KIND=r8),PUBLIC ,ALLOCATABLE :: PHI_C(:)
  REAL (KIND=r8),PUBLIC ,ALLOCATABLE :: PHI_M(:)
  REAL (KIND=r8),PUBLIC                           :: Uvel
  INTEGER           ,PUBLIC                            :: iMax
  PUBLIC :: Init_Class_Fields
CONTAINS  
!---------------------------------------------------------------------------------------------
  SUBROUTINE Init_Class_Fields(xdim,Uvel0)
    IMPLICIT NONE
    INTEGER            , INTENT (IN   ) :: xdim
    REAL (KIND=r8), INTENT (IN   ):: Uvel0
    iMax=xdim
    Uvel=Uvel0
    ALLOCATE (PHI_P(-1:iMax+2))
    ALLOCATE (PHI_C(-1:iMax+2))
    ALLOCATE (PHI_M(-1:iMax+2))
  END SUBROUTINE Init_Class_Fields
!---------------------------------------------------------------------------------------------
END MODULE Class_Fields


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODULE Class_NumericalMethod
  USE Class_Fields, Only : PHI_P,PHI_C,PHI_M,Uvel,iMax
 IMPLICIT NONE
 PRIVATE
 INTEGER, PUBLIC      , PARAMETER :: r8=8
 INTEGER, PUBLIC      , PARAMETER :: r4=4
 REAL (KIND=r8) :: Dt
 REAL (KIND=r8) :: Dx

 PUBLIC :: InitNumericalScheme
 PUBLIC :: SchemeForward
 PUBLIC :: SchemeUpdate
 PUBLIC :: SchemeUpStream
CONTAINS
!------------------------------------------------------------------------------------------
 SUBROUTINE InitNumericalScheme(dt_in,dx_in)
   IMPLICIT NONE
   REAL (KIND=r8), INTENT (IN   ) :: dt_in
   REAL (KIND=r8), INTENT (IN   ) :: dx_in
   INTEGER :: i
   Dt=dt_in
   Dx=dx_in
   DO i=-1,iMax+2
      IF (i*dx < 400.0) THEN
         PHI_C(i)= 0.0
      ELSE IF ( i*dx >= 400.0 .AND. i*dx <= 500.0 ) THEN
         PHI_C(i)= 0.1*(i*dx -400.0)
      ELSE IF( i*dx >= 500.0 .AND. i*dx <= 600.0 ) THEN
         PHI_C(i)= 20.0 - 0.1*(i*dx -400.0)
      ELSE IF( i*dx > 600.0 ) THEN
         PHI_C(i)= 0.0
      END IF
   END DO
   PHI_M=PHI_C
   PHI_P=PHI_C
 END SUBROUTINE InitNumericalScheme
!------------------------------------------------------------------------------------------

      FUNCTION SchemeForward()  RESULT (ok)
    IMPLICIT NONE
    ! Utilizando a diferenciacao forward
    !
    ! F(j,n+1) - F(j,n)        F(j+1,n) - F(j,n)
    !-------------------- + u -------------------- = 0
    !       dt                      dx
    !
    INTEGER :: ok
    INTEGER :: j
    DO j=1,iMax
       PHI_P(j) = PHI_C(j) - (Uvel*Dt/Dx)*(PHI_C(j+1)-PHI_C(j))
    END DO
    CALL UpdateBoundaryLayer()
 END FUNCTION SchemeForward

!---------------------------------------------------------------------------------------------
 FUNCTION SchemeUpStream()  RESULT (ok)
    IMPLICIT NONE
    ! Utilizando a diferenciacao forward no tempo e
    ! backward no espaco (upstream)
    !
    ! F(j,n+1) - F(j,n)        F(j,n) - F(j-1,n)
    !-------------------- + u -------------------- = 0
    !       dt                      dx
    !
    INTEGER :: ok
    INTEGER :: j
    DO j=1,iMax
       PHI_P(j) = PHI_C(j) - (Uvel*Dt/Dx)*(PHI_C(j)-PHI_C(j-1))
    END DO
    CALL UpdateBoundaryLayer()
 END FUNCTION SchemeUpStream

!----------------------------------------------------------------------------------------
 SUBROUTINE UpdateBoundaryLayer()
    IMPLICIT NONE
    PHI_P(0)      =  PHI_P(iMax )
    PHI_P(-1)     =  PHI_P(iMax-1)
    PHI_P(imax+1) =  PHI_P(1)
    PHI_P(iMax+2) =  PHI_P(2)
 END  SUBROUTINE UpdateBoundaryLayer
!----------------------------------------------------------------------------------------
 FUNCTION SchemeUpdate()  RESULT (ok)
    IMPLICIT NONE
    INTEGER :: ok
    PHI_M=PHI_C
    PHI_C=PHI_P
    ok=0
 END FUNCTION SchemeUpdate
!----------------------------------------------------------------------------------------
!----------------------------------------------------------------------------------------
END MODULE Class_NumericalMethod
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODULE Class_WritetoGrads
 USE Class_Fields, Only: PHI_P,PHI_C,PHI_M,Uvel,iMax
 IMPLICIT NONE
 PRIVATE
 INTEGER, PUBLIC      , PARAMETER  :: r8=8
 INTEGER, PUBLIC      , PARAMETER  :: r4=4
 INTEGER                     , PARAMETER :: UnitData=1
 INTEGER                     , PARAMETER :: UnitCtl=2
 CHARACTER (LEN=400)                    :: FileName
 LOGICAL                                              :: CtrlWriteDataFile
 PUBLIC :: SchemeWriteCtl
 PUBLIC :: SchemeWriteData
 PUBLIC :: InitClass_WritetoGrads
CONTAINS
!---------------------------------------------------------------------------------------------

 SUBROUTINE InitClass_WritetoGrads()
    IMPLICIT NONE
   FileName=''
   FileName='AdvecLinear1D'
   CtrlWriteDataFile=.TRUE.
 END SUBROUTINE InitClass_WritetoGrads
!---------------------------------------------------------------------------------------------

 FUNCTION SchemeWriteData(irec)  RESULT (ok)
    IMPLICIT NONE
    INTEGER   , INTENT (INOUT) :: irec
    INTEGER                   :: ok
    INTEGER                   :: lrec
    REAL (KIND=r4)        :: Yout(iMax)
    INQUIRE (IOLENGTH =lrec) Yout
    IF(CtrlWriteDataFile)OPEN(UnitData,FILE=TRIM(FileName)//'.bin', &
         FORM='UNFORMATTED', ACCESS='DIRECT',  &
         STATUS='UNKNOWN',  ACTION='WRITE',RECL =lrec)

    CtrlWriteDataFile=.FALSE.
    Yout= REAL(PHI_C(1:iMax),KIND=r4)
    irec=irec+1
    WRITE(UnitData,rec=irec)Yout
    ok=0
 END FUNCTION SchemeWriteData


!---------------------------------------------------------------------------------------------

 FUNCTION SchemeWriteCtl(nrec)  RESULT (ok)
    IMPLICIT NONE
    INTEGER, INTENT (IN) :: nrec
    INTEGER             :: ok

    OPEN(UnitCtl,FILE=TRIM(FileName)//'.ctl', FORM='FORMATTED', &
        ACCESS='SEQUENTIAL',STATUS='UNKNOWN',ACTION='WRITE')
    WRITE (UnitCtl,'(A6,A        )')'dset ^',TRIM(FileName)//'.bin'
    WRITE (UnitCtl,'(A           )')'title  EDO'
    WRITE (UnitCtl,'(A           )')'undef  -9999.9'
    WRITE (UnitCtl,'(A6,I8,A18   )')'xdef  ',iMax,' linear 0.00 0.001'
    WRITE (UnitCtl,'(A           )')'ydef  1 linear  -1.27 1'
    WRITE (UnitCtl,'(A6,I6,A25   )')'tdef  ',nrec,' linear  00z01jan0001 1hr'
    WRITE (UnitCtl,'(A20         )')'zdef  1 levels 1000 '
    WRITE (UnitCtl,'(A           )')'vars 1'
    WRITE (UnitCtl,'(A           )')'phic 0 99 resultado da edol yc'
    WRITE (UnitCtl,'(A           )')'endvars'
    CLOSE (UnitCtl, STATUS='KEEP')
    CLOSE (UnitData, STATUS='KEEP')
    ok=0
 END FUNCTION SchemeWriteCtl

!---------------------------------------------------------------------------------------------

END MODULE Class_WritetoGrads

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

PROGRAM Main
 USE Class_Fields, Only: Init_Class_Fields
 USE Class_NumericalMethod, Only:InitNumericalScheme, &
             SchemeForward,SchemeUpdate,SchemeUpStream
 USE Class_WritetoGrads, Only :InitClass_WritetoGrads, &
            SchemeWriteData,SchemeWriteCtl
 IMPLICIT NONE
 INTEGER                   , PARAMETER  :: r8=8
 INTEGER                   , PARAMETER  :: r4=4
 INTEGER                   , PARAMETER :: xdim=1000
 REAL (KIND=r8)        , PARAMETER :: Uvel0=10.0!m/s
 REAL (KIND=r8)        , PARAMETER :: dt=0.1   !s
 REAL (KIND=r8)        , PARAMETER :: dx=10.0   !m
 INTEGER                   , PARAMETER :: ninteraction=20000
 CALL Init()
 CALL run()

CONTAINS
!---------------------------------------------------------------------------------------------

 SUBROUTINE Init()
   IMPLICIT NONE
   CALL Init_Class_Fields(xdim,Uvel0)
   CALL InitNumericalScheme(dt,dx)
   CALL InitClass_WritetoGrads
 END SUBROUTINE Init
!---------------------------------------------------------------------------------------------

 SUBROUTINE Run()
   IMPLICIT NONE
   INTEGER :: test,it,irec
   irec=0
   DO it=1,ninteraction
      test=SchemeUpStream()
      test=SchemeWriteData(irec)
      test=SchemeUpdate()
   END DO
   test=SchemeWriteCtl(ninteraction)
 END SUBROUTINE Run
!---------------------------------------------------------------------------------------------

 SUBROUTINE Finalize()
   IMPLICIT NONE

 END SUBROUTINE Finalize
!---------------------------------------------------------------------------------------------

END PROGRAM Main

