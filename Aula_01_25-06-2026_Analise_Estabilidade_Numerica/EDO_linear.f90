MODULE Class_NumericalScheme
  IMPLICIT NONE
  PRIVATE
  INTEGER, PUBLIC      , PARAMETER :: r8=8
  INTEGER, PUBLIC      , PARAMETER :: r4=4

  REAL (KIND=r8)                    :: dt
  REAL (KIND=r8)                    :: lambda
  INTEGER              , PARAMETER :: UnitData=1
  INTEGER              , PARAMETER :: UnitCtl=2
  CHARACTER (LEN=400)               :: FileName
  LOGICAL                          :: CtrlWriteDataFile
  PUBLIC :: InitNumericalScheme
  PUBLIC :: SchemeForward
  PUBLIC :: SchemeUpdate
  PUBLIC :: SchemeWriteData
  PUBLIC :: SchemeWriteCtl
  PUBLIC :: AnaliticFunction

CONTAINS

 SUBROUTINE InitNumericalScheme(dt_in,lambda_in)
    IMPLICIT NONE
   REAL (KIND=r8) :: dt_in
   REAL (KIND=r8) :: lambda_in
   FileName=''
   dt=dt_in
   lambda=lambda_in
   FileName='EDOL'
   CtrlWriteDataFile=.TRUE.
 END SUBROUTINE InitNumericalScheme

FUNCTION SchemeForward(yc)  RESULT(yp)
    IMPLICIT NONE
    ! Utilizando a diferenciacao forward
    !
    ! y(n+1) - y(n)
    !--------------- = lambda * y(n) ; onde lambda > 0
    !       dt
    !
    REAL (KIND=r8), INTENT (IN   ) :: yc
    REAL (KIND=r8) :: yp

    yp =  yc -lambda*dt*yc  

 END FUNCTION SchemeForward

 FUNCTION SchemeUpdate(y_in)  RESULT (y_out)
    IMPLICIT NONE
    REAL (KIND=r8), INTENT(IN) :: y_in
    REAL (KIND=r8)             :: y_out
    y_out=y_in
 END FUNCTION SchemeUpdate


 FUNCTION AnaliticFunction(y0,tn)  RESULT (y_out)
    IMPLICIT NONE
    REAL (KIND=r8), INTENT (IN) :: y0
    INTEGER      ,       INTENT (IN) :: tn

    REAL (KIND=r8)             :: y_out
    y_out=y0*exp(-lambda*tn*dt)
 END FUNCTION AnaliticFunction

FUNCTION SchemeWriteData(irec,y_in,ya)  RESULT (ok)
    IMPLICIT NONE
    INTEGER   , INTENT (INOUT) :: irec
    REAL (KIND=r8), INTENT (IN) :: y_in
    REAL (KIND=r8), INTENT (IN) :: ya
    INTEGER              :: ok
    INTEGER              :: lrec
    REAL (KIND=r4)        :: Yout
    INQUIRE (IOLENGTH=lrec) Yout
    IF(CtrlWriteDataFile)OPEN(UnitData,FILE=TRIM(FileName)//'.bin',& 
    FORM='UNFORMATTED', ACCESS='DIRECT', STATUS='UNKNOWN', &
    ACTION='WRITE',RECL=lrec)
    CtrlWriteDataFile=.FALSE.
    Yout=REAL(y_in,KIND=r4)
    irec=irec+1
    WRITE(UnitData,rec=irec) Yout
    irec=irec+1
    WRITE(UnitData,rec=irec) REAL (ya,KIND=r4)
    ok=0
 END FUNCTION SchemeWriteData


 FUNCTION SchemeWriteCtl(nrec)  RESULT( ok)
    IMPLICIT NONE
    INTEGER, INTENT (IN) :: nrec
    INTEGER             :: ok

    OPEN(UnitCtl,FILE=TRIM(FileName)//'.ctl',FORM='FORMATTED',&
    ACCESS='SEQUENTIAL',STATUS='UNKNOWN',ACTION='WRITE')
    WRITE (UnitCtl,'(A6,A)')'dset ^',TRIM(FileName)//'.bin'
    WRITE (UnitCtl,'(A   )')'title  EDO'
    WRITE (UnitCtl,'(A   )')'undef  -9999.9'
    WRITE (UnitCtl,'(A   )')'xdef  1 linear -48.00 1'
    WRITE (UnitCtl,'(A   )')'ydef  1 linear  -1.27 1'
    WRITE (UnitCtl,'(A6,I6,A25)')'tdef  ',nrec,' linear  00z01jan0001 1hr'
    WRITE (UnitCtl,'(A20 )')'zdef  1 levels 1000 '
    WRITE (UnitCtl,'(A)')'vars 2'
    WRITE (UnitCtl,'(A)')'yc 0 99 resultado da edol yc'
    WRITE (UnitCtl,'(A)')'ya 0 99 funcao analitica'
    WRITE (UnitCtl,'(A)')'endvars'
    CLOSE (UnitCtl,STATUS='KEEP') 
    CLOSE (UnitData,STATUS='KEEP') 

    ok=0   
 END FUNCTION SchemeWriteCtl

END MODULE Class_NumericalScheme


PROGRAM MAIN
 USE Class_NumericalScheme, Only :r8, InitNumericalScheme, SchemeForward, SchemeUpdate,&
                                      SchemeWriteCtl, SchemeWriteData, AnaliticFunction
 IMPLICIT NONE
 REAL (KIND=r8) :: yn
 REAL (KIND=r8) :: yc
 REAL (KIND=r8) :: yp
 REAL (KIND=r8) :: ya
 
 REAL (KIND=r8), PARAMETER :: lambda=0.1 ![1/s]
 REAL (KIND=r8), PARAMETER ::  y0=1.0_r8
 INTEGER       , PARAMETER ::  nrec=200
 REAL(KIND=r8) :: dt=0.5/lambda
 INTEGER       :: test
 INTEGER       :: irec
 INTEGER       :: tn
 
 CALL Init()
 yc=y0
irec=0
 DO tn=0,nrec
    yp=SchemeForward(yc)
    ya=AnaliticFunction(y0,tn)
    test=SchemeWriteData(irec,yc,ya) 
    yn=SchemeUpdate(yc)
    yc=SchemeUpdate(yp)
 END DO
 test=SchemeWriteCtl(nrec)

CONTAINS

  SUBROUTINE Init()
    IMPLICIT NONE
    CALL InitNumericalScheme(dt,lambda)
  END SUBROUTINE Init
END PROGRAM MAIN

