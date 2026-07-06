!
! ! > gfortran AdvecSpec.f90 NCAR_fft.f
!
MODULE Class_Fields
  IMPLICIT NONE
  PRIVATE
  INTEGER, PUBLIC      , PARAMETER  :: r8=8
  INTEGER, PUBLIC      , PARAMETER  :: r4=4
  REAL,PUBLIC ,ALLOCATABLE ::  Spec_Qc(:)
  REAL,PUBLIC ,ALLOCATABLE ::  Spec_Qm(:)
  REAL,PUBLIC ,ALLOCATABLE ::  Spec_dqdx(:)
  REAL,PUBLIC ,ALLOCATABLE ::  dqdx(:)
  REAL,PUBLIC ,ALLOCATABLE ::  Tend_Qp(:)
  REAL,PUBLIC ,ALLOCATABLE ::  Grid_Qc(:)
  REAL,PUBLIC ,ALLOCATABLE ::  Grid_Qm(:)
  REAL,PUBLIC ,ALLOCATABLE ::  Grid_Qp(:)
  REAL,PUBLIC ,ALLOCATABLE ::  Qc(:)
  REAL,PUBLIC ,ALLOCATABLE ::  x(:)
  REAL,PUBLIC ,ALLOCATABLE ::  trig(:)

  REAL           ,PUBLIC                  :: Uvel
  INTEGER       ,PUBLIC                   :: nn
  PUBLIC :: Init_Class_Fields
CONTAINS  
  !-----------------------------------------------------------------------------------------
  SUBROUTINE Init_Class_Fields(xdim,Uvel0)
    IMPLICIT NONE
    INTEGER      , INTENT (IN   ) :: xdim
    REAL         , INTENT (IN   ) :: Uvel0
    nn=xdim
    Uvel=Uvel0
    ALLOCATE (Spec_Qc(0:nn-1))
    ALLOCATE (Spec_Qm(0:nn-1))
    ALLOCATE (Spec_dqdx(0:nn-1))
    ALLOCATE (dqdx   (0:nn-1))  
    ALLOCATE (Tend_Qp(0:nn-1))
    ALLOCATE (Grid_Qc(0:nn-1))
    ALLOCATE (Grid_Qm(0:nn-1))
    ALLOCATE (Grid_Qp(0:nn-1))
    ALLOCATE (Qc     (0:nn-1))
    ALLOCATE (x      (0:nn-1))
    ALLOCATE (trig   (2*nn+15))
  END SUBROUTINE Init_Class_Fields
  !------------------------------------------------------------------------------------------
END MODULE Class_Fields


 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODULE Class_WritetoGrads
 USE Class_Fields, ONLY: Spec_Qc,Spec_Qm,dqdx,Spec_dqdx,Grid_Qc,Grid_Qm,Grid_Qp,nn
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

 FUNCTION SchemeWriteData(irec)  RESULT (ok)
    IMPLICIT NONE
    INTEGER   , INTENT (INOUT) :: irec
    INTEGER                                  :: ok
    INTEGER                                  :: lrec
    REAL (KIND=r4)                      :: Yout(nn)
    INQUIRE (IOLENGTH=lrec) Grid_Qc(0:nn-1)
    IF(CtrlWriteDataFile)OPEN(UnitData,FILE=TRIM(FileName)//'.bin',&
    FORM='UNFORMATTED', ACCESS='DIRECT', STATUS='UNKNOWN', &
    ACTION='WRITE',RECL=lrec)
    ok=1
    CtrlWriteDataFile=.FALSE.
    Yout=REAL(Grid_Qc(0:nn-1),KIND=r4)
    irec=irec+1
    WRITE(UnitData,rec=irec)Yout
    ok=0
 END FUNCTION SchemeWriteData

 FUNCTION SchemeWriteCtl(nrec)  RESULT (ok)
    IMPLICIT NONE
    INTEGER, INTENT (IN) :: nrec
    INTEGER             :: ok
    ok=1
   OPEN(UnitCtl,FILE=TRIM(FileName)//'.ctl',FORM='FORMATTED', &
   ACCESS='SEQUENTIAL',STATUS='UNKNOWN',ACTION='WRITE')
    WRITE (UnitCtl,'(A6,A           )')'dset ^',TRIM(FileName)//'.bin'
    WRITE (UnitCtl,'(A                 )')'title  EDO'
    WRITE (UnitCtl,'(A                 )')'undef  -9999.9'
    WRITE (UnitCtl,'(A6,I8,A18   )')'xdef  ',nn,' linear 0.00 0.001'
    WRITE (UnitCtl,'(A                  )')'ydef  1 linear  -1.27 1'
    WRITE (UnitCtl,'(A6,I6,A25   )')'tdef  ',nrec,' linear  00z01jan0001 1hr'
    WRITE (UnitCtl,'(A20             )')'zdef  1 levels 1000 '
    WRITE (UnitCtl,'(A           )')'vars 1'
    WRITE (UnitCtl,'(A           )')'Qc   0 99 resultado da edol yc'
    WRITE (UnitCtl,'(A           )')'endvars'
    CLOSE (UnitCtl,STATUS='KEEP')
    CLOSE (UnitData,STATUS='KEEP')
    ok=0
 END FUNCTION SchemeWriteCtl
END MODULE Class_WritetoGrads



PROGRAM t_specderiv
  USE Class_WritetoGrads, ONLY :InitClass_WritetoGrads, &
       SchemeWriteData, SchemeWriteCtl
  USE Class_Fields, ONLY :Init_Class_Fields,Spec_Qc,Spec_Qm,dqdx,Spec_dqdx,Tend_Qp,Grid_Qc,Grid_Qm,Grid_Qp,Uvel,Qc,x,trig,nn
  ! Example of computing first derivative of a real function 
  ! using discrete spectral transform and calling NCAR FFTPACK

  IMPLICIT NONE
  REAL               :: tmp
  REAL               :: Dt   =0.1
  REAL               :: uVel0=0.5
  INTEGER            :: ninteraction=100
  INTEGER, PARAMETER :: nPtsMax=16
  REAL,PARAMETER:: pi = ACOS(-1.0)
  INTEGER :: i, ii,it,irec,test

  CALL Init_Class_Fields(nPtsMax,uVel0)
  CALL InitClass_WritetoGrads()

  !--Give values: f(x) = cos2x + sin5x

  DO i= 0, nn-1
     x(i)=2.0*pi*REAL(i-(nn/2.0))/REAL(nn)
     Qc(i)=COS(x(i)) 
  END DO

  Grid_Qc=Qc
  Grid_Qm=Qc
  Grid_Qp=Qc

  !f(x) = cos(xj
  Spec_Qc=Qc  ! =COS(x(i)) 

  irec=0
  test=SchemeWriteData(irec)

  DO it=1,ninteraction
      !
      !--Forward transform to compute the complex coefficients
      !
      !     RFFTI INITIALIZES THE ARRAY WSAVE WHICH IS USED IN             
      !     BOTH RFFTF AND RFFTB. THE PRIME FACTORIZATION OF N TOGETHER WITH          
      !     A TABULATION OF THE TRIGONOMETRIC FUNCTIONS ARE COMPUTED AND              
      !     STORED IN WSAVE.
      !
      !                 N/2 -1                      N-1  
      !              -------                       -------
      !              \                            \
      !   f(j)=       \  (ffn) * exp (i*n*xj)  =   \  (ffn) * exp (i*n*xj)
      !               /                            /
      !              /                            / 
      !              -------                      -------
      !                 n=-N/2                      n=0
      !
      !where ffn is the discrete Fourier transform:
      !
      !                 N-1 
      !               -------
      !          1    \
      !   ffn = ----   \  f(xj) * exp (i*n*xj)
      !          N     /
      !               / 
      !              -------
      !                 j=0

      !
      CALL rffti(nn  ,trig)
      !     COMPUTES THE FOURIER COEFFICIENTS OF A REAL              
      !     PERODIC SEQUENCE (FOURIER ANALYSIS). THE TRANSFORM IS DEFINED             
      !     BELOW AT OUTPUT PARAMETER R.
      !
      !                 N/2 -1 
      !              -------
      ! df(x)        \
      !------ =       \  (i*n)*ffn * exp (i*n*x)
      !  dx           /
      !              / 
      !              -------
      !                 n=-N/2 
      !
      !          
      !           
      !Spec_Qc =    f(xj) 
      !            
      !
      CALL rfftf(nn,Spec_Qc,trig)
      !
      !             N-1 
      !           -------
      !           \
      !Spec_Qc =   \  f(xj) * exp (i*n*xj) 
      !            /
      !           / 
      !          -------
      !              j=0
      DO i= 0, nn-1
         Tend_Qp(i)=Spec_Qc(i)
      END DO
      !
      !--Set 0 to the coefficient of nn/2 mode 
      !
      Tend_Qp(nn-1) = 0.0
      !
      !--Multiply and swap the Fourier coefficients for first derivative
      !
      !                 N/2 -1 
      !              -------
      !              \
      !Tend_Qp=       \  (i*n*ffn) exp (i*n*x)
      !               /
      !              / 
      !              -------
      !                 n=-N/2 
      !
      !
      ii = 1
      DO i= 1, nn-3, 2
        !                 N/2 -1 
        !              -------
        !               \
        !Tend_Qp=  ----- \  (i*n)*f(xj) * exp (i*n*x)
        !                /
        !              / 
        !              -------
        !                 n=-N/2 
        !
         tmp          = -ii*Tend_Qp(i+1)
         Tend_Qp(i+1) =  ii*Tend_Qp(i)
         Tend_Qp(i)   =  tmp
         ii = ii + 1
      END DO
      !                    N/2 -1 
      !                   -------
      !               1   \
      !Spec_dqdx =  -----  \  (i*n)*f(xj) * exp (i*n*x)
      !               N    /
      !                   / 
      !                   -------
      !                    n=-N/2 
      !
      Spec_dqdx = Tend_Qp/nn

      !--Backward transform
      !    COMPUTES THE REAL PERODIC SEQUENCE FROM ITS              
      !    FOURIER COEFFICIENTS (FOURIER SYNTHESIS). THE TRANSFORM IS DEFINED        
      !    BELOW AT OUTPUT PARAMETER R. 
      CALL rfftb(nn,Spec_dqdx,trig)
      dqdx=Spec_dqdx
      !
      !
      !  Qp - Qc           dQc
      !---------- = - U * -----
      !     Dt              dx
      !
      !
      Grid_Qp  = Grid_Qc - (uVel * Dt)  * dqdx
      Grid_Qc=Grid_Qp
      Grid_Qm=Grid_Qc
      Spec_Qc=Grid_Qc
      test=SchemeWriteData(irec)

      !-Output and compare with exact values

      WRITE(*,*) '          j    spectral        exact'
      DO i= 0, nn-1
         WRITE(*,*) i, Grid_Qc(i), Qc(i)
      END DO
      WRITE(*,*) ''
      WRITE(*,*) 'Max error: ', MAXVAL(ABS(Grid_Qc-Qc))
  END DO   !DO it=1,iteration
  test=SchemeWriteCtl(ninteraction)


END PROGRAM t_specderiv
