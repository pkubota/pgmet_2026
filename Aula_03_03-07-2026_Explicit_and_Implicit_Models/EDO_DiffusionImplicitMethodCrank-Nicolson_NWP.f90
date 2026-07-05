MODULE LinearSolve 
 IMPLICIT NONE
 PRIVATE
 INTEGER, PARAMETER :: r8 = selected_real_kind(15, 307)
 INTEGER, PARAMETER :: r4 = selected_real_kind(6, 37)
 PUBLIC :: solve_tridiag,inverse
CONTAINS
subroutine solve_tridiag(a,b,c,d,x,n)
      implicit none
!       a - sub-diagonal (diagonal abaixo da diagonal principal)
!       b - diagonal principal
!       c - sup-diagonal (diagonal acima da diagonal principal)
!       d - parte à direita
!       x - resposta
!       n - número de equações
        integer, intent(in) :: n
        real (r8), dimension (n),intent (in   ) :: a,b,c,d
        real (r8), dimension (n),intent (out) :: x
        real (r8), dimension (n) :: cp, dp
        real (r8) :: m
        integer :: i
! inicializar c-primo e d-primo
        cp(1) = c(1)/b(1)
        dp(1) = d(1)/b(1)
! resolver para vetores c-primo e d-primo
         do i = 2,n
           m = b(i)-cp(i-1)*a(i)
           cp(i) = c(i)/m
           dp(i) = (d(i)-dp(i-1)*a(i))/m
         enddo
! inicializar x
         x(n) = dp(n)
! resolver para x a partir de vetores c-primo e d-primo
        do i = n-1, 1, -1
          x(i) = dp(i)-cp(i)*x(i+1)
        end do
    end subroutine solve_tridiag

  subroutine inverse(a_in,c,n)
!============================================================
! Inverse matrix
! Method: Based on Doolittle LU factorization for Ax=b
! Alex G. December 2009
!-----------------------------------------------------------
! input ...
! a(n,n) - array of coefficients for matrix A
! n      - dimension
! output ...
! c(n,n) - inverse matrix of A
! comments ...
! the original matrix a(n,n) will be destroyed 
! during the calculation
!===========================================================
implicit none 
integer n
real (r8) a_in(n,n), c(n,n)
real (r8) L(n,n), U(n,n), b(n), d(n), x(n)
real (r8) coeff
integer i, j, k
real (r8) a(n,n)

a=a_in
! step 0: initialization for matrices L and U and b
! Fortran 90/95 aloows such operations on matrices
L=0.0
U=0.0
b=0.0

! step 1: forward elimination
do k=1, n-1
   do i=k+1,n
      coeff=a(i,k)/a(k,k)
      L(i,k) = coeff
      do j=k+1,n
         a(i,j) = a(i,j)-coeff*a(k,j)
      end do
   end do
end do

! Step 2: prepare L and U matrices 
! L matrix is a matrix of the elimination coefficient
! + the diagonal elements are 1.0
do i=1,n
  L(i,i) = 1.0
end do
! U matrix is the upper triangular part of A
do j=1,n
  do i=1,j
    U(i,j) = a(i,j)
  end do
end do

! Step 3: compute columns of the inverse matrix C
do k=1,n
  b(k)=1.0
  d(1) = b(1)
! Step 3a: Solve Ld=b using the forward substitution
  do i=2,n
    d(i)=b(i)
    do j=1,i-1
      d(i) = d(i) - L(i,j)*d(j)
    end do
  end do
! Step 3b: Solve Ux=d using the back substitution
  x(n)=d(n)/U(n,n)
  do i = n-1,1,-1
    x(i) = d(i)
    do j=n,i+1,-1
      x(i)=x(i)-U(i,j)*x(j)
    end do
    x(i) = x(i)/u(i,i)
  end do
! Step 3c: fill the solutions x(n) into column k of C
  do i=1,n
    c(i,k) = x(i)
  end do
  b(k)=0.0
end do
end subroutine inverse



END MODULE LinearSolve


MODULE Class_Fields
 IMPLICIT NONE
 PRIVATE
 INTEGER, PARAMETER :: r8 = selected_real_kind(15, 307)
 INTEGER, PARAMETER :: r4 = selected_real_kind(6, 37)
 REAL (KIND=r8), PUBLIC, ALLOCATABLE :: A0  (:)
 REAL (KIND=r8), PUBLIC, ALLOCATABLE :: A   (:)
 REAL (KIND=r8), PUBLIC, ALLOCATABLE :: Anew(:)
 REAL (KIND=r8), PUBLIC, ALLOCATABLE :: AA  (:,:)
 REAL (KIND=r8), PUBLIC, ALLOCATABLE :: AAm (:,:)
 REAL (KIND=r8), PUBLIC, ALLOCATABLE :: AAinv  (:,:)
 REAL (KIND=r8), PUBLIC, ALLOCATABLE :: B   (:)
 REAL (KIND=r8), PUBLIC, ALLOCATABLE :: X   (:)
 INTEGER      , PUBLIC              :: ilo
 INTEGER      , PUBLIC              :: ihi
 INTEGER      , PUBLIC              :: iMax 
 PUBLIC ::  Init_Class_Fields

 CONTAINS 
 
SUBROUTINE Init_Class_Fields(nx,dx, coef_C,Coef_CN )
 IMPLICIT NONE
 INTEGER        , INTENT (IN   ) :: nx
 REAL (KIND=r8) , INTENT (IN   ) :: dx
 REAL (KIND=r8) , INTENT (IN   ) :: coef_C
 REAL (KIND=r8) , INTENT (IN   ) :: Coef_CN
 REAL (KIND=r8) :: coord_X(0:nx)
 INTEGER        :: i
 iMax=nx
 ALLOCATE (A0   (0:iMax)       )
 ALLOCATE (A    (0:iMax)       )
 ALLOCATE (Anew (0:iMax)       )
 ALLOCATE (AA   (0:iMax+1,0:iMax+1))
 ALLOCATE (AAm  (0:iMax+1,0:iMax+1)) 
 ALLOCATE (AAinv(0:iMax+1,0:iMax+1)) 
 ALLOCATE (B    (0:iMax)       )
 ALLOCATE (X    (0:iMax)       )

!# python is zero-based.  We are assuming periodic BCs, so 
!# points 0 and N-1 are the same.  Set some integer indices to 
!# allow us to easily access points 1 through N-1.  Point 0 
!# won't be explicitly updated, but rather filled by the BC 
!# routine. 
 ilo = 1 
 ihi = iMax-1
 DO i=0,iMax 
    coord_X(i) = REAL(i,KIND=r8)*dx
 END DO 
 !
 ! initialize the data -- tophat
 !
! DO i=0,iMax 
!    IF(coord_X(i) >= 0.333_r8 .and. coord_X(i) <= 0.666_r8) THEN
!       A0(i) = COS(0.50_r8 - coord_X(i))
!    ELSE
!       A0(i) = 0.0_r8
!    END IF
! END DO 

 DO i=0,iMax 
   ! IF(coord_X(i) >= 0.333_r8 .and. coord_X(i) <= 0.666_r8) THEN
       A0(i) = exp(- coord_X(i)*10)
   ! ELSE
   !    A0(i) = 0.0_r8
   ! END IF
 END DO 
 A=A0
 AA=0.0_r8
 AAm=0.0_r8 
 !
 ! """ we don't explicitly update point 0, since it is identical 
 !            to N-1, so fill it here """ 
 !A(0) = A(ihi)

 !                                  -                                                                                           -
 !  a(t+1,i)  -   a(t,i)        1  |      a(t  ,i+1) - 2*a(t  ,i) + a(t  ,i-1)          a(t+1,i+1) - 2*a(t+1,i) + a(t+1,i-1)     |
 !  -----------------------  = --- | K ------------------------------------------ + K ------------------------------------------ |
 !          Dt                  2  |                   Dx*Dx                                       Dx*Dx                         |
 !                                 -                                                                                            -
 !                                      -                                                                             -
 !                               Dt    |                                                                               |
 !  a(t+1,i)  -  a(t,i)  =    K ----   |  a(t  ,i+1) - 2*a(t  ,i) + a(t  ,i-1)  + a(t+1,i+1) - 2*a(t+1,i) + a(t+1,i-1) |
 !                               Dx*Dx |                                                                               |
 !                                     -                                                                              -

 !                               -                                                                             -
 !                              |                                                                               |
 !  2a(t+1,i)  -  2a(t,i)  =  C |  a(t  ,i+1) - 2*a(t  ,i) + a(t  ,i-1)  + a(t+1,i+1) - 2*a(t+1,i) + a(t+1,i-1) |
 !                              |                                                                               |
 !                              -                                                                              -

 !                -                                     -           -                                       -               
 !                |                                      |          |                                        |               
 ! 2a(t+1,i) -C * | a(t+1,i+1) - 2*a(t+1,i) + a(t+1,i-1) |   =    C |  a(t  ,i+1) - 2*a(t  ,i) + a(t  ,i-1)  |  +  2a(t,i)
 !                |                                      |          |                                        |               
 !                -                                      -          -                                       -               


 !           
 !           
 !  2a(t+1,i) + 2C*a(t+1,i) -Ca(t+1,i+1)  -C a(t+1,i-1)  = 2a(t,i) - 2C*a(t  ,i) +  C a(t  ,i+1)   + C a(t  ,i-1) 
 !           
 !           

 !           
 !           
 !  -Ca(t+1,i+1)  +(1+2C) a(t+1,i)  -C a(t+1,i-1)  =  C a(t  ,i+1) + (1 -2C) a(t,i)  + C a(t  ,i-1) 
 !           
 !           
 !  -Ca(t+1,i-1)  +(2+2C) a(t+1,i)  -C a(t+1,i+1)  =  C a(t  ,i-1) + (2 -2C) a(t,i)  + C a(t  ,i+1) 

 ! 
 ! 
 !
 ! ................................................................
 !
 !  -Ca(t+1,i-1)  +2(1+C) a(t+1,i)  -C a(t+1,i+1)  =  C a(t  ,i-1) + 2(1 -C) a(t,i)  + C a(t  ,i+1) 
 !
 ! ................................................................
 !
 !     -C a(t+1,  0)  + 2(1 + C) a(t+1,1)  -  C* a(t+1,  2)  =   C a(t  ,0) + 2(1 -C) a(t,1)  + C a(t  ,2) 
 !     -C a(t+1,  1)  + 2(1 + C) a(t+1,2)  -  C* a(t+1,  3)  =   C a(t  ,1) + 2(1 -C) a(t,2)  + C a(t  ,3) <---------------------------
 !     -C a(t+1,  2)  + 2(1 + C) a(t+1,3)  -  C* a(t+1,  4)  =   C a(t  ,2) + 2(1 -C) a(t,3)  + C a(t  ,4)                           |
 !     -C a(t+1,  3)  + 2(1 + C) a(t+1,4)  -  C* a(t+1,  5)  =   C a(t  ,3) + 2(1 -C) a(t,4)  + C a(t  ,5)                           |
 !     -C a(t+1,i-1)  + 2(1 + C) a(t+1,i)  -  C* a(t+1,i+1)  =   C a(t  ,4) + 2(1 -C) a(t,5)  + C a(t  ,6)                           |
 !                                                                                                                                   | equatios adv
 !                                                                                                                                   |__________|
 !     _____                               ___   __         __          ____                                ____   __         __                |
 !    |                                       | |             |        |                                        | |             |               |
 !x=0 |  0            0      0         0      | | a(t+1,0  )  | =  x=0 |  0             0      0       0        | | a(t,0  )    |               |
 !x=1 | 2(1 + C)     -C      0        -C      | | a(t+1,1  )  | =  x=1 | 2(1 - C)       C      0       C        | | a(t,1  )    | <-------------
 !x=2 | -C       2(1 + C)   -C         0      | | a(t+1,2  )  | =  x=2 |  C        2(1 - C)    C       0        | | a(t,2  )    | 
 !x=3 |  0           -C   2(1 + C)    -C      | | a(t+1,3  )  | =  x=3 |  0             C   2(1 - C)   C        | | a(t,3  )    | 
 !x=4 | -C            0     -C      2(1 + C)  | | a(t+1,4  )  | =  x=4 |  C             0      C      2(1 - C)  | | a(t,4  )    | 
 !x=5 |  0            0      0         0      | | a(t+1,i+1)  | =  x=5 |  0             0      0       0        | | a(t,i+1)    |
 !    |____                               ____| |__         __|        |____                                ____| |__         __|
 !
 !                 A                                X         =     B
 !
 !# create the matrix      AAm 
 !         # loop over rows [ilo,ihi] and construct the matrix.  This will 
 !         # be almost bidiagonal, but with the upper right entry also 
 !         # nonzero. 
 !
 ! ilo = 1 
 ! ihi = iMax-1 
 ! AA(0,ihi  ) = -coef_C 
 ! AA(0,ilo  ) = -coef_C    !-coef_C 
  DO i =ilo,ihi
      AAm(i,i+1) = coef_C 
      AAm(i,i  ) = 2*(1.0_r8 - coef_C) 
      AAm(i,i-1) = coef_C
  END DO
 ! AA(nx,ihi  ) = -coef_C 
 ! AA(nx,ilo  ) =  -coef_C 

 !# create the matrix   AA     
 !         # loop over rows [ilo,ihi] and construct the matrix.  This will 
 !         # be almost bidiagonal, but with the upper right entry also 
 !         # nonzero. 
 !
 ! ilo = 1 
 ! ihi = iMax-1 
 ! AA(0,ihi  ) = -coef_C 
 ! AA(0,ilo  ) = -coef_C    !-coef_C 
  DO i =ilo,ihi
      AA(i,i+1) = -coef_C 
      AA(i,i  ) = 2*(1.0_r8 + coef_C) 
      AA(i,i-1) = -coef_C
  END DO
 ! AA(nx,ihi  ) = -coef_C 
 ! AA(nx,ilo  ) =  -coef_C 
END SUBROUTINE Init_Class_Fields
END MODULE Class_Fields



MODULE Class_WritetoGrads
 USE Class_Fields, Only: Anew,A,iMax
 IMPLICIT NONE
 PRIVATE
 INTEGER, PUBLIC      , PARAMETER  :: r8=8
 INTEGER, PUBLIC      , PARAMETER  :: r4=4
 INTEGER              , PARAMETER  :: UnitData=1
 INTEGER              , PARAMETER  :: UnitCtl=2
 CHARACTER (LEN=400)               :: FileName
 LOGICAL                                         :: CtrlWriteDataFile
 PUBLIC :: SchemeWriteCtl
 PUBLIC :: SchemeWriteData
 PUBLIC :: InitClass_WritetoGrads

CONTAINS

 SUBROUTINE InitClass_WritetoGrads()
    IMPLICIT NONE
   FileName=''
   FileName='ImplicitLinearDiffusion1DMethodCrankNicolson_NWP'
   CtrlWriteDataFile=.TRUE.
 END SUBROUTINE InitClass_WritetoGrads

 FUNCTION SchemeWriteData(irec)  RESULT (ok)
    IMPLICIT NONE
    INTEGER   , INTENT (INOUT) :: irec
    INTEGER              :: ok
    INTEGER              :: lrec
    REAL (KIND=r4)        :: Yout(iMax)
    INQUIRE (IOLENGTH=lrec) Yout
    IF(CtrlWriteDataFile) OPEN (UnitData,FILE=TRIM(FileName)//'.bin', &
     FORM='UNFORMATTED', ACCESS='DIRECT',STATUS='UNKNOWN',&
     ACTION='WRITE',RECL=lrec)
    CtrlWriteDataFile=.FALSE.
    Yout=REAL(A(1:iMax),KIND=r4)
    irec=irec+1
    WRITE(UnitData,rec=irec)Yout
    ok=0
 END FUNCTION SchemeWriteData
FUNCTION SchemeWriteCtl(nrec)  RESULT(ok)
    IMPLICIT NONE
    INTEGER, INTENT (IN) :: nrec
    INTEGER             :: ok

    OPEN (UnitCtl,FILE=TRIM(FileName)//'.ctl', &
    FORM='FORMATTED',ACCESS='SEQUENTIAL', &
    STATUS='UNKNOWN',ACTION='WRITE')
    WRITE (UnitCtl,'(A6,A        )')'dset ^',TRIM(FileName)//'.bin'
    WRITE (UnitCtl,'(A           )')'title  EDO'
    WRITE (UnitCtl,'(A           )')'undef  -9999.9'
    WRITE (UnitCtl,'(A6,I8,A18   )')'xdef  ',iMax,' linear 0.00 0.001'
    WRITE (UnitCtl,'(A           )')'ydef  1 linear  -1.27 1'
    WRITE (UnitCtl,'(A6,I6,A25   )')'tdef  ',nrec,' linear  00z01jan0001 1hr'
    WRITE (UnitCtl,'(A20         )')'zdef  1 levels 1000 '
    WRITE (UnitCtl,'(A           )')'vars 1'
    WRITE (UnitCtl,'(A           )')'A   0 99 resultado da edol yc'
    WRITE (UnitCtl,'(A           )')'endvars'
    CLOSE (UnitCtl,STATUS='KEEP') 
    CLOSE (UnitData,STATUS='KEEP') 
    ok=0   
 END FUNCTION SchemeWriteCtl


END MODULE Class_WritetoGrads
PROGRAM Main
 USE  Class_Fields, Only : Init_Class_Fields,B,A,AA,AAm,ilo,ihi,Anew,AAinv
 USE  LinearSolve, OnLy : solve_tridiag,inverse
 USE  Class_WritetoGrads, OnLy :schemeWriteCtl, schemeWriteData,initClass_WritetoGrads

 ! main program to check the Tridiagonal system solver
 INTEGER, PARAMETER :: r8 = selected_real_kind(15, 307)
 INTEGER, PARAMETER :: r4 = selected_real_kind(6, 37)
 INTEGER      , PARAMETER :: nn=200
 INTEGER      , PARAMETER :: nx=nn
 REAL (KIND=8) , PARAMETER :: xmin=0.0
 REAL (KIND=8) , PARAMETER :: xmax=1.0
 REAL (KIND=8) , PARAMETER :: K=2.9e-1        ! # CFL number
 REAL (KIND=8)  :: Coef_C        !
 REAL (KIND=8)  :: Coef_CN        !

 REAL (KIND=8) , PARAMETER :: C=0.8        ! # CFL number
                                                                         !    [0.5, 1.0, 10.0]
 REAL(KIND=8) , PARAMETER :: u = 10.0
 REAL (KIND=8) , PARAMETER :: Dx=(xmax - xmin)/(nx-1)
 REAL(KIND=8) , PARAMETER :: Dt=C*dx/u
 INTEGER      , PARAMETER :: ninteraction=400
 INTEGER :: i,j,ii
 


 Coef_C= K* Dt/ (Dx*Dx)  
 Coef_CN =K* Dt/ (Dx*Dx)         !
 
 CALL Init()
 CALL run()
 STOP

 CONTAINS

 SUBROUTINE Init()
  IMPLICIT NONE
    CALL Init_Class_Fields(nx,dx, Coef_C,Coef_CN)
    CALL initClass_WritetoGrads()
 END SUBROUTINE Init
 
SUBROUTINE Run()
  IMPLICIT NONE
  INTEGER :: test,irec,n



  irec=0
  DO i=1,ninteraction
        PRINT*,'ninteraction=',i,'ave=',sum(A(ilo:ihi))/size(A(ilo:ihi),dim=1)
        test=SchemeWriteData(irec)
        n=ihi-ilo+1
        call inverse(AA(ilo:ihi,ilo:ihi),AAinv(ilo:ihi,ilo:ihi),n)
       ! U(n+1,i)= inv(M1(n+1,i))* M2(n+1,i)*U(n,i)

        Anew(ilo:ihi)=matmul(matmul(AAinv(ilo:ihi,ilo:ihi), AAm(ilo:ihi,ilo:ihi)), A(ilo:ihi))
        A   (ilo:ihi) = Anew(ilo:ihi)
  END DO    !   t += dt
  test=SchemeWriteCtl(ninteraction)

 END SUBROUTINE Run
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 SUBROUTINE Finalize()
 
 END SUBROUTINE Finalize
END PROGRAM Main
