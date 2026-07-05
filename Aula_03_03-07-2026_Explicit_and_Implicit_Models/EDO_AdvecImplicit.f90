MODULE LinearSolve 
 IMPLICIT NONE
 PRIVATE
 INTEGER, PARAMETER :: r8 = selected_real_kind(15, 307)
 INTEGER, PARAMETER :: r4 = selected_real_kind(6, 37)
 PUBLIC :: solve_tridiag
CONTAINS
subroutine solve_tridiag(a,b,c,d,x,n)
      implicit none
!	a - sub-diagonal (diagonal abaixo da diagonal principal)
!	b - diagonal principal
!	c - sup-diagonal (diagonal acima da diagonal principal)
!	d - parte à direita
!	x - resposta
!	n - número de equações
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
 REAL (KIND=r8), PUBLIC, ALLOCATABLE :: B   (:)
 REAL (KIND=r8), PUBLIC, ALLOCATABLE :: X   (:)
 REAL (KIND=r8), PUBLIC, ALLOCATABLE :: xa(:) 
 REAL (KIND=r8), PUBLIC, ALLOCATABLE :: ua(:) 
 REAL (KIND=r8), PUBLIC :: xb0=100.0
 REAL (KIND=r8), PUBLIC :: xf0=400.0
 REAL (KIND=r8), PUBLIC :: tb0 =0
 REAL (KIND=r8), PUBLIC :: tf0 =0
 REAL (KIND=r8), PUBLIC :: xxb
 REAL (KIND=r8), PUBLIC :: yyf
 REAL (KIND=r8), PUBLIC :: Area
 REAL (KIND=r8) :: DeltaX
 REAL (KIND=r8) :: DeltaT
 INTEGER      , PUBLIC              :: ilo
 INTEGER      , PUBLIC              :: ihi
 INTEGER      , PUBLIC              :: iMax 
 PUBLIC ::  Init_Class_Fields,AnaliticFunction

 CONTAINS 
 
SUBROUTINE Init_Class_Fields(nx,dx,dt, coef_C,u_cte )
 IMPLICIT NONE
 INTEGER        , INTENT (IN   ) :: nx
 REAL (KIND=r8) , INTENT (IN   ) :: dx
 REAL (KIND=r8) , INTENT (IN   ) :: dt
 REAL (KIND=r8) , INTENT (IN   ) :: coef_C
 REAL (KIND=r8) , INTENT (IN   ) :: u_cte
 REAL (KIND=r8) :: coord_X(0:nx)
 INTEGER :: i,xb(1),xf(1)
 REAL (KIND=r8):: t
 REAL (KIND=r8),ALLOCATABLE    :: diff(:)
 iMax=nx
 ALLOCATE (A0   (0:iMax)       )
 ALLOCATE (A    (0:iMax)       )
 ALLOCATE (Anew (0:iMax)       )
 ALLOCATE (AA   (0:iMax+1,0:iMax+1))
 ALLOCATE (B    (0:iMax)       )
 ALLOCATE (X    (0:iMax)       )
 if (.not. allocated(ua)) ALLOCATE (ua(0:iMax))
 ua=0.0
 if (.not. allocated(xa)) ALLOCATE (xa(0:iMax))
 xa(0:iMax)=0.0
 if (.not. allocated(diff)) ALLOCATE (diff(0:iMax))
 diff(0:iMax)=0.0
 DeltaX=dx
 DeltaT=dt
 PRINT*,'DeltaX=',DeltaX,'DeltaT=',DeltaT,'CFL=',coef_C*DeltaT/DeltaX
!# python is zero-based.  We are assuming periodic BCs, so 
!# points 0 and N-1 are the same.  Set some integer indices to 
!# allow us to easily access points 1 through N-1.  Point 0 
!# won't be explicitly updated, but rather filled by the BC 
!# routine. 
 ilo = 1 
 ihi = iMax-1
 DO i=0,iMax 
    coord_X(i) = REAL(i,KIND=r8)*dx
    xa(i)      = REAL(i,KIND=r8)*DeltaX
 END DO 
 xb0=xa(iMax)/4.0
 xf0=xa(iMax)/2.0
 tb0 =0
 tf0 =0
 t=0
 xxb= xb0 + u_cte*(t-tb0)
 yyf= xf0 + u_cte*(t-tf0)
 DO i=0,iMax
    IF(xa(i) >xxb .and. xa(i) <yyf)THEN
       ua(i)=1.0
    ELSE
       ua(i)=0.0
    END IF
 END DO
 diff=ABS(xa-xxb)
 xb=MINLOC(diff(0:iMax)) 
 diff=ABS(xa-yyf)
 xf=MINLOC(diff(0:iMax)) 
 Area=MAX(( ua(xf(1)-2)-ua(xb(1)-1))*(xa(xf(1)-2)-xa(xb(1)-1))/(xf(1)-2-xb(1)-1),1.0)

 DO i=0,iMax
    IF(ua(i) ==1.0)THEN
         ua(i)=Area

    END IF
 END DO

 !
 ! initialize the data -- tophat
 !
 A=ua
 AA=0.0_r8
 !
 ! """ we don't explicitly update point 0, since it is identical 
 !            to N-1, so fill it here """ 
 A(0) = A(ihi)

 !
 !  a(t+1,i)  -   a(t,i)                 a(t+1,i) - a(t+1,i-1)
 !  -----------------------  = -  U ----------------------------------
 !          Dt                                    Dx
 !
 !                                    -                     -
 !                               Dt  |                       |
 !  a(t+1,i)  -  a(t,i)  =  - U ---- | a(t+1,i) - a(t+1,i-1) |
 !                                Dx |                       |
 !                                    -                     -

 !                                 -                     -
 !                                |                       |
 !  a(t+1,i)  -  a(t,i)  =  - C * | a(t+1,i) - a(t+1,i-1) |
 !                                |                       |
 !                                 -                     -
 ! 
 !  a(t+1,i)  -  a(t,i)  =  - C* a(t+1,i) + C*a(t+1,i-1) 
 ! 
 !  a(t+1,i) + C*a(t+1,i) - C*a(t+1,i-1) =   a(t,i)
 !
 ! ................................................................
 !
 !  -C a(t+1,i-1)  + (1 + C) a(t+1,i)   =   a(t,i)
 !
 ! ................................................................
 !
 !  -C a(t+1,  0)  + (1 + C) a(t+1,1)   =   a(t,1)
 !  -C a(t+1,  1)  + (1 + C) a(t+1,2)   =   a(t,2)<---------------------------
 !  -C a(t+1,  2)  + (1 + C) a(t+1,3)   =   a(t,3)                           |
 !  -C a(t+1,  3)  + (1 + C) a(t+1,4)   =   a(t,4)                           |
 !  -C a(t+1,i-1)  + (1 + C) a(t+1,i)   =   a(t,i)                           |
 !                                                                           | equatios adv
 !                                                                           |
 !  ____                 ____   __           __                                |
 ! |                         | |               |                               |
 ! | (1 + C)      0     -C   | | a(t+1,i-1)    | =   a(t,1  ) <----------------
 ! | -C       (1 + C)    0   | | a(t+1,i  )    | =   a(t,2  )
 ! |  0          -C   (1 + C)| | a(t+1,i+1)    | =   a(t,i+1)
 ! |____                 ____| |__           __|
 !
 !              A                    X         =     B
 !
 !# create the matrix       
 !         # loop over rows [ilo,ihi] and construct the matrix.  This will 
 !         # be almost bidiagonal, but with the upper right entry also 
 !         # nonzero. 
 !
 ! ilo = 1 
 ! ihi = iMax-1 

  DO i =ilo,ihi
      AA(i,i+1) = 0.0_r8 
      AA(i,i  ) = 1.0_r8 + coef_C 
      AA(i,i-1) = -coef_C
  END DO

 ! |  0       0          0  | | a(t+1,i-1)    | =   a(t,1  ) <----------------
 ! | (1 + C)  -c         0  | | a(t+1,i  )    | =   a(t,2  )
 ! |  0          0       0  | | a(t+1,i+1)    | =   a(t,i+1)

  AA(ilo,ilo+1  ) = -coef_C
  AA(ilo,ilo    ) =  AA(ihi,ihi    ) 
  AA(ilo,ilo-1  ) =0

  END SUBROUTINE Init_Class_Fields


!   
  FUNCTION AnaliticFunction(termX,C,it)  RESULT (ok)

      REAL(KIND=r8), INTENT(INOUT) :: termX(0:iMax)
      REAL(KIND=r8), INTENT(IN   ) :: C
      INTEGER, INTENT(IN   ) :: it
      INTEGER          :: i2,xb,xc,xf,i
      INTEGER         :: ok
      REAL(KIND=r8)    :: t
      t=(it)*DeltaT

      yyf= xf0 + C*(t-tf0)
      IF(yyf >= xa(iMax))THEN
         xf0=0.0
         yyf=xf0
         tf0=t
      END IF 

      xxb= xb0 + C*(t-tb0)
      IF(xxb >= xa(iMax))THEN
         xb0=0.0
         xxb=xb0
         tb0=t
      END IF 
      IF(xf0 <= xb0 .and. yyf <= xxb) THEN  
         DO i=1,iMax
            IF(xa(i) > xxb )THEN
               termX(i)=Area
            ELSE IF( xa(i) < yyf )THEN
               termX(i)=Area
            ELSE
               termX(i)=0.0
            END IF
         END DO
      ELSE
         DO i=1,iMax
            IF(xa(i) > xxb .and. xa(i) < yyf)THEN
               termX(i)=Area
            ELSE
               termX(i)=0.0
            END IF
         END DO
      END IF
    ok=0
   END FUNCTION AnaliticFunction
END MODULE Class_Fields



MODULE Class_WritetoGrads
 USE Class_Fields, Only: Anew,A,ua,iMax
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
   FileName='ImplicitLinearAdvection1D'
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
    Yout=REAL(ua(1:iMax),KIND=r4)
    irec=irec+1
    WRITE(UnitData,rec=irec)Yout
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
    WRITE (UnitCtl,'(A           )')'vars 2'
    WRITE (UnitCtl,'(A           )')'FA  0 99 resultado analitico da edol yc'
    WRITE (UnitCtl,'(A           )')'FC  0 99 resultado numerico  da edol yc'
    WRITE (UnitCtl,'(A           )')'endvars'
    CLOSE (UnitCtl,STATUS='KEEP') 
    CLOSE (UnitData,STATUS='KEEP') 
    ok=0   
 END FUNCTION SchemeWriteCtl


END MODULE Class_WritetoGrads
PROGRAM Main
 USE  Class_Fields, Only : Init_Class_Fields,B,A,AA,ua,ilo,ihi,ua,Anew,AnaliticFunction
 USE  LinearSolve, OnLy : solve_tridiag
 USE  Class_WritetoGrads, OnLy :schemeWriteCtl, schemeWriteData,initClass_WritetoGrads

 ! main program to check the Tridiagonal system solver
 INTEGER, PARAMETER :: r8 = selected_real_kind(15, 307)
 INTEGER, PARAMETER :: r4 = selected_real_kind(6, 37)
 INTEGER      , PARAMETER :: nn=200
 INTEGER      , PARAMETER :: nx=nn
 REAL (KIND=8) , PARAMETER :: xmin=0.0
 REAL (KIND=8) , PARAMETER :: xmax=1.0
 REAL(KIND=8)  , PARAMETER :: u = 10.0
 REAL (KIND=8) , PARAMETER :: Dx=(xmax - xmin)/(nx-1)
 REAL (KIND=8) , PARAMETER :: C=0.5    !=(u*dt/dx)    ! # CFL number

 REAL(KIND=8) , PARAMETER :: Dt=C*Dx/u
 INTEGER      , PARAMETER :: ninteraction=400
 INTEGER :: i,j

 CALL Init()
 CALL run()
 STOP

 CONTAINS

 SUBROUTINE Init()
  IMPLICIT NONE
    CALL Init_Class_Fields(nx,dx,dt, C,u)
    CALL initClass_WritetoGrads()
 END SUBROUTINE Init
 
SUBROUTINE Run()
  IMPLICIT NONE
  REAL (KIND=r8) :: a_sub_diagonal   (0:nx)
  REAL (KIND=r8) :: b_pri_diagonal   (0:nx)
  REAL (KIND=r8) :: c_sup_diagonal   (0:nx)
  INTEGER :: test,irec
  !ilo = 1 
  !ihi = iMax-1
  DO i =ilo,ihi
!     cc(i)= 0
!     b (i)= 1.0 + C 
!     a (i)= - C 
      a_sub_diagonal(i) = AA(i,i-1)  ! -C         !AA(i,i-1)
      b_pri_diagonal(i) = AA(i,i  )  ! 1.0_r8 + C !AA(i,i  )
      c_sup_diagonal(i) = AA(i,i+1)  ! 0.0_r8     !AA(i,i+1)
  END DO
  DO i =ilo,ihi
    WRITE(*,"(3F6.2)")c_sup_diagonal(i), &
                      b_pri_diagonal(i), &
                      a_sub_diagonal(i)
  END DO 
   irec=0
  DO i=1,ninteraction
        test=AnaliticFunction(ua,u,i)

        ! create the RHS -- this holds all entries except for a[0] 
        
        B    (ilo:ihi)  = A(ilo:ihi) 
        ! tridag(a,b,c,d,nn)
        ! PRINT*,A
        ! PRINT*," ! tridag(a,b,c,d,nn)tridag(a,b,c,d,nn)"
         Anew (ilo:ihi)  = 0.0_r8
         test=SchemeWriteData(irec)

       CALL solve_tridiag( a_sub_diagonal(ilo:ihi), &
                           b_pri_diagonal(ilo:ihi), &
                           c_sup_diagonal(ilo:ihi), &
                           B             (ilo:ihi), &
                           Anew          (ilo:ihi), &
                           ihi                      )
        !ihi = nx-1
        A   (ilo:ihi) = 0.0_r8
        A   (ilo:ihi) = Anew(ilo:ihi)
        !
        ! """ we don't explicitly update point 0, since it is identical 
        !            to N-1, so fill it here """ 
        IF(u > 0.0_r8)THEN
           A(ilo)   = A(ihi)  
        ELSE
           A(ihi)   = A(ilo)  
        END IF
  END DO    !   t += dt
  test=SchemeWriteCtl(ninteraction) 

 END SUBROUTINE Run
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 SUBROUTINE Finalize()
 
 END SUBROUTINE Finalize
END PROGRAM Main
