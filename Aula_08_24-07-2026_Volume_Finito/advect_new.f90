! solve the linear advection equation on a finite-volume grid using
! using Godunov's method (piecewise constant), piecewise linear, or
! piecewise parabolc reconstruction.
!
! u_t + a u_x = 0
!
! M. Zingale (2012-02-29)
!
! version 1.01 (2013-03-24):
!
!      bug fix in PLM slope computation -- now gets second-order for u
!      positive or negative, and a bug in the computation of the PPM
!      uminus -- we were missing the leftmost value
!
! To really test this for errors, compile as:
!
!   gfortran -g -O0 -fbacktrace -ffpe-trap=invalid,zero,overflow -finit-real=snan -Wuninitialized -o advect.x advect.f90
!

MODULE grid_module
  IMPLICIT NONE

  TYPE grid_t
      INTEGER ::  nx = -1
      INTEGER ::  ng = -1
      INTEGER :: ilo = -1
      INTEGER :: ihi = -1 

      DOUBLE PRECISION :: xmin
      DOUBLE PRECISION :: xmax
      DOUBLE PRECISION :: dx

      DOUBLE PRECISION, POINTER :: x(:) => NULL()
  END TYPE grid_t

  TYPE gridvar_t
      INTEGER                       :: nvar
      DOUBLE PRECISION, ALLOCATABLE :: d(:)
      TYPE(grid_t) :: gd
  END TYPE gridvar_t

CONTAINS

  SUBROUTINE build_grid(grid, nx, ng, xmin, xmax)

    TYPE(grid_t), INTENT(inout) :: grid
    INTEGER, INTENT(in) :: nx
    INTEGER, INTENT(in) :: ng
    DOUBLE PRECISION, INTENT(in) :: xmin
    DOUBLE PRECISION, INTENT(in) :: xmax

    INTEGER :: i

    grid%ilo = 0
    grid%ihi = nx - 1

    grid%nx = nx
    grid%ng = ng

    grid%xmin = xmin
    grid%xmax = xmax

    grid%dx = (xmax - xmin)/nx

    ALLOCATE(grid%x(-ng:nx+ng-1))

    DO i = grid%ilo - ng, grid%ihi + ng
       grid%x(i) = xmin + DBLE(i + 0.5d0) * grid%dx
    ENDDO

  END SUBROUTINE build_grid

  SUBROUTINE build_gridvar(gridvar, grid)

    TYPE(gridvar_t), INTENT(inout) :: gridvar
    TYPE(grid_t),    INTENT(in   ) :: grid

    IF (grid%nx == -1) THEN
       PRINT *, "ERROR: grid not initialized"
    ENDIF

    ! gridvar's grid type is simply a copy of the input grid
    gridvar%gd = grid

    ! now initialize the storage for the grid data
    ALLOCATE(gridvar%d(-grid%ng:grid%nx + grid%ng-1))
    gridvar % d(:) = 0.0d0

  END SUBROUTINE build_gridvar

  SUBROUTINE destroy_grid(grid)
    TYPE(grid_t), INTENT(inout) :: grid
    DEALLOCATE(grid % x)
  END SUBROUTINE destroy_grid

  SUBROUTINE destroy_gridvar(gridvar)
    TYPE(gridvar_t), INTENT(inout) :: gridvar
    DEALLOCATE(gridvar % d)
  END SUBROUTINE destroy_gridvar

END MODULE grid_module

PROGRAM advect

  USE grid_module

  IMPLICIT NONE

  ! the number of zones (nx) and number of guardcells (ng)
  INTEGER, PARAMETER :: nx = 64
  INTEGER, PARAMETER :: ng = 3
  ! the domain size
  DOUBLE PRECISION, PARAMETER :: xmin = 0.d0
  DOUBLE PRECISION, PARAMETER :: xmax = 1.d0
  ! the advection velocity
  DOUBLE PRECISION, PARAMETER :: u = 1.d0
  ! the CFL number
  DOUBLE PRECISION, PARAMETER :: C = 0.8d0
 !                           cfl = u*(dt/dx)
  DOUBLE PRECISION            :: dt
  INTEGER                     :: itr 
  INTEGER                     :: itr_max=100
  ! initial condition type
  INTEGER, PARAMETER          :: inittype = 2
  ! iinittype == 1! sin wave
  ! iinittype == 2! square wave
  ! iinittype == 3! wave packet
  ! iinittype == 4! gaussian
  INTEGER, PARAMETER :: islopetype = 1
  ! slope type (1=godunov, 2=plm, 3=ppm)

  INTEGER, PARAMETER :: plmlimiter = 1
  !IF (islopetype == 2) THEN
  !   IF (plmlimiter == 1) THEN
  !      slope = "plm+MC"
  !   ELSE IF (plmlimiter == 2) THEN
  !      slope = "plm+SBee"
  !   ENDIF
   !END IF
  LOGICAL :: First = .true.
  INTEGER :: irec = 0

  ! maximum simulation time
  DOUBLE PRECISION, PARAMETER :: tmax = 50.d0

  TYPE (grid_t) :: g
  TYPE (gridvar_t) :: a, al, ar, f, ainit

  DOUBLE PRECISION :: time

  ! setup the grid and set the initial conditions
  CALL build_grid(g, nx, ng, xmin, xmax)

  CALL build_gridvar(a    , g)
  CALL build_gridvar(ainit, g)
  CALL build_gridvar(al   , g)
  CALL build_gridvar(ar   , g)
  CALL build_gridvar(f    , g)

  CALL init(inittype, a)
  ainit % d(:) = a % d(:)

  time = 0.d0

  CALL output(inittype, islopetype, plmlimiter, a, time, u,First,irec)


  ! evolution loop -- construct the interface states, solve the Riemann
  ! problem, and update the solution to the new time level
  itr=0
  DO WHILE (itr < itr_max)

     CALL fillBC(a)

     CALL timestep(a, C, u, dt)
     IF (time + dt > tmax) THEN
        dt = tmax - time
     ENDIF

     CALL states(dt, u, islopetype, plmlimiter, a, al, ar)

     CALL riemann(u, al, ar, f)

     CALL update(dt, a, f)

     time = time + dt
     itr = itr +1
     PRINT*,'itr=',itr,'time=',time,'dt=',dt

     CALL output(inittype, islopetype, plmlimiter, a, time, u,First,irec)

  ENDDO


  PRINT *, 'N, error: ', g % nx, &
       SQRT(g % dx*SUM((a % d(a % gd %ilo:a % gd % ihi) - &
       ainit % d(a % gd % ilo:a % gd % ihi))**2))

END PROGRAM advect


!============================================================================
! init: set the initial conditions
!============================================================================
SUBROUTINE init(inittype, a)

  USE grid_module

  IMPLICIT NONE

  INTEGER, INTENT(in) :: inittype
  TYPE (gridvar_t), INTENT(inout) :: a

  INTEGER :: i
  DOUBLE PRECISION, PARAMETER :: pi = 3.14159d0

  ! loop over all the zones and set the initial conditions.  To be
  ! consistent with the finite-volume discretization, we should store
  ! the zone averages here, but, to second-order accuracy, it is
  ! sufficient to evaluate the initial conditions at the zone center.

  DO i = a % gd % ilo, a % gd % ihi

     IF (inittype == 1) THEN
        ! sin wave
        a % d(i) = SIN(2.d0*pi*a % gd % x(i))

     ELSE IF (inittype == 2) THEN
        ! square wave
        IF (a % gd % x(i) > 0.333d0 .AND. a % gd % x(i) < 0.666d0) THEN
           a % d(i) = 1.d0
        ELSE
           a % d(i) = 0.d0
        ENDIF

     ELSE IF (inittype == 3) THEN
        ! wave packet
        a % d(i) = SIN(16.d0*pi*a % gd % x(i)) * &
             EXP(-36.d0*(a % gd % x(i)-0.5d0)**2)

     ELSE IF (inittype == 4) THEN
        ! gaussian
        a % d(i) = EXP(-(a % gd % x(i) - 0.5d0)**2/0.1d0**2)

     ENDIF

  ENDDO

  RETURN
END SUBROUTINE init


!============================================================================
! output: write out the solution
!============================================================================
SUBROUTINE output(inittype, islopetype, plmlimiter, a, time, u,First,irec)

  USE grid_module

  IMPLICIT NONE

  INTEGER, INTENT(in) :: inittype, islopetype, plmlimiter
  TYPE (gridvar_t), INTENT(in) :: a
  DOUBLE PRECISION, INTENT(in) :: time, u
  CHARACTER (len=526) :: name
  LOGICAL             :: First
  INTEGER             :: irec
  CHARACTER (len=4) :: time_string
  CHARACTER (len=16) :: slope, init, res

  INTEGER :: i,lrec

  IF (islopetype == 1) THEN
     slope = "godunov"
  ELSE IF (islopetype == 2) THEN
     IF (plmlimiter == 1) THEN
        slope = "plm+MC"
     ELSE IF (plmlimiter == 2) THEN
        slope = "plm+SBee"
     ENDIF
  ELSE IF (islopetype == 3) THEN
     slope = "ppm"
  ENDIF

  IF (inittype == 1) THEN
     init = "sine"
  ELSE IF (inittype == 2) THEN
     init = "tophat"
  ELSE IF (inittype == 3) THEN
     init = "packet"
  ELSE IF (inittype == 4) THEN
     init = "gaussian"
  ENDIF

  IF(First)THEN

  ! open the output file
  WRITE (time_string, '(f4.2)') time
  WRITE (res, '(i8)') a % gd % nx
  name='advect_'//TRIM(slope)//'_'//TRIM(init)//'_nx_'//TRIM(ADJUSTL(res))//'_t_'//TRIM(time_string)
  !OPEN(unit=10, file=TRIM(name)//'.txt', status="unknown")

    INQUIRE(IOLENGTH=lrec) REAL(a % d(a % gd % ilo : a % gd % ihi),kind=4)
    OPEN(unit=20, file=TRIM(name)//'.bin', ACCESS='direct', FORM='UNFORMATTED',ACTION='WRITE',RECL=lrec, status="unknown")
    First=.FALSE.
  END IF 
  !WRITE (10,*) "# advection problem: a_t + u a_x = 0"
  !WRITE (10,*) "# u = ", u
  !WRITE (10,*) "# init = ", inittype
  !WRITE (10,*) "# slope type = ", islopetype
  !IF (islopetype == 2) THEN
  !   WRITE (10,*) "# plm limiter = ", plmlimiter
  !ENDIF
  !WRITE (10,*) "# time = ", time


  !DO i = a % gd % ilo, a % gd % ihi
  !   WRITE (10,*) a % gd % x(i), a % d(i)
  !ENDDO
  PRINT*,'a % gd % ilo=',a % gd % ilo,' a % gd % ihi=',a % gd % ihi,&
         'a % gd % ihi-a % gd % ilo+1=',a % gd % ihi-a % gd % ilo+1
  irec=irec+1
  WRITE(20,rec=irec) real(a % d(a % gd % ilo : a % gd % ihi),kind=4)
  RETURN
END SUBROUTINE output


!============================================================================
! fillBC: fill the boundary conditions
!============================================================================
SUBROUTINE fillBC(a)

  USE grid_module

  IMPLICIT NONE

  TYPE (gridvar_t), INTENT(inout) :: a

  INTEGER :: i, ilo, ihi, ng

  ilo = a%gd%ilo
  ihi = a%gd%ihi
  ng  = a%gd%ng

  ! left boundary
  DO i = 1, ng
     a % d(ilo - i) = a % d(ihi + 1 - i)
  ENDDO

  ! right boundary
  DO i = 1, ng
     a % d(ihi + i) = a % d(ilo - 1 + i)
  ENDDO

  RETURN
END SUBROUTINE fillBC


!============================================================================
! timestep: compute the new timestep
!============================================================================
SUBROUTINE timestep(a, cfl, u, dt)

  USE grid_module

  IMPLICIT NONE

  TYPE (gridvar_t), INTENT(in) :: a
  DOUBLE PRECISION, INTENT(in) :: cfl, u
  DOUBLE PRECISION, INTENT(inout) :: dt

  INTEGER :: i

  ! in the linear advection equation, the timestep is trivial
  dt = cfl  *  a%gd%dx/ABS(u)
  !dt = dt!cfl  *  a%gd%dx/ABS(u)

  RETURN
END SUBROUTINE timestep


!============================================================================
! states: compute the interface states used in solving the Riemann problem
!============================================================================
SUBROUTINE states(dt, u, islopetype, plmlimiter, a, al, ar)

  USE grid_module

  IMPLICIT NONE

  INTEGER, INTENT(in) :: islopetype, plmlimiter
  TYPE (gridvar_t), INTENT(in) :: a
  TYPE (gridvar_t), INTENT(inout) :: al, ar
  DOUBLE PRECISION, INTENT(in) :: dt, u

  TYPE (gridvar_t) :: slope, aminus, aplus
  DOUBLE PRECISION :: du0, dup
  DOUBLE PRECISION :: slope1, slope2

  DOUBLE PRECISION :: minmod, maxmod

  INTEGER :: i, ilo, ihi
  DOUBLE PRECISION :: dx

  CALL build_gridvar(slope , a % gd)
  CALL build_gridvar(aminus, a % gd)
  CALL build_gridvar(aplus , a % gd)

  ilo = a%gd%ilo
  ihi = a%gd%ihi
  dx  = a%gd%dx

  ! compute the centered difference for linear slopes
  IF (islopetype == 1) THEN

      ! Godunov's method (piecewise constant)

      ! for each interface, we want to construct the left and right
      ! states.  Here, interface i refers to the left edge of zone i 

      ! interfaces imin to imax+1 affect the data in zones [imin,imax]
      DO i = ilo, ihi+1

         ! the left state on the current interface comes from zone i-1.
         al%d(i) = a%d(i-1)
 
         ! the right state on the current interface comes from zone i
         ar%d(i) = a%d(i)
      ENDDO

  ELSE IF (islopetype == 2) THEN
      !
      ! Piecewise Linear Method (PLM)
      !
     IF (plmlimiter == 1) THEN

        ! Parabolic Linear PLM with MC limiter

        ! interface states are found by Taylor expansion in time
        ! (through dt/2) and space (dx/2 toward the interface)

        ! for each interface, we want to construct the left and right
        ! states.  Here, interface i refers to the left edge of zone i
        !
        DO i = ilo-1, ihi+1
           slope % d(i) = minmod(minmod(2.d0*(a % d(i  ) - a % d(i-1))/dx, &
                                        2.d0*(a % d(i+1) - a % d(i  ))/dx), &
                                       0.5d0*(a % d(i+1) - a % d(i-1))/dx)
        ENDDO

     ELSE IF (plmlimiter == 2) THEN
        !
        ! PLM with SuperBee limiter
        !
        DO i = ilo-1, ihi+1
           slope1 = minmod((a % d(i+1) - a % d(i  ))/dx, &
                      2.d0*(a % d(i  ) - a % d(i-1))/dx)

           slope2 = minmod(2.d0*(a % d(i+1) - a % d(i  ))/dx, &
                                (a % d(i  ) - a % d(i-1))/dx)

           slope % d(i) = maxmod(slope1, slope2)
        ENDDO

     ENDIF
      !
      ! interfaces ilo to ihi+1 affect the data in zones [ilo, ihi]
      !
     DO i = ilo, ihi+1

        ! the left state on the current interface comes from zone i-1.

        al % d(i) = a % d(i-1) + 0.5*dx*(1.d0 - u*(dt/dx))*slope % d(i-1)

        ! the right state on the current interface comes from zone i

        ar % d(i) = a % d(i  ) - 0.5*dx*(1.d0 + u*(dt/dx))*slope % d(i)

     ENDDO

  ELSE IF (islopetype == 3) THEN
      !
      ! Piecewise Parabolic Method (PPM)

      ! refer to the PPM paper for equation references

      ! use a cubic interpolation polynomial to find the edge states in
      ! zone i -- aminus(i) and aplus(i).  Here we loop over
      ! interfaces.
     DO i = ilo-2, ihi+1

        ! du (C&W Eq. 1.7)
        du0 = 0.5*(a % d(i+1) - a % d(i-1))
        dup = 0.5*(a % d(i+2) - a % d(i))

        ! limiting (C&W Eq. 1.8)
        IF ((a % d(i+1) - a % d(i))*(a % d(i) - a % d(i-1)) > 0) THEN
           du0 = SIGN(1.0d0, du0)*MIN(ABS(du0), &
                2.0*ABS(a % d(i) - a % d(i-1)), &
                2.0*ABS(a % d(i+1) - a % d(i)))
        ELSE
           du0 = 0.0
        ENDIF

        IF ((a % d(i+2) - a % d(i+1))*(a % d(i+1) - a % d(i)) > 0) THEN
           dup = SIGN(1.0d0, dup)*MIN(ABS(dup), &
                2.0*ABS(a % d(i+1) - a % d(i)), &
                2.0*ABS(a % d(i+2) - a % d(i+1)))
        ELSE
           dup = 0.0
        ENDIF

        ! cubic (C&W Eq. 1.6)
        aplus  % d(i  ) = 0.5*(a % d(i) + a % d(i+1)) - (1.0/6.0)*(dup - du0)
        aminus % d(i+1) = aplus % d(i)

     ENDDO

      ! now limit (C&W 1.10).  Here the loop is over cells, and
      ! considers the values on either side of the center of the cell
      ! (uminus and uplus)

     DO i = ilo-1, ihi+1
        IF ( (aplus % d(i) - a % d(i)) * &
             (a % d(i) - aminus % d(i)) <= 0.0) THEN

              aminus % d(i) = a % d(i)
              aplus  % d(i) = a % d(i)

        ELSE IF ( (aplus % d(i) - aminus % d(i)) * &
             (a % d(i) - 0.5*(aminus % d(i) + aplus % d(i))) > &
             (aplus % d(i) - aminus % d(i))**2/6.0 ) THEN

           aminus % d(i) = 3.0*a % d(i) - 2.0*aplus % d(i)

        ELSE IF ( -(aplus % d(i) - aminus % d(i))**2/6.0 > &
             (aplus % d(i) - aminus % d(i)) * &
             (a % d(i) - 0.5*(aminus % d(i) + aplus % d(i))) ) THEN

           aplus % d(i) = 3.0*a % d(i) - 2.0*aminus % d(i)

        ENDIF

     ENDDO

      ! finally integrate under the parabola (C&W Eq. 1.12), away from
      ! the interface.  Here the loop is over interfaces.  al and ar are
      ! the left and right states at the interface.
     DO i = ilo, ihi+1

        al % d(i) = aplus % d(i-1) - 0.5*u*(dt/dx)*( (aplus % d(i-1) - aminus % d(i-1)) - &
             (1.0 - (2.0/3.0)*u*(dt/dx))*6.0*(a % d(i-1) - 0.5*(aminus % d(i-1) + aplus % d(i-1))) )

        ar % d(i) = aminus % d(i) + 0.5*u*(dt/dx)*( (aplus % d(i) - aminus % d(i)) + &
             (1.0 - (2.0/3.0)*u*(dt/dx))*6.0*(a % d(i) - 0.5*(aminus % d(i) + aplus % d(i))) )

     ENDDO

  ENDIF

  RETURN
END SUBROUTINE states



!============================================================================
! riemann: solve the Riemann problem
!============================================================================
SUBROUTINE riemann(u, al, ar, f)

  USE grid_module

  IMPLICIT NONE

  DOUBLE PRECISION, INTENT(in) :: u
  TYPE (gridvar_t), INTENT(in) :: al, ar
  TYPE (gridvar_t), INTENT(inout) :: f

  INTEGER :: i

  ! loop over all the interfaces and solve the Riemann problem.  Here,
  ! since we are doing the linear advection eq, we just use the advection
  ! velocity to tell us which direction is upwind, and use that state

  IF (u >= 0.d0) THEN
     DO i = f % gd % ilo, f % gd % ihi+1
        f % d(i) = u * al % d(i)
     ENDDO

  ELSE
     DO i = f % gd % ilo, f % gd % ihi+1
        f % d(i) = u * ar % d(i)
     ENDDO

  ENDIF

  RETURN
END SUBROUTINE riemann



!============================================================================
! update: conservatively update the solution to the new time level
!============================================================================
SUBROUTINE update(dt, a, f)

  USE grid_module

  IMPLICIT NONE

  DOUBLE PRECISION, INTENT(in) :: dt
  TYPE (gridvar_t), INTENT(in) :: f
  TYPE (gridvar_t), INTENT(inout) :: a

  INTEGER :: i

  DO i = a % gd % ilo, a % gd % ihi
     a % d(i) = a % d(i) + (dt/a % gd % dx)*(f % d(i) - f % d(i+1))
  ENDDO

  RETURN
END SUBROUTINE update



!============================================================================
! various limiter functions
!============================================================================
FUNCTION minmod(a,b)

  IMPLICIT NONE

  DOUBLE PRECISION :: a, b
  DOUBLE PRECISION :: minmod

  IF (ABS(a) < ABS(b) .AND. a*b > 0.d0) THEN
     minmod = a
  ELSE IF (ABS(b) < ABS(a) .AND. a*b > 0) THEN
     minmod = b
  ELSE
     minmod = 0.d0
  ENDIF

  RETURN
END FUNCTION minmod



FUNCTION maxmod(a,b)

  IMPLICIT NONE

  DOUBLE PRECISION :: a, b
  DOUBLE PRECISION :: maxmod

  IF (ABS(a) > ABS(b) .AND. a*b > 0.d0) THEN
     maxmod = a
  ELSE IF (ABS(b) > ABS(a) .AND. a*b > 0) THEN
     maxmod = b
  ELSE
     maxmod = 0.d0
  ENDIF

  RETURN
END FUNCTION maxmod
