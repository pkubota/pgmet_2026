      module baromod
      implicit none
!
!     module baromod contains all global parameters/variables 
!     (i.e. which might be needed in  more that one subroutine)
!     to be included be *use baromod*
!
!     a) constants:
!
      integer,parameter :: NX = 64         ! x dimension
      integer,parameter :: NY = 32         ! y dimension
      real,parameter :: radea = 6.371E6    ! radius of the earth [m]
      real,parameter :: omega = 0.00007292 ! angular velocity [1/s] 
!
      integer :: nrun     = 240        !  no timesteps to be computed
      integer :: nout     = 1        !  output interval 
!
      real :: rlat    = 50.          ! central latitude of the channel
      real :: xchannel= 360.         ! channel (x-) length [dec]
      real :: ychannel= 40.          ! channel (y-) width  [dec]
      real :: delt    = 1200.        ! time step [s]
!
!     b) variables
!
      integer :: nstep    = 0        ! time step counter
      integer :: nwout    = 0        ! output counter
!      
      real :: s(0:NX+1,0:NY+1)   = 0. ! streamfct. [m**2/2]
      real :: sm(0:NX+1,0:NY+1)  = 0. ! streamfct. (old timestep) [m**2/s]
      real :: dsdt(0:NX+1,0:NY+1)= 0. ! streamfct tendnecy (ds/dt) [m**2/s**2] 
      real :: vo(0:NX+1,0:NY+1)  = 0. ! vorticity [1/s]
!
      real :: pi                      ! pi
      real :: beta                    ! beta parameter
      real :: dx                      ! gridpoint distance in x [m]
      real :: dy                      ! gridpopint distance in y [m]
      real :: rossby                  ! Rossby number
      real :: f0                      ! corriolis par. at cent.latitude [1./s]
      real :: delt2                   ! 2*delt [s] 
!
      end module baromod
!
!----------------------------------------------------------------------
!
      program baro
      use baromod
!
!
!     program baro organizes the model run by calling individual subroutines
!     for initialization, time stepping, output and finalization    
!
      integer :: jstep     ! loop index
!
!     a) initialization
!
      call baroini
!
!     b) time stepping + output
!
      do jstep=1,nrun
       call barostep()
       nstep=nstep+1
       if(mod(nstep,nout)==0) call baroout
      enddo
!
!     c) finalizing
!
      call barostop()
!
      stop
      end
!
!----------------------------------------------------------------------
!
      subroutine baroini
      use baromod
!
!     subroutine baroini initializes the run
!     
!
!
      real :: zlat        ! central latitude in rad
      real :: zxch        ! channel length in m
      real :: zych        ! channel width in m
      real :: zdelt       ! short initial time step for euler
      integer :: kits = 3 ! no short initial time steps
!
!     set some constants
!       
      pi=2.*asin(1.)
      zlat=pi*rlat/180.
      zxch=pi*radea*cos(zlat)*xchannel/180.
      zych=pi*radea*ychannel/180.
!
!     define grid (by dx,dy)
!
      dx=zxch/real(nx)
      dy=zych/real(ny+1)
!
!     set coriolis parameter amd beta according to central latitude
!
      f0=2.*omega*sin(zlat)
      beta=2.*omega*cos(zlat)/radea
!
!     print grid information  
!
      print*,' f0       = ',f0
      print*,' beta     = ',beta
      print*,' delta x  = ',dx
      print*,' delta y  = ',dy
!
!     compute rossby wave phase velocity and related courant number
!
      print*,' Rossby Cp   = ',beta*zxch*zxch/(2.*pi)**2
      print*,' Rossby C-No = ',(beta*zxch*zxch/(2.*pi)**2)*delt/dx
!
!     set delt2 for leapfrog
!
      delt2=2.*delt
!
!     open output file
!
      open(10,file='barogp',form='unformatted',status='unknown',ACTION='WRITE')
!
!     set the initial field(s)
!
      call initial
!
!     do initial (short) euler time steps to obtain values at t=delt
!
      zdelt=delt/2**(kits)
      do js=1,kits
       zdelt=zdelt+zdelt
       call mkvort(s,vo,dx,dy,NX,NY)
       call tendency()
       call euler(sm,dsdt,s,zdelt,NX,NY)
       call boundary(s,NX,NY)
      enddo
!
      return
      end
     
      subroutine RungeKutta4()
      use baromod
      REAL :: k1 (0:NX+1,0:NY+1)
      REAL :: k2 (0:NX+1,0:NY+1)
      REAL :: k3 (0:NX+1,0:NY+1)
      REAL :: k4 (0:NX+1,0:NY+1)
      REAL :: dt
      !real :: dsdt(0:NX+1,0:NY+1)= 0. ! streamfct tendnecy (ds/dt) [m**2/s**2] 
      dt=delt2/2
      sm=s
!____________________________________________________
!     subroutine barostep does the time steping
!
!     a) compute vorticity from streamfunction
!
      call mkvort(s,vo,dx,dy,NX,NY)
!
!     b) compute streamfunction tendencies
!
      call tendency()
!
!     c) do the leapfrog timestep
!
      k1= dsdt

!____________________________________________________
!     a) compute vorticity from streamfunction

      call mkvort(s + 0.5*dt*k1 ,vo ,dx,dy,NX,NY)
!
!     b) compute streamfunction tendencies
!
      call tendency()
!
!     c) do the leapfrog timestep
!
      k2= dsdt
!____________________________________________________
!     a) compute vorticity from streamfunction

      call mkvort(s + 0.5*dt*k2 ,vo ,dx,dy,NX,NY)
!
!     b) compute streamfunction tendencies
!
      call tendency()
!
!     c) do the leapfrog timestep
!
      k3= dsdt

!____________________________________________________
!     a) compute vorticity from streamfunction

      call mkvort(s + 1.0*dt*k3 ,vo ,dx,dy,NX,NY)
!
!     b) compute streamfunction tendencies
!
      call tendency()
!
!     c) do the leapfrog timestep
!
      k4= dsdt

      s=0.0  
                                                    !  1    2	 2    1     1+2+2+1
      s = sm + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )  ! ---+ ---+ ---+ --- = --------
                                                    !  6    6	 6    6        6
!
!     d) apply boundary conditions to streamfunction
!
      call boundary(s,NX,NY)

      end subroutine RungeKutta4
!
!----------------------------------------------------------------------
!
      subroutine leapfrog()
        use baromod
!
!     subroutine barostep does the time steping
!
!     a) compute vorticity from streamfunction
!
      call mkvort(s,vo,dx,dy,NX,NY)
!
!     b) compute streamfunction tendencies
!
      call tendency()
!
!     c) do the leapfrog timestep
!
      call leapfrog_update(s,sm,dsdt,delt2,NX,NY)
!
!     d) apply boundary conditions to streamfunction
!
      call boundary(s,NX,NY)
!
      return

      end subroutine leapfrog
!
!----------------------------------------------------------------------
!
      
      subroutine barostep()
      use baromod
       CHARACTER(LEN=20) :: scheme      = 'RungeKutta4'!s Possible values:
       !CHARACTER(LEN=20) :: scheme      = 'leapfrog'!s Possible values:

       if(TRIM(scheme) == 'RungeKutta4')then
         call RungeKutta4()
       else if(TRIM(scheme) == 'leapfrog')then
         call leapfrog()
       else
         print*,'erros',TRIM(scheme) 
         stop 
       endif
      return
      end
!
!----------------------------------------------------------------------
!
      subroutine barostop()
      use baromod
!
!     subroutine barostop finelizes the run
!
!     close output files
!
      close(10)
!
!     write grads control file
!
      call gradsinfo 
!
      print*,'****************************************************'
      print*,'Run finished at nstep= ',nstep
      print*,'****************************************************'
!
      return
      end
!
!----------------------------------------------------------------------
!
      subroutine baroout
      use baromod
!
!     subroutine baroout writes fields to output file
!
      real :: zu(NX,NY),zv(NX,NY) ! u and v (velocities)
!
!     comput u and v from streamfunction
!
      call mkuv(s,zu,zv,dx,dy,NX,NY)
!
!     write u
!
      write(10) zu(1:NX,1:NY)
!
!     write v
!
      write(10) zv(1:NX,1:NY)
!
!     write streamfct.
!
      write(10) s(1:NX,1:NY)
!
!     write geopotential height
!
      write(10) s(1:NX,1:NY)*f0/9.81
!
!     write vorticity
!
      write(10) vo(1:NX,1:NY)
!
!     advance counter
!
      nwout=nwout+1
!
      return
      end
!
      subroutine mkuv(ps,pu,pv,pdx,pdy,kx,ky)
      implicit none
!
!     subroutine mkuv computs u and v from streamfunction
!
      integer :: kx                  ! x-dimension
      integer :: ky                  ! y-dimension
      real :: pdx                    ! grid point distance x [m]
      real :: pdy                    ! grid point distance y [m]
      real :: ps(0:kx+1,0:ky+1)      ! streamfunction [m**2/s]
      real :: pu(kx,ky)              ! u [m/s]
      real :: pv(kx,ky)              ! v [m/s]
      integer :: i,j                 ! loop indizes
!
      do j=1,ky
       do i=1,kx
        pu(i,j)=-0.5*(ps(i,j+1)-ps(i,j-1))/pdy
        pv(i,j)= 0.5*(ps(i+1,j)-ps(i-1,j))/pdx
       enddo
      enddo
!
      return
      end
!
!----------------------------------------------------------------------
!
      subroutine tendency()
      use baromod
      !real, INTENT(inout) :: dsdt(0:NX+1,0:NY+1) ! streamfct tendnecy (ds/dt) [m**2/s**2] 

!
!     subroutine tendency computes the streamfunction tendency 
!
      real :: zdt(0:NX+1,0:NY+1) ! vorticity tendency
      real :: zj(0:NX+1,0:NY+1)  ! jacobian (advection of rel vort.)
      real :: zv(0:NX+1,0:NY+1)  ! y-velocity v (ds/dx)
!
!     compute jacobian J(s,vo) (adv. of rel. vort.)
!
      call jacobi(s,vo,zj,dx,dy,NX,NY)
!
!     compute v (ds/dx)
!
      call mkdfdx(s,zv,dx,NX,NY)
!
!     add advection of rel. and planetray vorticity to get vort. tendency
!
      zdt(:,:)=-zj(:,:)-beta*zv(:,:)
!
!     compute inverse Laplacian to get streamfunction tendency
!
      call sor(zdt,dsdt,dx,dy,NX,NY)
!
      return
      end
!
!----------------------------------------------------------------------
!
      subroutine mkvort(ps,pvo,pdx,pdy,kx,ky)
      implicit none
!
!     subroutine mkvort computes the vorticity from a given streamfunction
!
      integer :: kx                 ! x-dimension
      integer :: ky                 ! y-dimension
      real :: pdx                   ! x grid distance
      real :: pdy                   ! y grid distance
      real :: ps(0:kx+1,0:ky+1)     ! input: stream function
      real :: pvo(0:kx+1,0:ky+1)    ! output: vorticity
!      integer :: i,j                ! loop indizes
!
!     compute Laplacian
!
      call laplace(ps,pvo,pdx,pdy,kx,ky)
!
!     make boundary conditions
!
      call boundary(pvo,kx,ky)
!
      return
      end
!
!-----------------------------------------------------------------------     
!
      subroutine boundary(ps,kx,ky)
      implicit none      
!
!     subroutine boundary set boundary conditions (cyclic in x)
!
      integer :: kx,ky   ! dimensions
      real :: ps(0:kx+1,0:ky+1) ! input field
!
      ps(0,:)=ps(kx,:)
      ps(kx+1,:)=ps(1,:)
      ps(:,0)=0.
      ps(:,ky+1)=0.
!
      return
      end
!
!----------------------------------------------------------------------     
!
      subroutine initial
      use baromod
!
!     subroutine initial sets the initial streamfunktion field
!     here: a wave with wave numbers zk and zl (see below)
!
      real :: za = 10.*1.E6 ! initial amplitude [m**2/s]
      real :: zk = 1.       ! wave number in x-direction
      real :: zl = 0.5      ! wave number in y-direction
      integer :: i,j        ! loop indizes
!
!     set initial streamfunction
!
      do j=1,NY
       do i=1,NX
        s(i,j)=za*sin(2.*pi*real(i-1)*zk/real(NX))                      &
     &        *sin(2.*pi*real(j)*zl/real(NY+1))
       enddo
      enddo
!
!     set boundary conditions
!
      call boundary(s,NX,NY)
!
!     copy to sm
!
      sm(:,:)=s(:,:)
!
!     compute vorticity and write initial field to output
!
      call mkvort(s,vo,dx,dy,NX,NY)      
      call baroout
!
      return
      end
!
!-----------------------------------------------------------------------     
!
      subroutine gradsinfo
      use baromod
!
      real :: zy1,zdy
!
!     open file
!
      open(10,file='barogp.ctl',form='formatted')
!
!     write grads control file
!
      zdy=ychannel/real(ny+1)
      zy1=rlat-ychannel/2.+zdy
      write(10,1) 'DSET ^barogp'
      write(10,1) 'UNDEF 9E+09'
      write(10,1) 'OPTIONS SEQUENTIAL'
      write(10,2) 'XDEF ',NX,' LINEAR ',0.,' ',xchannel/real(NX)
      write(10,2) 'YDEF ',NY,' LINEAR ',zy1,' ',zdy
      write(10,1) 'ZDEF 1 LINEAR 1 1'
      write(10,3) 'TDEF ',nwout,' LINEAR 00:00Z01jan0001 20mn'
      write(10,1) 'VARS 5'
      write(10,1) 'U 0 99 zonal velocity'
      write(10,1) 'V 0 99 meridional velocity'
      write(10,1) 'S 0 99 streamfunction'
      write(10,1) 'Z 0 99 geopotential height'
      write(10,1) 'VO 0 99 vorticity'
      write(10,1) 'ENDVARS'
!
!     close file
!
      close(10)
!
      return
    1 format(A)
    2 format(A,I5,A,F10.5,A,F10.5)
    3 format(A,I5,A)
      end
!
!=======================================================================
!  
!     the folowing subroutines are just dummies and need to be
!     replaced 
!
      subroutine sor(pdf,pf,pdx,pdy,kx,ky)
      implicit none
!
!     subroutine sor compute the inverse Laplacian from a given field
!     by using the Successive OverRelaxation method (SOR)
!
      real,parameter :: wsor = 1.50 ! overrelaxaton factor
      real,parameter :: eps = 1.E-4 ! minimum reduction of error to stop iter.
!
      integer :: kx                ! x dimension
      integer :: ky                ! y dimension
      integer :: niter             ! maximum number of iterations
      real :: pdx                  ! x grid point distance
      real :: pdy                  ! y grid point distance
      real :: pdf(0:kx+1,0:ky+1)   ! input: field 
      real :: pf(0:kx+1,0:ky+1)    ! outpout: inverse Laplacian of input
!
      real :: zn(0:kx+1,0:ky+1)    ! work array: new result
      real :: zo(0:kx+1,0:ky+1)    ! work array: old result
      real :: zfac                 ! help
      real :: zzres                ! local error
      real :: zresf                ! initial error
      real :: zres                 ! actual error
      integer :: i,j,jiter         ! loop indizes
!
!     set maximum number of iteration 
!     (note: sor should need about sqrt(kx*ky)/3*log10(1/eps) iterations to 
!      decrease the initial error by factor eps... but it takes more(?))
!
      niter=ky*kx*NINT(log10(1./eps)+0.5)/3 
!
      zfac=2./(pdx*pdx)+2./(pdy*pdy)
!
!     copy first guess to work (here first guess is 0.)
!
      zn(:,:)=0. ! pf(:,:)
!
!     compute initial error
!
      zresf=0.
      do j=1,ky
       do i=1,kx
        zzres=-zfac*zn(i,j)+(zn(i-1,j)+zn(i+1,j))/(pdx*pdx)            &
     &       +(zn(i,j-1)+zn(i,j+1))/(pdy*pdy)-pdf(i,j)
        zresf=zresf+abs(zzres)/real(kx*ky)
       enddo
      enddo
!
!     do the iteration
!
      do jiter=1,niter
       zo(:,:)=zn(:,:)
       zres=0.
       do j=1,ky
        do i=1,kx
         zzres=-zfac*zo(i,j)+(zn(i-1,j)+zo(i+1,j))/(pdx*pdx)            &
     &        +(zn(i,j-1)+zo(i,j+1))/(pdy*pdy)-pdf(i,j)
         zn(i,j)=zo(i,j)+wsor*zzres/zfac
         zres=zres+abs(zzres)/real(kx*ky)
        enddo
       enddo
       call boundary(zn,kx,ky)
       if(zres <= eps*zresf) exit
      enddo
!
      if(jiter >= niter) then 
       print*,'MAXIMUM iterations needed',niter
       print*,'error, initial error, eps*ie= ',zres,zresf,eps*zresf
      endif
!
!     copy result to output
!
      pf(:,:)=zn(:,:)
!
      return
      end

!
!-----------------------------------------------------------------------
!
      subroutine mkdfdx(pf,pdfdx,pdx,kx,ky)
      implicit none
!
!     subroutine mkdfdx computes x derivation from a field 
!     using central differences
!
      integer :: kx                 ! x-dimension
      integer :: ky                 ! y-dimension
      real :: pdx                   ! x grid distance
      real :: pf(0:kx+1,0:ky+1)     ! input: field
      real :: pdfdx(0:kx+1,0:ky+1)  ! output: dfield/dx
      real :: zx                    ! 2.*dx
      integer :: i,j                ! loop indizes
!
!     df/dx=(f(i+1)-f(i-1))/2dx
!
      zx=2.*pdx
      do j=1,ky
       do i=1,kx
        pdfdx(i,j)=(pf(i+1,j)-pf(i-1,j))/zx
       enddo
      enddo
!
      return
      end
!
!-----------------------------------------------------------------------
!
      subroutine jacobi(p1,p2,pj,pdx,pdy,kx,ky)
      implicit none
!
!     subroutine jacobi computes the jacobi operator according to 
!     Arakawa (1966)
!
!     jacobi(1,2)=(j1+j2+j3)/3. after Arakawa 1966
!
      integer :: kx              ! x-dimension
      integer :: ky              ! y-dimension
      real :: p1(0:kx+1,0:ky+1)  ! input: field 1
      real :: p2(0:kx+1,0:ky+1)  ! input: field 2
      real :: pj(0:kx+1,0:ky+1)  ! output: jacobi(1,2)
      real :: pdx                ! x grid distance
      real :: pdy                ! y grid distance
      real :: zx                 ! 2.*dx
      real :: zy                 ! 2.*dy
      real :: zj1(kx,ky)         ! 1st jacobi
      real :: zj2(kx,ky)         ! 2nd jacobi
      real :: zj3(kx,ky)         ! 3rd jacobi 
      integer :: i,j             ! loop indizes
      integer :: im,jm,ip,jp     ! i-1,j-1,i+1,j+1
!
      zx=2.*pdx
      zy=2.*pdy
      do j=1,ky
       jp=j+1
       jm=j-1
       do i=1,kx
        ip=i+1
        im=i-1
        zj1(i,j)=(p1(ip,j)-p1(im,j))/zx*(p2(i,jp)-p2(i,jm))/zy          &
     &          -(p1(i,jp)-p1(i,jm))/zy*(p2(ip,j)-p2(im,j))/zx
        zj2(i,j)=(p2(i,jp)*(p1(ip,jp)-p1(im,jp))/zx                     &
     &           -p2(i,jm)*(p1(ip,jm)-p1(im,jm))/zx)/zy                 &
     &          -(p2(ip,j)*(p1(ip,jp)-p1(ip,jm))/zy                     &
     &           -p2(im,j)*(p1(im,jp)-p1(im,jm))/zy)/zx
        zj3(i,j)=(p1(ip,j)*(p2(ip,jp)-p2(ip,jm))/zy                     &
     &           -p1(im,j)*(p2(im,jp)-p2(im,jm))/zy)/zx                 &
     &          -(p1(i,jp)*(p2(ip,jp)-p2(im,jp))/zx                     &
     &           -p1(i,jm)*(p2(ip,jm)-p2(im,jm))/zx)/zy
        pj(i,j)=(zj1(i,j)+zj2(i,j)+zj3(i,j))/3.
       enddo
      enddo
!
      return
      end

!
!-----------------------------------------------------------------------
!
      subroutine laplace(pf,pdf,pdx,pdy,kx,ky)
      implicit none
!
!     subroutine laplace computes the laplacian from a field 
!
      integer :: kx                 ! x-dimension
      integer :: ky                 ! y-dimension
      real :: pdx                   ! x grid distance
      real :: pdy                   ! y grid distance
      real :: pf(0:kx+1,0:ky+1)     ! input: field
      real :: pdf(0:kx+1,0:ky+1)    ! output: Laplacian
      integer :: i,j                ! loop indizes
!
      do j=1,ky
       do i=1,kx
        pdf(i,j)=(pf(i-1,j)-2.*pf(i,j)+pf(i+1,j))/(pdx*pdx)             &
     &          +(pf(i,j-1)-2.*pf(i,j)+pf(i,j+1))/(pdy*pdy)
       enddo
      enddo
!
      return
      end

!
!-----------------------------------------------------------------------
!
      subroutine euler(pfm,pdfdt,pf,pdelt,kx,ky)
      implicit none
!
!     subroutine euler does an explicit Euler time step
!
      integer :: kx                   ! x dimension
      integer :: ky                   ! y dimension
      real    :: pdelt                ! time step [s]
      real    :: pfm(0:kx+1,0:ky+1)   ! input f(t)
      real    :: pdfdt(0:kx+1,0:ky+1) ! input tendency
      real    :: pf(0:kx+1,0:ky+1)    ! output f(t+1)
!
!     add tendency
!
      pf(1:kx,1:ky)=pfm(1:kx,1:ky)+pdelt*pdfdt(1:kx,1:ky)
!
      return
      end

!
!-----------------------------------------------------------------------
!
      subroutine leapfrog_update(pf,pfm,pdfdt,pdelt2,kx,ky)
      implicit none
!
!     subroutine leapfrog does a leapfrog time step 
!     with Robert Asselin filter
!
      real,parameter :: gamma=0.1  ! filter const.
      integer :: kx                ! x dimension
      integer :: ky                ! y dimension
      real :: pdelt2               ! 2.* timestep
      real :: pf(0:kx+1,0:ky+1)    ! input/output f(t)
      real :: pfm(0:kx+1,0:ky+1)   ! input/output f(t-1) (filtered)
      real :: pdfdt(0:kx+1,0:ky+1) ! input tendency
      real :: zsp(0:kx+1,0:ky+1)   ! new value at t+delt
!
!     a) add tendency to old (t-1) (filtered) value
!
      zsp(:,:)=pfm(:,:)+pdelt2*pdfdt(:,:)
!
!     b) compute the new (filtered) (t-1) value 
!
      pfm(:,:)=pf(:,:)+gamma*(pfm(:,:)-2.*pf(:,:)+zsp(:,:))
!
!     c) move the (t+1) value to the new (t) value
!
      pf(:,:)=zsp(:,:)
!
      return
      end subroutine leapfrog_update

