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
