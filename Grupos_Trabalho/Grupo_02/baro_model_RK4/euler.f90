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
