      subroutine leapfrog(pf,pfm,pdfdt,pdelt2,kx,ky)
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
      end
