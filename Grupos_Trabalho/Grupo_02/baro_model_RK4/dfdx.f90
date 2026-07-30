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
