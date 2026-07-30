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
