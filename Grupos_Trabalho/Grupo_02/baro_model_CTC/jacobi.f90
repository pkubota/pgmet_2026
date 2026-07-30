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
