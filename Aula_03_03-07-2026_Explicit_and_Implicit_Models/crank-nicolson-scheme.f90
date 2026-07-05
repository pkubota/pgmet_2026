program crank_nicolson
implicit none
real, allocatable :: x(:),u(:),a(:),b(:),c(:),d(:)
real:: m,dx,dt,tmax
integer:: j,ni,ji

print*, 'enter the total number of time steps'
read*, ni
print*, 'enter the final time'
read*, tmax
dt=tmax/ni !the size of timestep
print*, 'this gives stepsize of time dt=',dt 
dx= 0.1 !delta x =0.1
ji= 20
m = dt/(2*dx**2)

allocate (x(0:ji),u(0:ji+1),a(ji),b(ji),c(ji),d(ji))
open(10,file='crank_nicolsan.m')

!initial condition
do j=1,ji
  x(j)= -1+j*dx
  u(j)=(1+x(j))*(1-x(j))**2 !x=-1+j*dx
end do
x(0)= -1
x(JI)= 1

!boundary condition
u(0)=0
u(ji+1)=0 

  do j = 1, ji ! a,b,c are the coefficients of c-n scheme and d is right part
    a(j) = -m
    b(j) = 1+2.0*m
    c(j) = -m
    d(j) = m*u(j+1)+(1-2.0*m)*u(j)+m*u(j-1) 
  end do
  call thomas(a,b,c,d)
  do j=1,ji-1
    u(j)=d(j)
  end do
 
!Print out the Approximate solution in matlab file 
   write(10,*)  'ApproximateSolution =[',x(0),u(0)
   do j =1, JI-1  
      write(10,*) x(j),u(j)
   end do
   write(10,*)x(JI),u(JI),']'
   write(10,*) "plot(ApproximateSolution(:,1),ApproximateSolution(:,2),'r')"
   write(10,*) "xlabel('x'),ylabel('temperature'),legend('Approximate C-N Scheme')"
   close(10) 
contains

subroutine thomas (a,b,c,d)

  real, intent(inout) :: a(:),b(:),c(:),d(:)
  integer j

  do j = 2,ji !combined decomposition and forward substitution
!	 a - sub-diagonal (means it is the diagonal below the main diagonal)
!	 b - the main diagonal
!	 c - sup-diagonal (means it is the diagonal above the main diagonal)
!	 d - right part
    
    a(j) = a(j)/b(j-1)
    b(j) = b(j)-a(j)*c(j-1)
    d(j) = d(j)-a(j)*d(j-1)
  end do 

  !back substitution
  d(ji) = d(ji)/b(ji)
  do j = ji-1,1,-1
    d(j) = (d(j)-c(j)*d(j+1))/b(j)
  end do

end subroutine thomas 

end program crank_nicolson 
