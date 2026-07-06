program advection_spectral

  implicit none

  ! Parameters
  integer, parameter :: N = 64       ! Number of grid points
  double precision, parameter :: L = 2.0 * 3.141592653589793d0 ! Length of the domain (2*pi for periodic)
  double precision, parameter :: c = 0.5d0   ! Advection velocity
  double precision, parameter :: dt = 0.01d0 ! Time step
  double precision, parameter :: t_final = 1.0d0 ! Final time

  ! Variables
  double precision, dimension(N) :: x          ! Spatial grid
  double precision, dimension(N) :: u          ! Solution at current time step
  double precision, dimension(N) :: u_complex(2*N) ! Complex array for FFT (real, imag, real, imag...)
  double precision, dimension(N) :: u_hat_real   ! Real part of Fourier transform of u
  double precision, dimension(N) :: u_hat_imag   ! Imaginary part of Fourier transform of u
  double precision, dimension(N) :: k_real       ! Real part of wavenumbers
  double precision, dimension(N) :: k_imag       ! Imaginary part of wavenumbers
  double precision, dimension(3*N) :: wsave      ! Workspace for FFTPACK

  integer :: i, n_steps,nn,lrec,irec
  double precision :: t
  real, dimension(N) :: uu  

  ! External FFT routines from FFTPACK
  external dfftf, dfftb, dffti

  inquire(iolength=lrec)uu
  open(1,file='espectral_method.bin',ACCESS='direct',FORM='unformatted',action='write',&
       status='unknown',recl=lrec)
  
  ! Initialize spatial grid
  do i = 1, N
    x(i) = (i - 1) * L / N
  end do

  ! Initialize wavenumbers
  ! For N even, k = [0, 1, ..., N/2-1, -N/2, ..., -1]
  ! For this spectral method, we need k_real and k_imag for complex multiplication
  do i = 1, N/2
    k_real(i) = 0.0d0
    k_imag(i) = (i - 1) * 2.0 * 3.141592653589793d0 / L
    k_real(N - i + 2) = 0.0d0
    k_imag(N - i + 2) = -i * 2.0 * 3.141592653589793d0 / L
  end do
  k_real(N/2 + 1) = 0.0d0
  k_imag(N/2 + 1) = N/2 * 2.0 * 3.141592653589793d0 / L ! Nyquist frequency for even N

  ! Initial condition (e.g., a sine wave)
  do i = 1, N
    u(i) = sin(x(i))
  end do
  uu=u
  ! Initialize FFTPACK workspace
  call dffti(N, wsave)

  ! Main time loop (using explicit Euler for simplicity)
  n_steps = int(t_final / dt)
  t = 0.0d0
  irec=1
  write(1,rec=irec) uu
  do nn = 1, n_steps
    irec=irec+1

    ! Prepare u for FFT (real part in u_complex(1::2), imag part in u_complex(2::2))
    do i = 1, N
      u_complex(2*i - 1) = u(i)
      u_complex(2*i    ) = 0.0d0
    end do

    ! Compute Fourier transform of u (u_complex -> u_complex)
    call dfftf(N, u_complex, wsave)

    ! Extract real and imaginary parts of u_hat
    do i = 1, N
      u_hat_real(i) = u_complex(2*i - 1)
      u_hat_imag(i) = u_complex(2*i    )
    end do

    ! Compute du/dt in Fourier space
    ! du/dt = -c * du/dx
    ! In Fourier space: d/dx -> i*k
    ! du_dt_hat =  * (i*k) * u_hat
    ! (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    !      (i*k_imag) * (u_hat_real + i*u_hat_imag)
    ! =    (i*k_imag*u_hat_real     -          k_imag*u_hat_imag)
    ! =     -  k_imag*u_hat_imag     + i * k_imag*u_hat_real
    do i = 1, N
      u_complex(2*i - 1) =  - k_imag(i) * u_hat_imag(i)
      u_complex(2*i    ) =    k_imag(i) * u_hat_real(i)
    end do

    ! Update u_hat using explicit Euler
    ! u_hat = u_hat - dt *c* du_dt_hat
    do i = 1, N
      u_complex(2*i - 1) = u_hat_real(i) - dt *c* u_complex(2*i - 1)
      u_complex(2*i    ) = u_hat_imag(i) - dt *c* u_complex(2*i    )
    end do

    ! Inverse Fourier transform to get u at next time step
    call dfftb(N, u_complex, wsave)

    ! Normalize and extract real part of u
    do i = 1, N
      u(i) = u_complex(2*i - 1) / N
    end do
    uu=u
    write(1,rec=irec) uu
    ! Atualiza a concentracao para o proximo passo de tempo
    print *,'interaction=',nn

    t = t + dt
  end do

  ! Output results to a file
  open(unit=10, file='output.dat')
  do i = 1, N
    write(10, '(F10.6, 1X, F10.6)') x(i), u(i)
  end do
  close(10)

end program advection_spectral


