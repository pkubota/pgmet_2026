program advection_semi_lagrangian

  implicit none

  ! Parâmetros do problema
  integer, parameter :: nx = 100        ! Numero de pontos na grade espacial
  real, parameter    :: dx = 1.0         ! Espacamento da grade espacial
  real, parameter    :: dt = 0.1         ! Passo de tempo
  real, parameter    :: u = 1.0          ! Velocidade de advecção
  integer, parameter :: nsteps = 200     ! Numero de passos de tempo

  ! Escolha do tipo de interpolacao: 1 para linear, 2 para quadratica, 3 para cubica
  integer, parameter :: interpolation_type = 1 ! Alterado para testar cubica

  ! Variáveis
  real, dimension(nx) :: C_old, C_new   ! Concentracao nos tempos antigo e novo
  real, dimension(nx) :: x              ! Coordenadas espaciais
  integer             :: i, n           ! Iteradores
  integer             :: lrec,irec
  real                :: x_departure    ! Posicao de partida
  inquire(iolength=lrec)C_new
  open(1,file='semi_lagrangian.bin',ACCESS='direct',FORM='unformatted',action='write',&
       status='unknown',recl=lrec)
  ! Inicializacao da grade espacial
  do i = 1, nx
    x(i) = (i - 1) * dx
  end do

  ! Inicializacao da concentracao (exemplo: uma funcao gaussiana)
  do i = 1, nx
    C_old(i) = exp(-(x(i) - 50.0)**2 / (2.0 * 5.0**2))
  end do
  irec=1
  write(1,rec=irec) C_old
  ! Loop de tempo
  do n = 1, nsteps
    irec=irec+1
    ! Aplica o metodo Semi-Lagrangiano
    do i = 1, nx
      ! Calcula a posicao de partida
      x_departure = x(i) - u * dt

      ! Interpola a concentracao na posicao de partida
      select case (interpolation_type)
        case (1)
          C_new(i) = interpolate_linear(C_old, x, x_departure, nx)
        case (2)
          C_new(i) = interpolate_quadratic(C_old, x, x_departure, nx)
        case (3)
          C_new(i) = interpolate_cubic(C_old, x, x_departure, nx)
        case default
          print *, "Tipo de interpolacao invalido! Usando linear por padrao."
          C_new(i) = interpolate_linear(C_old, x, x_departure, nx)
      end select
    end do
    write(1,rec=irec) C_new
    ! Atualiza a concentracao para o proximo passo de tempo
    C_old = C_new
    print *,'interaction=',n
  end do

  ! Imprimir resultados finais
  !do i = 1, nx
  !  print *, x(i), C_old(i)
  !end do

contains

  ! Subrotina para interpolacao linear
  function interpolate_linear(C, x_grid, x_dep, nx_grid) result(val)
    implicit none
    real, dimension(:), intent(in) :: C
    real, dimension(:), intent(in) :: x_grid
    real, intent(in)              :: x_dep
    integer, intent(in)           :: nx_grid
    real                          :: val
    real    :: c_min, c_max

    integer :: i1, i2
    real    :: alpha

    if (x_dep < x_grid(1)) then
      val = C(1)
    else if (x_dep > x_grid(nx_grid)) then
      val = C(nx_grid)
    else
      i1 = int(floor(x_dep/(x_grid(2) - x_grid(1))))

      i2 = i1 + 1

      if (i1 < 1) i1 = 1
      if (i2 > nx_grid) i2 = nx_grid
      if (i2 < 1) i2 = 1

      alpha = (x_dep - x_grid(i1)) / (x_grid(i2) - x_grid(i1))
      val = C(i1) * (1.0 - alpha) + C(i2) * alpha
      ! Monotonicity check (simple bounds)
      c_min = min( C(i1), C(i2))
      c_max = max( C(i1), C(i2))
      val = max(c_min, min(val, c_max))
    end if
 end function interpolate_linear

  ! Subrotina para interpolacao quadratica (exemplo simples, pode ser melhorado)
  function interpolate_quadratic(C, x_grid, x_dep, nx_grid) result(val)
    implicit none
    real, dimension(:), intent(in) :: C
    real, dimension(:), intent(in) :: x_grid
    real, intent(in)              :: x_dep
    integer, intent(in)           :: nx_grid
    real                          :: val

    integer :: i0, i1, i2
    real    :: h, xi
    real    :: c_min, c_max

    if (x_dep < x_grid(1)) then
      val = C(1)
    else if (x_dep > x_grid(nx_grid)) then
      val = C(nx_grid)
    else
      i1 = int(floor(x_dep/(x_grid(2) - x_grid(1))))
      i0 = max(1, i1 - 1)
      i2 = min(nx_grid, i1 + 1)

      h = x_grid(2) - x_grid(1)
      xi = (x_dep - x_grid(i1)) / h

      ! Lagrange quadratic interpolation
      val = C(i0) * (xi * (xi - 1.0) / 2.0) + &
            C(i1) * (1.0 - xi**2) + &
            C(i2) * (xi * (xi + 1.0) / 2.0)

      ! Monotonicity check (simple bounds)
      c_min = min(C(i0), C(i1), C(i2))
      c_max = max(C(i0), C(i1), C(i2))
      val = max(c_min, min(val, c_max))
    end if
  end function interpolate_quadratic

  ! Subrotina para interpolacao cubica (exemplo simples, pode ser melhorado)
  function interpolate_cubic(C, x_grid, x_dep, nx_grid) result(val)
    implicit none
    real, dimension(:), intent(in) :: C
    real, dimension(:), intent(in) :: x_grid
    real, intent(in)              :: x_dep
    integer, intent(in)           :: nx_grid
    real                          :: val

    integer :: i0, i1, i2, i3
    real    :: h, xi
    real    :: c_min, c_max

    if (x_dep < x_grid(1)) then
      val = C(1)
    else if (x_dep > x_grid(nx_grid)) then
      val = C(nx_grid)
    else
      i1 = int(floor(x_dep/(x_grid(2) - x_grid(1))))
      i0 = max(1, i1 - 1)
      i2 = min(nx_grid, i1 + 1)
      i3 = min(nx_grid, i1 + 2)

      h  = x_grid(2) - x_grid(1)
      xi = (x_dep - x_grid(i1)) / h

      ! Lagrange cubic interpolation (using 4 points)
      val = C(i0) * ((-xi + 2.0) * (xi - 1.0) * (xi + 0.0) / 6.0) + &
            C(i1) * (( xi - 2.0) * (xi - 1.0) * (xi + 1.0) / 2.0) + &
            C(i2) * ((-xi + 1.0) * (xi + 1.0) * (xi + 2.0) / 2.0) + &
            C(i3) * (( xi + 0.0) * (xi + 1.0) * (xi + 2.0) / 6.0)

      ! Monotonicity check (simple bounds)
      c_min = min(C(i0), C(i1), C(i2), C(i3))
      c_max = max(C(i0), C(i1), C(i2), C(i3))
      val = max(c_min, min(val, c_max))
    end if
  end function interpolate_cubic

end program advection_semi_lagrangian


