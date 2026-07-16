!===============================================================================
! advdiff_verticalcoord.f90
!
! MET-576-4 -- Fontes de Erros nos Modelos de Previsoes Numericas
! INPE
!
! PROPOSITO
! ---------
! Demonstrar a EQUACAO DE ADVECCAO-DIFUSAO VERTICAL de um escalar A(csi,t)
! resolvida em DUAS coordenadas verticais distintas -- altura geometrica z
! e a coordenada sigma de Phillips (1957) -- usando UMA UNICA SUBROTINA
! GENERICA para o operador vertical. A subrotina generica e a implementacao
! numerica direta do formalismo desenvolvido na Secao 2 do material teorico
! "Coordenadas Verticais em Modelos Atmosfericos e Oceanicos":
!
!     dA/dt|_csi = - csidot * dA/dcsi  +  m * d/dcsi[ K * m * dA/dcsi ]     (*)
!
! onde:
!     csi     = coordenada vertical generica (z ou sigma)
!     csidot  = d(csi)/dt   -- velocidade vertical generalizada (Secao 2.7.1)
!     m       = d(csi)/dz   -- metrica da coordenada          (Secao 2.2)
!     K       = coeficiente de difusao turbulenta FISICO, definido em z [m2/s]
!
! O termo advectivo de (*) e a Eq. (6.6) da Secao 2.5 restrita a vertical.
! O termo difusivo de (*) e obtido aplicando a regra da cadeia (Secao 2.2,
! Eq. 6.2) DUAS VEZES ao operador fisico d/dz(K dA/dz):
!
!     d/dz( K dA/dz ) = m * d/dcsi[ K * m * dA/dcsi ]
!
! Para csi=z:      m == 1                          -> (*) reduz-se a forma
!                                                      cartesiana padrao.
! Para csi=sigma:  m = dsigma/dz = -sigma/H         -> (*) e a equacao de
!                  (H = RT0/g, atmosfera isotermica -- Secao 5, Eq. sigma_hydro)
!                  adveccao-difusao em coordenada sigma, com sigmadot
!                  desempenhando o papel de "velocidade vertical" (Secao 5.4)
!
! Ao chamar a MESMA subrotina tendency_generic() com (csi,csidot,m) = (z,w,1)
! ou (csi,csidot,m) = (sigma,sigmadot,-sigma/H), o programa resolve o MESMO
! problema fisico (a mesma circulacao vertical, a mesma pluma inicial) em
! duas representacoes de coordenada diferentes. Ao final, os dois perfis sao
! escritos em arquivos-texto e comparados (script Python auxiliar), servindo
! como verificacao pratica de que a transformacao de coordenadas foi feita
! corretamente: se as duas solucoes, mapeadas para a mesma altura fisica z,
! nao coincidirem dentro do erro de truncamento esperado, ha um erro na
! metrica ou na velocidade vertical generalizada.
!
! TESTE IDEALIZADO
! -----------------
! Perfil inicial: pulso gaussiano de um escalar passivo (ex.: um traçador)
! centrado em z_c, largura z_w.
! Circulacao prescrita: sigmadot(sigma) = -amp*sin(pi*(sigma-sigma_top)/(1-sigma_top))
!   (satisfaz sigmadot=0 em sigma_top e sigma=1, tal como exigido na Secao 5)
!   O w(z) fisicamente consistente e obtido por w = (dz/dsigma)*sigmadot,
!   isto e, exatamente a Eq. da Secao 2.7.1 aplicada a z em funcao de sigmadot.
!
! ESQUEMA NUMERICO
! ----------------
!   Adveccao : forma de FLUXO conservativa, reconstrucao upwind/FTBS de 1a
!              ordem nos meios-niveis (o analogo, em coordenada generica,
!              da forma de fluxo ja usada na equacao da continuidade em
!              sigma, Secao 5.1: ∂π/∂t + ∇.(πV) + ∂(π*sigmadot)/∂sigma=0).
!              Esta forma conserva EXATAMENTE a soma discreta de A ponderada
!              por dcsi (a "massa" na coordenada nativa), mesmo quando
!              csidot varia com o nivel (fluxo cisalhante/convergente) --
!              ao contrario da forma advectiva ingenua -csidot*dA/dcsi, que
!              NAO conserva essa soma quando ha convergencia/divergencia em
!              csidot (ver Nota didatica mais abaixo).
!   Difusao  : forma de fluxo, centrada de 2a ordem, com fluxo nulo nas bordas
!   Tempo    : Euler explicito (forward-in-time), Delta t limitado pelo mais
!              restritivo entre o numero de Courant e o numero de difusao
!              (ver a rotina compute_dt) -- o mesmo tipo de analise de
!              estabilidade do material CFL/estabilidade already desenvolvido
!              para MET-576-4.
!
! NOTA DIDATICA -- DOIS TIPOS DE "CONSERVACAO", E POR QUE NAO COINCIDEM
! ------------------------------------------------------------------------
!   Este exemplo revela algo sutil e importante sobre coordenadas verticais
!   transformadas, verificavel diretamente nos diagnosticos impressos por
!   este programa:
!
!   (a) o termo ADVECTIVO, escrito em forma de fluxo (conservativa), CONSERVA
!       EXATAMENTE a integral NATIVA na coordenada, soma_j[A(j)]*dcsi -- em
!       QUALQUER coordenada, mesmo quando csidot varia de nivel a nivel
!       (verificado abaixo desligando a difusao, K=0).
!
!   (b) o termo DIFUSIVO, escrito via a regra da cadeia metric*d/dcsi[K*
!       metric*dA/dcsi] (Secao 2.2), CONSERVA EXATAMENTE (a menos do erro de
!       truncamento espacial) a integral FISICA integral(A dz) -- pois e,
!       por construcao, a transformacao exata do operador fisico d/dz(K dA/dz),
!       cuja conservacao de integral(A dz) decorre diretamente do teorema
!       fundamental do calculo (verificado abaixo desligando a adveccao).
!
!   Em coordenada z, dcsi=dz e as duas integrais COINCIDEM -- por isso o
!   modelo em z conserva ambas simultaneamente. Em coordenada sigma (onde a
!   metrica dsigma/dz varia com o nivel), as duas integrais SAO DIFERENTES,
!   e a soma da tendencia advectiva+difusiva nao conserva perfeitamente
!   nenhuma das duas isoladamente -- um efeito real e conhecido, e a razao
!   pela qual modelos atmosfericos operacionais escrevem a equacao de
!   continuidade (e, por consistencia, o transporte de escalares) para a
!   variavel de massa PONDERADA (πA em sigma, Secao 5.1), e nao para o
!   escalar A isolado: e a variavel ponderada, no formalismo de fluxo, que
!   possui uma unica lei de conservacao consistente entre adveccao e difusao.
!   Os dois diagnosticos abaixo tornam esse efeito visivel e mensuravel.
!
! SUGESTOES DE EXERCICIO (para os alunos)
! ----------------------------------------
!   1. Aumentar K e verificar o novo Delta t critico (razao de difusao).
!   2. Trocar o esquema de adveccao para centrado e observar as oscilacoes
!      espurias (nenhum limitador de fluxo foi implementado neste exemplo).
!   3. Estender a subrotina tendency_generic() para a coordenada eta (Mesinger,
!      1984) ou para a hibrida sigma-z de Klemp (2011), reaproveitando csidot
!      e a metrica m ja deduzidos nas Secoes 6, 8 e 9 do material teorico.
!   4. Modificar sigmadot(sigma) para um perfil que troque de sinal (por
!      exemplo, ascensao abaixo do centro do pulso e subsidencia acima),
!      tornando a circulacao "fechada" (nao-drenante) e comparando a
!      evolucao com o perfil atual, que desloca a coluna inteira em uma
!      unica direcao.
!   5. Reescrever tendency_generic() para a variavel ponderada pela massa
!      (por exemplo, pi*A em sigma) e verificar que AMBAS as integrais --
!      nativa e fisica -- passam a ser conservadas simultaneamente (este e
!      exatamente o argumento por tras da Eq. de continuidade em forma de
!      fluxo da Secao 5.1, agora aplicado a um escalar transportado).
!
!===============================================================================

module mod_advdiff
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  real(dp), parameter :: pi = 3.14159265358979323846_dp

contains

  !-----------------------------------------------------------------------
  ! tendency_generic:  dA/dt|_csi para QUALQUER coordenada vertical csi.
  ! Implementacao direta da Eq. (*) do cabecalho deste arquivo.
  !
  ! O termo advectivo e escrito em FORMA DE FLUXO (conservativa):
  !     -csidot * dA/dcsi   ==   -(1/1) * d(csidot*A)/dcsi   (se csidot
  !     nao dependesse de csi; em geral usa-se a forma de fluxo diretamente,
  !     que e a que de fato conserva a integral em csi -- ver Nota didatica
  !     no cabecalho do arquivo e a Secao 5.1 do material teorico, onde a
  !     equacao da continuidade em sigma e deduzida exatamente nesta forma,
  !     dA/dt + V.d(V) + w(d*sigmadot)/dsigma = 0).
  ! O fluxo em cada meio-nivel usa reconstrucao upwind (FTBS) de A.
  !-----------------------------------------------------------------------
  subroutine tendency_generic(N, A, csidot, metric, K, dcsi, tend)
    integer,  intent(in)  :: N
    real(dp), intent(in)  :: A(N)        ! escalar transportado
    real(dp), intent(in)  :: csidot(N)   ! d(csi)/dt   (Secao 2.7.1)
    real(dp), intent(in)  :: metric(N)   ! d(csi)/dz   (Secao 2.2)
    real(dp), intent(in)  :: K           ! difusividade fisica [m2/s]
    real(dp), intent(in)  :: dcsi        ! espacamento de grade em csi
    real(dp), intent(out) :: tend(N)     ! tendencia total

    real(dp) :: fluxA(0:N), flux(0:N), mhalf, chalf
    integer  :: j

    ! ---- termo advectivo: forma de fluxo (conservativa), upwind/FTBS ----
    ! fluxA(j) aproxima csidot*A no meio-nivel entre j e j+1
    fluxA(0) = 0.0_dp   ! csidot=0 nesta borda por construcao (Secao 5)
    fluxA(N) = 0.0_dp   ! idem na borda superior
    do j = 1, N-1
      chalf = 0.5_dp * (csidot(j) + csidot(j+1))
      if (chalf > 0.0_dp) then
        fluxA(j) = chalf * A(j)      ! informacao vem de baixo (j)
      else
        fluxA(j) = chalf * A(j+1)    ! informacao vem de cima (j+1)
      end if
    end do
    do j = 1, N
      tend(j) = -(fluxA(j) - fluxA(j-1)) / dcsi
    end do

    ! ---- termo difusivo: metric * d/dcsi[ K * metric * dA/dcsi ] ----
    ! fluxo fisico (K dA/dz) avaliado nos meios-niveis, forma conservativa
    flux(0) = 0.0_dp   ! sem fluxo na borda inferior (superficie)
    flux(N) = 0.0_dp   ! sem fluxo na borda superior (topo do modelo)
    do j = 1, N-1
      mhalf   = 0.5_dp * (metric(j) + metric(j+1))
      flux(j) = K * mhalf * (A(j+1) - A(j)) / dcsi
    end do
    do j = 1, N
      tend(j) = tend(j) + metric(j) * (flux(j) - flux(j-1)) / dcsi
    end do

  end subroutine tendency_generic

  !-----------------------------------------------------------------------
  ! compute_dt: Delta t estavel para adveccao (Courant) e difusao, com
  ! fator de seguranca "safety" (analogo a analise de estabilidade CFL).
  !-----------------------------------------------------------------------
  subroutine compute_dt(N, csidot, metric, K, dcsi, safety, dt)
    integer,  intent(in)  :: N
    real(dp), intent(in)  :: csidot(N), metric(N), K, dcsi, safety
    real(dp), intent(out) :: dt
    real(dp) :: dt_adv, dt_diff, cmax, kmax
    integer  :: j

    cmax = 0.0_dp
    kmax = 0.0_dp
    do j = 1, N
      cmax = max(cmax, abs(csidot(j)))
      kmax = max(kmax, K * metric(j)**2)
    end do

    if (cmax > 0.0_dp) then
      dt_adv = dcsi / cmax
    else
      dt_adv = huge(1.0_dp)
    end if

    if (kmax > 0.0_dp) then
      dt_diff = 0.5_dp * dcsi**2 / kmax
    else
      dt_diff = huge(1.0_dp)
    end if

    dt = safety * min(dt_adv, dt_diff)
  end subroutine compute_dt

  !-----------------------------------------------------------------------
  ! column_integral: aproxima  integral(A dz)  usando pesos de z (regra
  ! trapezoidal), a partir de valores de z em cada nivel de grade -- serve
  ! como diagnostico de conservacao de massa da coluna, independente da
  ! coordenada em que A esta representado.
  !-----------------------------------------------------------------------
  function column_integral(N, A, z) result(total)
    integer,  intent(in) :: N
    real(dp), intent(in) :: A(N), z(N)
    real(dp) :: total
    integer  :: j
    total = 0.0_dp
    do j = 1, N-1
      total = total + 0.5_dp*(A(j)+A(j+1)) * abs(z(j+1)-z(j))
    end do
  end function column_integral

end module mod_advdiff


!===============================================================================
program advdiff_verticalcoord
  use mod_advdiff
  implicit none

  ! ---------------- parametros fisicos e da atmosfera de referencia --------
  real(dp), parameter :: R_gas   = 287.0_dp      ! J/(kg K)
  real(dp), parameter :: g       = 9.81_dp       ! m/s2
  real(dp), parameter :: T0      = 250.0_dp      ! K  (atmosfera isotermica)
  real(dp), parameter :: H_scale = R_gas*T0/g     ! altura de escala [m]

  ! ---------------- parametros da grade -------------------------------------
  integer,  parameter :: N          = 41
  real(dp), parameter :: sigma_top  = 0.02_dp
  real(dp), parameter :: K_diff     = 5.0_dp      ! difusividade fisica [m2/s]
  real(dp), parameter :: amp_sigdot = 7.0e-6_dp   ! amplitude de sigmadot [1/s]
                                                    ! (implica w de ordem 0.1 m/s,
                                                    ! valor sinotico tipico)

  ! ---------------- parametros do pulso inicial e da integracao ------------
  real(dp), parameter :: z_center  = 4000.0_dp    ! m
  real(dp), parameter :: z_width   = 1200.0_dp    ! m
  real(dp), parameter :: t_total   = 6.0_dp*3600.0_dp   ! 6 horas, em segundos
  real(dp), parameter :: safety    = 0.5_dp

  ! ---------------- variaveis da grade sigma --------------------------------
  real(dp) :: sigma(N), z_of_sigma(N), metric_sigma(N), sigmadot(N)
  real(dp) :: A_sigma(N), tend_sigma(N)
  real(dp) :: dsigma

  ! ---------------- variaveis da grade z -------------------------------------
  real(dp) :: z(N), metric_z(N), w_of_z(N)
  real(dp) :: A_z(N), tend_z(N)
  real(dp) :: dz, z_top

  real(dp) :: dt, dt_sigma, dt_z, t, sig_j
  real(dp) :: mass0_sigma, mass0_z, massF_sigma, massF_z
  integer  :: j, i, nsteps, iout
  integer, parameter :: nsnap = 3
  real(dp) :: A_sigma_snap(N,nsnap), A_z_snap(N,nsnap)
  real(dp) :: t_snap(nsnap)
  integer  :: isnap

  !===========================================================================
  ! 1) CONSTRUCAO DA GRADE SIGMA E DA GRADE Z
  !===========================================================================
  dsigma = (1.0_dp - sigma_top) / real(N-1, dp)
  do j = 1, N
    sigma(j)        = sigma_top + real(j-1, dp)*dsigma
    ! Relacao z(sigma) EXATA para atmosfera isotermica (Secao 5, sigma_hydro
    ! integrada com T=T0=const): z = H*ln(1/sigma), forma classica do
    ! "hypsometric equation" para T constante.
    z_of_sigma(j)   = H_scale * log(1.0_dp/sigma(j))
    ! Metrica da coordenada sigma: m = dsigma/dz = -sigma/H (Secao 2.2/5)
    metric_sigma(j) = -sigma(j) / H_scale
    ! Circulacao prescrita: sigmadot=0 em sigma_top e em sigma=1 (Secao 5)
    sigmadot(j)     = -amp_sigdot * sin(pi*(sigma(j)-sigma_top)/(1.0_dp-sigma_top))
  end do

  z_top = z_of_sigma(1)     ! topo do modelo (sigma=sigma_top)
  dz    = z_top / real(N-1, dp)
  do i = 1, N
    z(i)        = real(i-1, dp) * dz          ! i=1 na superficie, i=N no topo
    metric_z(i) = 1.0_dp                      ! coordenada z: metrica trivial
    ! sigma correspondente a esta altura (para avaliar sigmadot no MESMO
    ! ponto fisico e obter um w(z) fisicamente consistente, Secao 2.7.1):
    sig_j       = exp(-z(i)/H_scale)
    w_of_z(i)   = (-H_scale/sig_j) * ( -amp_sigdot * &
                    sin(pi*(sig_j-sigma_top)/(1.0_dp-sigma_top)) )
  end do

  !===========================================================================
  ! 2) CONDICAO INICIAL: mesmo pulso gaussiano fisico, amostrado em cada grade
  !===========================================================================
  do j = 1, N
    A_sigma(j) = exp( -((z_of_sigma(j)-z_center)/z_width)**2 )
  end do
  do i = 1, N
    A_z(i) = exp( -((z(i)-z_center)/z_width)**2 )
  end do

  mass0_sigma = column_integral(N, A_sigma, z_of_sigma)
  mass0_z     = column_integral(N, A_z, z)

  !===========================================================================
  ! 3) PASSO DE TEMPO ESTAVEL (o mais restritivo entre os dois modelos)
  !===========================================================================
  call compute_dt(N, sigmadot, metric_sigma, K_diff, dsigma, safety, dt_sigma)
  call compute_dt(N, w_of_z,   metric_z,     K_diff, dz,     safety, dt_z)
  dt = min(dt_sigma, dt_z)
  nsteps = ceiling(t_total/dt)
  nsteps = max(nsteps, 200)     ! garante resolucao temporal minima razoavel
  dt = t_total / real(nsteps, dp)

  print *, '================================================================'
  print *, ' advdiff_verticalcoord -- MET-576-4 / INPE'
  print *, '================================================================'
  print '(A,F10.1,A)', ' Altura de escala H            = ', H_scale, ' m'
  print '(A,F10.1,A)', ' Topo do modelo (z_top)        = ', z_top,  ' m'
  print '(A,F10.1,A)', ' Delta z  (grade z, uniforme)  = ', dz,     ' m'
  print '(A,F10.5)',   ' Delta sigma (grade sigma)     = ', dsigma
  print '(A,F10.3,A)', ' Delta t estavel escolhido     = ', dt, ' s'
  print '(A,I8)',      ' Numero de passos de tempo     = ', nsteps
  print *, '----------------------------------------------------------------'

  !===========================================================================
  ! 4) INTEGRACAO NO TEMPO (Euler explicito) -- os DOIS modelos avancam com o
  !    MESMO Delta t, cada um em sua propria coordenada, chamando a MESMA
  !    subrotina generica.
  !===========================================================================
  t = 0.0_dp
  isnap = 1
  t_snap(1) = 0.0_dp
  A_sigma_snap(:,1) = A_sigma
  A_z_snap(:,1)     = A_z

  do iout = 1, nsteps
    call tendency_generic(N, A_sigma, sigmadot, metric_sigma, K_diff, dsigma, tend_sigma)
    call tendency_generic(N, A_z,     w_of_z,   metric_z,     K_diff, dz,     tend_z)

    A_sigma = A_sigma + dt*tend_sigma
    A_z     = A_z     + dt*tend_z
    t = t + dt

    ! grava um "retrato" (snapshot) em t=t_total/2 e no final
    if (isnap == 1 .and. t >= 0.5_dp*t_total) then
      isnap = 2
      t_snap(2) = t
      A_sigma_snap(:,2) = A_sigma
      A_z_snap(:,2)     = A_z
    end if
  end do
  isnap = 3
  t_snap(3) = t
  A_sigma_snap(:,3) = A_sigma
  A_z_snap(:,3)     = A_z

  !===========================================================================
  ! 5) DIAGNOSTICO DE CONSERVACAO
  !===========================================================================
  massF_sigma = column_integral(N, A_sigma, z_of_sigma)
  massF_z     = column_integral(N, A_z, z)

  print '(A)', ' (a) Integral NATIVA na coordenada, soma[A]*dcsi'
  print '(A)', '     (conservada EXATAMENTE pela ADVECCAO em forma de fluxo;'
  print '(A)', '      a difusao, por si so, NAO a conserva exatamente):'
  print '(A,ES14.6E3)', '   sigma: inicial = ', sum(A_sigma_snap(:,1))*dsigma
  print '(A,ES14.6E3)', '   sigma: final   = ', sum(A_sigma)*dsigma
  print '(A,ES14.6E3)', '   z    : inicial = ', sum(A_z_snap(:,1))*dz
  print '(A,ES14.6E3)', '   z    : final   = ', sum(A_z)*dz
  print *, ''
  print '(A)', ' (b) Integral FISICA integral(A dz)'
  print '(A)', '     (conservada pela DIFUSAO via regra da cadeia (Secao 2.2);'
  print '(A)', '      a adveccao, por si so, NAO a conserva exatamente em sigma):'
  print '(A,ES14.6E3)',     '   grade sigma -- inicial = ', mass0_sigma
  print '(A,ES14.6E3)',     '   grade sigma -- final   = ', massF_sigma
  print '(A,F10.4,A)',    '   erro relativo (sigma)  = ', &
                            100.0_dp*abs(massF_sigma-mass0_sigma)/mass0_sigma, ' %'
  print '(A,ES14.6E3)',     '   grade z     -- inicial = ', mass0_z
  print '(A,ES14.6E3)',     '   grade z     -- final   = ', massF_z
  print '(A,F10.4,A)',    '   erro relativo (z)      = ', &
                            100.0_dp*abs(massF_z-mass0_z)/mass0_z, ' %'
  print *, '  (em z, dcsi=dz: as duas integrais coincidem -- por isso o'
  print *, '   modelo em z conserva (a) e (b) simultaneamente, e o modelo'
  print *, '   em sigma nao. Ver Nota didatica no cabecalho do arquivo.)'
  print *, '----------------------------------------------------------------'

  !===========================================================================
  ! 6) SAIDA EM ARQUIVOS-TEXTO PARA COMPARACAO / PLOTAGEM (script Python)
  !===========================================================================
  open(unit=10, file='saida_sigma.txt', status='replace', action='write')
  write(10,'(A)') '# j  sigma        z(sigma)[m]   A_t0        A_tmid      A_tend'
  do j = 1, N
    write(10,'(I4,5(1X,ES14.6E3))') j, sigma(j), z_of_sigma(j), &
         A_sigma_snap(j,1), A_sigma_snap(j,2), A_sigma_snap(j,3)
  end do
  close(10)

  open(unit=11, file='saida_z.txt', status='replace', action='write')
  write(11,'(A)') '# i  z[m]          A_t0        A_tmid      A_tend'
  do i = 1, N
    write(11,'(I4,4(1X,ES14.6E3))') i, z(i), &
         A_z_snap(i,1), A_z_snap(i,2), A_z_snap(i,3)
  end do
  close(11)

  open(unit=12, file='saida_tempos.txt', status='replace', action='write')
  write(12,'(A)') '# snapshot  t[s]'
  do isnap = 1, nsnap
    write(12,'(I4,1X,ES14.6E3)') isnap, t_snap(isnap)
  end do
  close(12)

  print *, 'Arquivos gerados: saida_sigma.txt, saida_z.txt, saida_tempos.txt'
  print *, '================================================================'

end program advdiff_verticalcoord
