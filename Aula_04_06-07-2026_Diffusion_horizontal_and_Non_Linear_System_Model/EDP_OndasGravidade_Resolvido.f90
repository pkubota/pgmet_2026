! EDP_OndasGravidade_Corrigido.f90
! Versao corrigida de EDP_OndasGravidade_Resolvido.f90
! Correcoes marcadas com "! CORRECAO (bug N):" no ponto exato da mudanca.
MODULE Class_Fields
  IMPLICIT NONE
  PRIVATE
  INTEGER, PUBLIC      , PARAMETER  :: r8=8
  INTEGER, PUBLIC      , PARAMETER  :: r4=4
 REAL (KIND=r8) :: xMax=1
 REAL (KIND=r8) :: xMin=0
 REAL (KIND=r8), PUBLIC :: DeltaX=500.0
 REAL (KIND=r8), PUBLIC :: DeltaT=2.0
 REAL (KIND=r8), PUBLIC :: C     =2.0
 REAL (KIND=r8), PUBLIC :: Kdif  =2.0e-5
 REAL (KIND=r8), PUBLIC :: Kdif_RK4 =2.0e-5
 REAL (KIND=r8), PUBLIC :: Grav  = 9.8
 REAL (KIND=r8), PUBLIC :: H0    = 1

 INTEGER, PUBLIC :: Idim
 REAL (KIND=r8), PUBLIC :: xb0=100.0
 REAL (KIND=r8), PUBLIC :: xf0=400.0
 REAL (KIND=r8), PUBLIC :: tb0 =0
 REAL (KIND=r8), PUBLIC :: tf0 =0
 REAL (KIND=r8), PUBLIC :: xxb
 REAL (KIND=r8), PUBLIC :: yyf
 REAL (KIND=r8), PUBLIC :: Area
 REAL (KIND=r8), PUBLIC :: alfa
 REAL (KIND=r8), PUBLIC :: beta

  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: xa(:)
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: ua(:)
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: Deslc(:)

  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: h (:)
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: hm(:)
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: hp(:)

  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: u (:)
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: um(:)
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: up(:)

  PUBLIC :: Init_Class_Fields

CONTAINS
!-----------------------------------------------------------------------------------------
  SUBROUTINE Init_Class_Fields(alfa_in,beta_in,Kdif_in,Kdif_RK4_in)
    IMPLICIT NONE
      REAL (KIND=r8), INTENT(in) :: alfa_in
      REAL (KIND=r8), INTENT(in) :: beta_in
      REAL (KIND=r8), INTENT(in) :: Kdif_in
      REAL (KIND=r8), INTENT(in) :: Kdif_RK4_in
      INTEGER :: i
      REAL (KIND=r8):: t
      ! CORRECAO (bug 1): xb(1), xf(1) (MINLOC) e diff(:) so eram usados para
      ! calcular Area a partir de h(xf)-h(xb-1), que gerava 0/0 = NaN (ver abaixo).
      ! Como Area passou a ser definida diretamente como a amplitude fisica do
      ! pulso, essas variaveis auxiliares nao sao mais necessarias.
      PRINT*,'DeltaX=',DeltaX,'DeltaT=',DeltaT,'CFL=',C*DeltaT/DeltaX
      Idim=360
      alfa=alfa_in
      beta=beta_in
      Kdif=Kdif_in
      Kdif_RK4=Kdif_RK4_in
      ! CORRECAO (bug 5): a velocidade de propagacao do pulso "analitico" de
      ! referencia (usado em AnaliticFunction) deve ser igual a velocidade de
      ! fase real das ondas de gravidade do sistema acoplado (u,h), que e
      ! sqrt(Grav*H0), e nao um valor arbitrario (C=2.0 no original). Sem essa
      ! correcao, a comparacao com a solucao analitica fica sem sentido fisico.
      C = SQRT(Grav*H0)
      !Idim=  (xMax-Xmin)/DeltaX
      if (.not. allocated(u))  ALLOCATE (u(Idim))
      u=0.0
      if (.not. allocated(um))  ALLOCATE (um(Idim))
      um=0.0
      if (.not. allocated(up))  ALLOCATE (up(Idim))
      up=0.0
      if (.not. allocated(h))  ALLOCATE (h(Idim))
      h=0.0
      if (.not. allocated(hm))  ALLOCATE (hm(Idim))
      hm=0.0
      if (.not. allocated(hp))  ALLOCATE (hp(Idim))
      hp=0.0

      if (.not. allocated(Deslc))   ALLOCATE (Deslc(Idim))
      Deslc=0.0
      if (.not. allocated(ua)) ALLOCATE (ua(Idim))
      ua=0.0
      if (.not. allocated(xa)) ALLOCATE (xa(Idim))

      DO i=1,Idim
         xa(i)=(i-1)*DeltaX
      END DO
      xb0=xa(Idim)/4.0
      xf0=xa(Idim)/2.0
      tb0 =0
      tf0 =0
      t=0
      xxb= xb0 + C*(t-tb0)
      yyf= xf0 + C*(t-tf0)
      DO i=1,Idim
         IF(xa(i) >xxb .and. xa(i) <yyf)THEN
            h(i)=1.0
         ELSE
            h(i)=0.0
         END IF
      END DO

      ! CORRECAO (bug 1): a formula original,
      !   Area = ( h(xf(1)) - h(xb(1)-1) ) * (xa(xf(1))-xa(xb(1))) / (xf(1)-xb(1)+1)
      ! calculava xb(1)/xf(1) via MINLOC sobre os pontos de fronteira do pulso.
      ! Como o pulso e construido com desigualdade estrita (xa(i) > xxb .and.
      ! xa(i) < yyf), h vale 0 exatamente nesses pontos de fronteira, entao
      ! Area dava sempre 0. Mais abaixo, h(i) = -0.9*Area/Area calculava 0/0
      ! (NaN), contaminando TODO o campo h antes do loop temporal comecar.
      ! Alem disso, h(xb(1)-1) podia acessar indice 0 (fora dos limites do
      ! array) quando xb(1)=1. Corrigido definindo Area diretamente como a
      ! amplitude fisica do pulso (usada tambem em AnaliticFunction).
      Area = -0.9
      DO i=1,Idim
         IF(.FALSE.)THEN
            h(i) = COS((xa(i) - xa(Idim/2))/xa(Idim))
         ELSE
            IF(h(i) ==1.0)THEN
               h(i) = Area
            END IF
         END IF
      END DO
      hm=h
      hp=h
  END SUBROUTINE Init_Class_Fields
!------------------------------------------------------------------------------------------
END MODULE Class_Fields


 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODULE Class_WritetoGrads
 USE Class_Fields, Only: Idim,xa
 IMPLICIT NONE
 PRIVATE
 INTEGER, PUBLIC      , PARAMETER  :: r8=8
 INTEGER, PUBLIC      , PARAMETER  :: r4=4
 INTEGER                    , PARAMETER :: UnitData=1
 INTEGER                    , PARAMETER :: UnitCtl=2
 CHARACTER (LEN=400)                   :: FileName
 LOGICAL                                            :: CtrlWriteDataFile
 PUBLIC :: SchemeWriteCtl
 PUBLIC :: SchemeWriteData
 PUBLIC :: InitClass_WritetoGrads
CONTAINS
 SUBROUTINE InitClass_WritetoGrads()
    IMPLICIT NONE
   FileName=''
   FileName='GravityWaveConceitual1D'
   CtrlWriteDataFile=.TRUE.
 END SUBROUTINE InitClass_WritetoGrads

 FUNCTION SchemeWriteData(vars,irec)  RESULT (ok)
    IMPLICIT NONE
    REAL (KIND=r8), INTENT (INOUT) :: vars(Idim)
    INTEGER       , INTENT (INOUT) :: irec
    INTEGER        :: ok
    INTEGER        :: lrec
    REAL (KIND=r4) :: Yout(Idim)
    IF(CtrlWriteDataFile)INQUIRE (IOLENGTH=lrec) Yout
    IF(CtrlWriteDataFile)OPEN(UnitData,FILE=TRIM(FileName)//'.bin',&
    FORM='UNFORMATTED', ACCESS='DIRECT', STATUS='UNKNOWN', &
    ACTION='WRITE',RECL=lrec)
    ok=1
    CtrlWriteDataFile=.FALSE.
    Yout=REAL(vars(1:Idim),KIND=r4)
    irec=irec+1
    WRITE(UnitData,rec=irec)Yout
     ok=0
 END FUNCTION SchemeWriteData

 FUNCTION SchemeWriteCtl(nrec)  RESULT (ok)
    IMPLICIT NONE
    INTEGER, INTENT (IN) :: nrec
    INTEGER             :: ok,i
    ok=1
   OPEN(UnitCtl,FILE=TRIM(FileName)//'.ctl',FORM='FORMATTED', &
   ACCESS='SEQUENTIAL',STATUS='UNKNOWN',ACTION='WRITE')
    WRITE (UnitCtl,'(A6,A           )')'dset ^',TRIM(FileName)//'.bin'
    WRITE (UnitCtl,'(A                 )')'title  EDO'
    WRITE (UnitCtl,'(A                 )')'undef  -9999.9'
    WRITE (UnitCtl,'(A6,I8,A8   )')'xdef  ',Idim,' levels '
    WRITE (UnitCtl,'(10F16.10   )')(xa(i)/xa(Idim),i=1,Idim)
    WRITE (UnitCtl,'(A                  )')'ydef  1 linear  -1.27 1'
    WRITE (UnitCtl,'(A6,I6,A25   )')'tdef  ',nrec,' linear  00z01jan0001 1hr'
    WRITE (UnitCtl,'(A20             )')'zdef  1 levels 1000 '
    WRITE (UnitCtl,'(A           )')'vars 2'
    WRITE (UnitCtl,'(A           )')'uc 0 99 resultado da edol yc'
    WRITE (UnitCtl,'(A           )')'hc 0 99 resultado da edol hc'
    WRITE (UnitCtl,'(A           )')'endvars'
    CLOSE (UnitCtl,STATUS='KEEP')
    CLOSE (UnitData,STATUS='KEEP')
    ok=0
 END FUNCTION SchemeWriteCtl
END MODULE Class_WritetoGrads




 MODULE ModAdvection
  USE Class_Fields, Only: DeltaT,DeltaX,Idim,r8,xa,tf0,tb0,yyf,&
                          xxb,xb0,xf0,Area,C,Deslc,alfa,beta,Kdif,Kdif_RK4,Grav,H0
   IMPLICIT NONE
   PRIVATE

  PUBLIC :: AnaliticFunction,Upstream,CentredTimeSpace,RungeKutta4,ComputationalDiffusion

CONTAINS

!
 FUNCTION ComputationalDiffusion(Q,dt,dx) RESULT(diffterm)
   !"""
   ! Difusao computacional de 2a e 4a ordem, conforme derivado na nota de
   ! aula (Prof. Kubota, Eq. 12.3.29-12.3.31), reaproveitavel por qualquer
   ! esquema (Upstream, CentredTimeSpace, RK4...):
   !
   !   mu  = C*dt/dx                (numero de Courant, 12.3.17a)
   !   Kc  = (C*dx/2)*(1-mu)        (difusao numerica de 2a ordem)
   !   eps = Kc*dx**2/12            (difusao de 4a ordem, tipo Smagorinsky;
   !                                  CORRECAO do erro de reducao de fracao
   !                                  na nota, que dava dx**2/24 em vez de
   !                                  dx**2/12 -- ver 2/4! = 2/24 = 1/12)
   !
   !   diffterm = Kc*d2Q/dx2 - eps*d4Q/dx4
   !
   ! O sinal de Kc e '+' (dissipativo p/ 0<mu<=1) e o de eps e '-' (para que
   ! o termo de 4a ordem tambem seja dissipativo em vez de anti-difusivo nas
   ! menores escalas resolvidas -- ver analise de Fourier/von Neumann:
   ! (ik)^2=-k^2 mas (ik)^4=+k^4, entao o termo de 4a ordem so amortece se
   ! entrar com sinal negativo, convencao padrao de hiperdifusao em modelos
   ! atmosfericos/oceanicos, -K4*Laplaciano^2).
   !
   ! Usa C = sqrt(Grav*H0) (velocidade de fase das ondas de gravidade do
   ! sistema acoplado u,h -- ver correcao do bug 5 em Init_Class_Fields).
   !
   ! Retorna a TENDENCIA (nao multiplicada por dt); quem chama deve somar
   !   Q_novo = Q_antigo + dt*ComputationalDiffusion(Q,dt,dx)
   !"""
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   REAL(KIND=r8)             :: diffterm(Idim)
   REAL(KIND=r8)             :: mu, Kc, eps
   INTEGER                   :: xb3,xb2,xb,xc,xf,xf2,xf3
   INTEGER                   :: i

   mu  = C*dt/dx
   Kc  = (C*dx/2.0_r8)*(1.0_r8-mu)
   eps = Kc*dx*dx/12.0_r8

   DO i=1,Idim
      CALL index2(i,xb3,xb2,xb,xc,xf,xf2,xf3)
      diffterm(i) = Kc*( Q(xf) - 2.0_r8*Q(xc) + Q(xb) )/(dx*dx) &
                   - eps*( Q(xf2) - 4.0_r8*Q(xf) + 6.0_r8*Q(xc) &
                           - 4.0_r8*Q(xb) + Q(xb2) )/(dx**4)
   END DO
 END FUNCTION ComputationalDiffusion

!   !************************************************************************************

  FUNCTION AnaliticFunction(termX,ua,it)  RESULT (ok)

      REAL(KIND=r8), INTENT(INOUT) :: termX(Idim)
      REAL(KIND=r8), INTENT(IN   ) :: ua(Idim)
      INTEGER, INTENT(IN   ) :: it
      INTEGER          :: i2,xb,xc,xf,i
      INTEGER         :: ok
      REAL(KIND=r8)    :: t
      t=(it)*DeltaT

      yyf= xf0 + C*(t-tf0)
      IF(yyf >= xa(Idim))THEN
         xf0=0.0
         yyf=xf0
         tf0=t
      END IF

      xxb= xb0 + C*(t-tb0)
      IF(xxb >= xa(Idim))THEN
         xb0=0.0
         xxb=xb0
         tb0=t
      END IF
      IF(xf0 <= xb0 .and. yyf <= xxb) THEN
         DO i=1,Idim
            IF(xa(i) > xxb )THEN
               termX(i)=Area
            ELSE IF( xa(i) < yyf )THEN
               termX(i)=Area
            ELSE
               termX(i)=0.0
            END IF
         END DO
      ELSE
         DO i=1,Idim
            IF(xa(i) > xxb .and. xa(i) < yyf)THEN
               termX(i)=Area
            ELSE
               termX(i)=0.0
            END IF
         END DO
      END IF
    ok=0
   END FUNCTION AnaliticFunction



 FUNCTION CentredTimeSpace(hm,hc,hp,Um,Uc,Up,t,dt,dx) RESULT (ok)
   !"""
   !central in time, Forward in space advection scheme (1st order)
   !"""
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(INOUT) :: hm(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: hc(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: hp(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Um(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Uc(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Up(Idim)

   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   INTEGER             :: xb,xc,xf
   REAL(KIND=r8)       :: Hnp1 (SIZE(hc))
   REAL(KIND=r8)       :: Unp1 (SIZE(hc))
   INTEGER         :: ok

   INTEGER             :: N0
   INTEGER             :: N1
   INTEGER             :: i
   N0=LBOUND(hc,DIM=1)
   N1=UBOUND(hc,DIM=1)
   DO i=N0,N1
      CALL index(i,xb,xc,xf)
      Unp1(i) = Um(xc) -(dt/dx)*Grav*( hc(xf) - hc(xb) )
   END DO
   ! Difusao computacional de 2a e 4a ordem (opcional, ativada por Kdif>0),
   ! ver funcao ComputationalDiffusion.
   IF( Kdif>0 ) Unp1 = Unp1 + dt*ComputationalDiffusion(Uc,dt,dx)

   DO i=N0,N1
      CALL index(i,xb,xc,xf)
      Hnp1(i) = hm(xc) - (dt/dx)*H0*( Uc(xf) - Uc(xb) )
   END DO
   IF( Kdif>0 ) Hnp1 = Hnp1 + dt*ComputationalDiffusion(hc,dt,dx)
   hp= Hnp1
   Up= Unp1
   ok=0
 END FUNCTION CentredTimeSpace
!


 FUNCTION Upstream(hm,hc,hp,Um,Uc,Up,t,dt,dx)  result(ok)
   !"""
   !Forward in time, Forward in space advection scheme (1st order)
   !"""
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(INOUT) :: hm(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: hc(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: hp(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Um(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Uc(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Up(Idim)
   INTEGER         :: ok
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   INTEGER             :: xb,xc,xf
   REAL(KIND=r8)       :: Hnp1 (SIZE(hc))
   REAL(KIND=r8)       :: Unp1 (SIZE(hc))
   INTEGER             :: N0
   INTEGER             :: N1
   INTEGER             :: i
   N0=LBOUND(hc,DIM=1)
   N1=UBOUND(hc,DIM=1)
    ! Difusao computacional de 2a e 4a ordem (funcao ComputationalDiffusion,
    ! ver comentario detalhado nela) somada como termo explicito quando
    ! Kdif>0.
    IF( Kdif>0 )THEN
        DO i=N0,N1
           CALL index(i,xb,xc,xf)
           Unp1(i) = Uc(xc) - (dt/dx)*Grav*( hc(xc) - hc(xb) )
        END DO
        Unp1 = Unp1 + dt*ComputationalDiffusion(Uc,dt,dx)

        DO i=N0,N1
           CALL index(i,xb,xc,xf)
           Hnp1(i) = hc(xc) - (dt/dx)*H0  *( Uc(xc) - Uc(xb) )
        END DO
        Hnp1 = Hnp1 + dt*ComputationalDiffusion(hc,dt,dx)

    ELSE
        DO i=N0,N1
           CALL index(i,xb,xc,xf)
           Unp1(i) = Uc(xc) - (dt/dx)*Grav*( hc(xc) - hc(xb) )
        END DO
        DO i=N0,N1
           CALL index(i,xb,xc,xf)
           Hnp1(i) = hc(xc) - (dt/dx)*H0  *( Uc(xc) - Uc(xb) )
        END DO
    END IF
   hp= Hnp1
   Up= Unp1
   ok=0

 END FUNCTION Upstream

!   !************************************************************************************

 FUNCTION RungeKutta4(EDO,hm,hc,hp,Um,Uc,Up,t,dt,dx,filter)  result(ok)
   ! """
   ! Runge-Kutta 4 time integration (4th order)
   ! """
   IMPLICIT NONE
   CHARACTER(LEN=*), INTENT(IN) :: EDO
   REAL(KIND=r8), INTENT(INOUT) :: hm(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: hc(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: hp(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Um(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Uc(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Up(Idim)

   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   CHARACTER(LEN=*), INTENT(IN) :: filter
   REAL(KIND=r8):: k1 (SIZE(hc))
   REAL(KIND=r8):: k2 (SIZE(hc))
   REAL(KIND=r8):: k3 (SIZE(hc))
   REAL(KIND=r8):: k4 (SIZE(hc))
   REAL(KIND=r8):: Unp1(SIZE(hc))
   REAL(KIND=r8):: Hnp1(SIZE(hc))

   INTEGER :: test,ok

   IF(EDO=='CentredSpace')THEN
      k1 = CentredSpaceH(hc             , t          , dt,dx)
      k2 = CentredSpaceH(hc + 0.5*dt*k1 , t + 0.5*dt , dt,dx)
      k3 = CentredSpaceH(hc + 0.5*dt*k2 , t + 0.5*dt , dt,dx)
      k4 = CentredSpaceH(hc +     dt*k3 , t +     dt , dt,dx)
   ELSE IF(EDO=='UpwindSpace')THEN
       k1 = UpwindSpaceH(hc             , t         , dt,dx)
       k2 = UpwindSpaceH(hc + 0.5*dt*k1 , t + 0.5*dt , dt,dx)
       k3 = UpwindSpaceH(hc + 0.5*dt*k2 , t + 0.5*dt , dt,dx)
       k4 = UpwindSpaceH(hc +     dt*k3 , t +    dt , dt,dx)
   ELSE IF(EDO=='Centred4thCS')THEN
       k1 = Solve_4thCS_H(hc            , t         , dt,dx)
       k2 = Solve_4thCS_H(hc + 0.5*dt*k1 , t + 0.5*dt , dt,dx)
       k3 = Solve_4thCS_H(hc + 0.5*dt*k2 , t + 0.5*dt , dt,dx)
       k4 = Solve_4thCS_H(hc +    dt*k3 , t +    dt , dt,dx)
   END IF



   IF(TRIM(filter) =='noFilter')THEN
      Unp1=0.0
      Unp1 = Uc + 1./6.*(dt)*( k1 + 2.*k2 + 2.*k3 + k4 )
   ELSE IF(TRIM(filter) =='FilterRA')THEN
      Unp1=0.0
      Unp1 = Uc + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )
      test=Filter_RA(Unp1,Uc,Um)
      Unp1 = Uc + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )
   ELSE IF(TRIM(filter) =='FilterRAW')THEN
      Unp1=0.0
      Unp1 = Uc + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )
      test=Filter_RAW(Unp1,Uc,Um)
      ! CORRECAO (bug 4): o codigo original, apos chamar Filter_RAW (que ja
      ! devolve em Unp1 o valor corretamente filtrado, via o ajuste
      ! Qf = Qf + Deslc*(beta-1) dentro da propria funcao), descartava esse
      ! resultado (guardava em Up so para nao perde-lo), RECALCULAVA Unp1 do
      ! zero usando k1..k4 (tendencias calculadas ANTES do filtro) somadas ao
      ! Uc ja filtrado, e entao misturava 50/50 os dois valores. Isso jogava
      ! fora metade do efeito do filtro RAW e misturava informacao "stale"
      ! (tendencias de antes do filtro com estado de depois do filtro).
      ! Mantemos apenas o Unp1 devolvido por Filter_RAW.
   ELSE
      Unp1=0.0
      Unp1 = Uc + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )
   END IF
   ! Difusao computacional de 2a e 4a ordem (opcional, ativada por
   ! Kdif_RK4>0 -- controle INDEPENDENTE do Kdif usado no Upstream e no
   ! CentredTimeSpace, para poder ligar/desligar ou escalar a difusao em
   ! cada esquema separadamente durante os experimentos com os alunos),
   ! somada como termo explicito adicional apos o passo RK4 -- necessaria
   ! sobretudo nos esquemas centrados (CentredSpace, Centred4thCS), que nao
   ! tem nenhuma difusao intrinseca e dependem so dos filtros RA/RAW para
   ! nao acumular ruido de grade. Ver funcao ComputationalDiffusion.
   IF( Kdif_RK4>0 ) Unp1 = Unp1 + dt*ComputationalDiffusion(Uc,dt,dx)
   Up=Unp1


   IF(EDO=='CentredSpace')THEN
       k1 = CentredSpaceU(Up             , t          , dt,dx)
       k2 = CentredSpaceU(Up + 0.5*dt*k1 , t + 0.5*dt , dt,dx)
       k3 = CentredSpaceU(Up + 0.5*dt*k2 , t + 0.5*dt , dt,dx)
       k4 = CentredSpaceU(Up +     dt*k3 , t +     dt , dt,dx)
   ELSE IF(EDO=='UpwindSpace')THEN
       k1 = UpwindSpaceU(Up             , t         , dt,dx)
       k2 = UpwindSpaceU(Up + 0.5*dt*k1 , t + 0.5*dt , dt,dx)
       k3 = UpwindSpaceU(Up + 0.5*dt*k2 , t + 0.5*dt , dt,dx)
       k4 = UpwindSpaceU(Up +     dt*k3 , t +    dt , dt,dx)
   ELSE IF(EDO=='Centred4thCS')THEN
       k1 = Solve_4thCS_U(Up            , t         , dt,dx)
       k2 = Solve_4thCS_U(Up + 0.5*dt*k1 , t + 0.5*dt , dt,dx)
       k3 = Solve_4thCS_U(Up + 0.5*dt*k2 , t + 0.5*dt , dt,dx)
       k4 = Solve_4thCS_U(Up +    dt*k3 , t +    dt , dt,dx)
   END IF

   IF(TRIM(filter) =='noFilter')THEN
      Hnp1=0.0
      Hnp1 = hc + 1./6.*(dt)*( k1 + 2.*k2 + 2.*k3 + k4 )
   ELSE IF(TRIM(filter) =='FilterRA')THEN
      Hnp1=0.0
      Hnp1 = hc + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )
      test=Filter_RA(Hnp1,hc,hm)
      Hnp1 = hc + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )
   ELSE IF(TRIM(filter) =='FilterRAW')THEN
      Hnp1=0.0
      Hnp1 = hc + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )
      test=Filter_RAW(Hnp1,hc,hm)
      ! CORRECAO (bug 4): mesma correcao aplicada acima para o campo h --
      ! Filter_RAW ja devolve o Hnp1 corretamente filtrado; removido o
      ! recalculo com k1..k4 "stale" e a mistura 50/50 que descartava metade
      ! do efeito do filtro.
   ELSE
      Hnp1=0.0
      Hnp1 = hc + 1./6.*dt*( k1 + 2.*k2 + 2.*k3 + k4 )
   END IF
   ! Difusao computacional de 2a e 4a ordem (opcional, ativada por
   ! Kdif_RK4>0) -- ver comentario equivalente logo acima, para o campo U.
   IF( Kdif_RK4>0 ) Hnp1 = Hnp1 + dt*ComputationalDiffusion(hc,dt,dx)
   hp=Hnp1
 END FUNCTION RungeKutta4

!************************************************************************************


 FUNCTION CentredSpaceH(Q,t,dt,dx)  result(rhs)
   ! """
   ! Centred in space right-hand-side computation (2nd order)
   ! """
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   INTEGER                   ::   xb,xc,xf
   REAL(KIND=r8)             :: rhs (SIZE(Q))
   INTEGER                   :: N0
   INTEGER                   :: N1
   INTEGER                   :: i

   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)

   DO i=N0,N1
      CALL index(i,xb,xc,xf)
      rhs(i) = -Grav*(Q(xf)  -  Q(xb) )/(2*dx)
   END DO

 END FUNCTION CentredSpaceH




 FUNCTION CentredSpaceU(Q,t,dt,dx)  result(rhs)
   ! """
   ! Centred in space right-hand-side computation (2nd order)
   ! """
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   INTEGER                   ::   xb,xc,xf
   REAL(KIND=r8)             :: rhs (SIZE(Q))
   INTEGER                   :: N0
   INTEGER                   :: N1
   INTEGER                   :: i

   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)

   DO i=N0,N1
      CALL index(i,xb,xc,xf)
      rhs(i) = -H0*(Q(xf)  -  Q(xb) )/(2*dx)
   END DO

 END FUNCTION CentredSpaceU

!   !************************************************************************************


 FUNCTION Solve_4thCS_H(Q,t,dt,dx)  result(rhs)
   ! """
   ! Centred in space right-hand-side computation (4nd order)
   ! """
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   INTEGER                   :: xb3,xb2,xb,xc,xf,xf2,xf3
   REAL(KIND=r8)             :: rhs  (SIZE(Q))
   INTEGER                   :: N0
   INTEGER                   :: N1
   INTEGER                   :: i

   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)

   ! F(j,n) - F(j-1,n)
   !--------------------
   !      dx
   DO i=N0,N1
        CALL index2(i,xb3,xb2,xb,xc,xf,xf2,xf3)
        rhs(i) = -Grav * (-Q(xf2) + 8.0*Q(xf) - 8.0*Q(xb) + Q(xb2))/ &
                             ((12*dx))
   END DO

 END FUNCTION Solve_4thCS_H


 FUNCTION Solve_4thCS_U(Q,t,dt,dx)  result(rhs)
   ! """
   ! Centred in space right-hand-side computation (4nd order)
   ! """
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   INTEGER                   :: xb3,xb2,xb,xc,xf,xf2,xf3
   REAL(KIND=r8)             :: rhs  (SIZE(Q))
   INTEGER                   :: N0
   INTEGER                   :: N1
   INTEGER                   :: i

   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)

   ! F(j,n) - F(j-1,n)
   !--------------------
   !      dx
   DO i=N0,N1
        CALL index2(i,xb3,xb2,xb,xc,xf,xf2,xf3)
        rhs(i) = -H0 * (-Q(xf2) + 8.0*Q(xf) - 8.0*Q(xb) + Q(xb2))/ &
                             ((12*dx))
   END DO

 END FUNCTION Solve_4thCS_U

!   !************************************************************************************

 FUNCTION UpwindSpaceH(Q,t,dt,dx)  result(rhs)
   ! """
   ! Forward in space right-hand-side computation (1st order)
   ! """
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   INTEGER             ::   xb,xc,xf
   REAL(KIND=r8)       :: rhs (SIZE(Q))
   INTEGER             :: N0
   INTEGER             :: N1
   INTEGER             :: i
   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)
   ! F(j,n) - F(j-1,n)
   !--------------------
   !      dx
   DO i=N0,N1
       CALL index(i,xb,xc,xf)
        rhs(i) = -Grav*(Q(xc)  - Q(xb) )/(dx)
   END DO

 END FUNCTION UpwindSpaceH


 FUNCTION UpwindSpaceU(Q,t,dt,dx)  result(rhs)
   ! """
   ! Forward in space right-hand-side computation (1st order)
   ! """
   IMPLICIT NONE
   REAL(KIND=r8), INTENT(IN) :: Q(Idim)
   REAL(KIND=r8), INTENT(IN) :: t
   REAL(KIND=r8), INTENT(IN) :: dt
   REAL(KIND=r8), INTENT(IN) :: dx
   INTEGER             ::   xb,xc,xf
   REAL(KIND=r8)       :: rhs (SIZE(Q))
   INTEGER             :: N0
   INTEGER             :: N1
   INTEGER             :: i
   N0=LBOUND(Q,DIM=1)
   N1=UBOUND(Q,DIM=1)
   ! F(j,n) - F(j-1,n)
   !--------------------
   !      dx
   DO i=N0,N1
       CALL index(i,xb,xc,xf)
        rhs(i) = -H0*(Q(xc)  -  Q(xb) )/(dx)
   END DO

 END FUNCTION UpwindSpaceU
 !------------------------------------------------------------------------------------------
 FUNCTION Filter_RA(Qf,Qc,Qb)  RESULT (ok)
    IMPLICIT NONE
   REAL(KIND=r8), INTENT(INOUT) :: Qf(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Qc(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Qb(Idim)
   INTEGER             :: N0
   INTEGER             :: N1
   INTEGER             ::   xb,xc,xf
   INTEGER :: i
   INTEGER :: ok
   N0=LBOUND(Qc,DIM=1)
   N1=UBOUND(Qc,DIM=1)
   DO i=N0,N1
      CALL index(i,xb,xc,xf)
      Deslc(xc) = alfa*(Qb(xc) - 2.0*Qc(xc) + Qf(xc) )
      Qc(xc) = Qc(xc) + Deslc(xc)
   END DO
    ok=0
 END FUNCTION Filter_RA

!   !************************************************************************************

 FUNCTION Filter_RAW(Qf,Qc,Qb)  RESULT (ok)
    IMPLICIT NONE
   REAL(KIND=r8), INTENT(INOUT) :: Qf(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Qc(Idim)
   REAL(KIND=r8), INTENT(INOUT) :: Qb(Idim)
   INTEGER             :: N0
   INTEGER             :: N1
   INTEGER             ::   xb,xc,xf
   INTEGER :: i
   INTEGER :: ok
   N0=LBOUND(Qc,DIM=1)
   N1=UBOUND(Qc,DIM=1)
   DO i=N0,N1
       CALL index(i,xb,xc,xf)
       Deslc(xc) = alfa*(Qb(xc) - 2.0*Qc(xc) + Qf(xc) )
       Qc(xc) = Qc(xc) + Deslc(xc)
       Qf(xc) = Qf(xc) + Deslc(xc)*(beta-1.0)
    END DO
    ok=0
 END FUNCTION Filter_RAW

!   !************************************************************************************

   SUBROUTINE index(i,xb,xc,xf)
      IMPLICIT NONE
      INTEGER, INTENT(IN   ) :: i
      INTEGER, INTENT(OUT  ) :: xb,xc,xf
      ! CORRECAO (bug 3): as fronteiras i==1 e i==Idim usavam xb=i (repetindo
      ! o proprio ponto) e xf=Idim (repetindo o ultimo ponto), impondo, na
      ! pratica, uma condicao de contorno reflexiva/gradiente-zero. Isso e
      ! inconsistente com o resto do codigo, que assume dominio PERIODICO:
      ! AnaliticFunction faz o pulso reaparecer do outro lado do dominio
      ! (xf0=0 quando yyf >= xa(Idim)), e a subrotina index2 (usada no
      ! esquema de 4a ordem) ja faz o wrap periodico corretamente. Corrigido
      ! para fechar o anel: ponto anterior a i=1 e Idim; ponto seguinte a
      ! i=Idim e 1.
      IF(i==1) THEN
        xb=Idim
        xc=i
        xf=i+1
      ELSE IF(i==Idim)THEN
        xb=Idim-1
        xc=Idim
        xf=1
      ELSE
        xb=i-1
        xc=i
        xf=i+1
      END IF
   END SUBROUTINE index

   SUBROUTINE index2(i,xb3,xb2,xb,xc,xf,xf2,xf3)
      IMPLICIT NONE
      INTEGER, INTENT(IN   ) :: i
      INTEGER, INTENT(OUT  ) :: xb3,xb2,xb,xc,xf,xf2,xf3
      IF(i==1) THEN
        xb3=Idim-2
        xb2=Idim-1
        xb=Idim
        xc=i
        xf=i+1
        xf2=i+2
        xf3=i+3
      ELSE IF(i==2)THEN
        xb3=Idim-1
        xb2=Idim
        xb=i-1
        xc=i
        xf=i+1
        xf2=i+2
        xf3=i+3
      ELSE IF(i==3)THEN
        xb3=Idim
        xb2=i-2
        xb=i-1
        xc=i
        xf=i+1
        xf2=i+2
        xf3=i+3
      ELSE IF(i==Idim)THEN
        xb3=Idim-3
        xb2=Idim-2
        xb=Idim-1
        xc=i
        xf=1
        xf2=2
        xf3=3
      ELSE IF(i==Idim-1)THEN
        xb3=Idim-4
        xb2=Idim-3
        xb=Idim-2
        xc=i
        xf=Idim
        xf2=1
        xf3=2
      ELSE IF(i==Idim-2)THEN
        xb3=Idim-5
        xb2=Idim-4
        xb=Idim-3
        xc=i
        xf=Idim-1
        xf2=Idim
        xf3=1
      ELSE
        xb3=i-3
        xb2=i-2
        xb=i-1
        xc=i
        xf=i+1
        xf2=i+2
        xf3=i+3
      END IF
   END SUBROUTINE index2
END MODULE ModAdvection
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PROGRAM  Main
  USE Class_Fields, Only : Init_Class_Fields,DeltaT,DeltaX,ua, u,um,up, hm,h,hp,r8,Idim,C,Grav,H0
  USE ModAdvection, Only : AnaliticFunction,Upstream,CentredTimeSpace,RungeKutta4
  USE Class_WritetoGrads, Only : SchemeWriteCtl,SchemeWriteData,InitClass_WritetoGrads
   IMPLICIT NONE
   REAL(KIND=r8)               :: tend = 10000!End time of simulation
   CHARACTER(LEN=10) :: scheme      = 'RK4CS4'  !
                                                   ! Possible values: UPSTREAM,
                                                   !                   CTCS,
                                                   !                   RK4CS,
                                                   !                   RK4CS4
                                                   !                   RK4UP,
                                                   !                   RK4CS_RA,
                                                   !                   RK4CS_RAW
   REAL(KIND=r8)               :: alfa = 0.95
   REAL(KIND=r8)               :: beta = 0.99 ! 0.5<  beta  <=1

!   REAL(KIND=r8)               :: alfa = 0.3
!   REAL(KIND=r8)               :: beta = 0.8 ! 0.5<  beta  <=1
   REAL(KIND=r8)               :: Kdif =5.0e-5
   ! Kdif_RK4: controle INDEPENDENTE da difusao computacional de 2a/4a ordem
   ! usada dentro do esquema RungeKutta4 (Kdif continua controlando so o
   ! Upstream e o CentredTimeSpace). Ponha 0.0 para desligar so no RK4 (por
   ! exemplo, para comparar o efeito isolado dos filtros RA/RAW sem a
   ! difusao espacial extra), ou um valor >0 para ligar (o valor numerico em
   ! si so funciona como uma chave liga/desliga: Kc e eps sao recalculados
   ! internamente a partir de C, Grav, H0, dx, dt -- nao sao escalados por
   ! este numero).
   REAL(KIND=r8)               :: Kdif_RK4 =5.0e-5
   INTEGER :: irec_err,unit2

      CALL Init()
      CALL Run(irec_err,unit2)
      CALL Finalize()

  CONTAINS

  SUBROUTINE Init()
      CALL Init_Class_Fields(alfa,beta,Kdif,Kdif_RK4)
      CALL InitClass_WritetoGrads
  END SUBROUTINE Init

  SUBROUTINE Run(irec_err,unit)
      INTEGER, INTENT(INOUT) :: irec_err
      INTEGER, INTENT(IN   ) :: unit
      REAL (KIND=r8) :: termX(Idim)
      REAL (KIND=r8) :: termXa(Idim)
      REAL (KIND=r8) :: err,DT,dx,t
      INTEGER :: i,nn
      INTEGER :: it,lrec,irec,test
      irec=0
      err=0
      test=SchemeWriteData(u ,irec)
      test=SchemeWriteData(h ,irec)
      ! Time stepping
       t = 0.
       nn = 0

       DO WHILE  (t < tend )
          nn=nn+1
          DO i=1,Idim
             termXa(i)=0.0
          END DO
          ! To make simulation end exactly at tend Centred4thCS
          DT=min(DeltaT,tend-t)
          ! Propagate one time step
          if( TRIM(scheme) == 'RK4CS')THEN
             test = RungeKutta4('CentredSpace',hm,h,hp,Um,U,Up,t,DT,DeltaX,'noFilter')
          else if( TRIM(scheme) == 'RK4UP')THEN
             test = RungeKutta4('UpwindSpace',hm,h,hp,Um,U,Up,t,DT,DeltaX,'noFilter')
          else if( TRIM(scheme) == 'RK4CS4')THEN
             test = RungeKutta4('Centred4thCS',hm,h,hp,Um,U,Up,t,DT,DeltaX,'FilterRAW')
          else if( TRIM(scheme) == 'RK4CS_RA')THEN
             test = RungeKutta4('CentredSpace',hm,h,hp,Um,U,Up,t,DT,DeltaX,'FilterRA')
          else if( TRIM(scheme) == 'RK4CS_RAW')THEN
             test = RungeKutta4('CentredSpace',hm,h,hp,Um,U,Up,t,DT,DeltaX,'FilterRAW')
          else if(TRIM(scheme)=='UPSTREAM')THEN
             test = Upstream(hm,h,hp,Um,U,Up,t,DT,DeltaX)
          else if(TRIM(scheme)=='CTCS')THEN
             test = CentredTimeSpace(hm,h,hp,Um,U,Up,t,DT,DeltaX)
          else
             scheme = 'default'
             !### Comment/Uncomment desired scheme
             test = Upstream(hm,h,hp,Um,U,Up,t,DT,DeltaX)
          endif

          err=err+SUM((u-um)**2)

          t =t+DT
          um=u
          u=up
          hm=h
          h=hp
          ! CORRECAO (cosmetico): "up=up" e "hp=hp" no codigo original eram
          ! no-ops (auto-atribuicao) sem efeito; removidos por clareza.
          test=SchemeWriteData(u  ,irec)
          test=SchemeWriteData(h  ,irec)
          print*, 'iter [',nn,']  time [',t,']',MAXVAL(u),MAXVAL(h)
        END DO
        test=SchemeWriteCtl(nn)

   ! CORRECAO (cosmetico): o CFL fisico do sistema acoplado (u,h) e
   ! sqrt(Grav*H0)*DeltaT/DeltaX, nao Grav*DeltaT/DeltaX nem H0*DeltaT/DeltaX
   ! separadamente.
   PRINT*,'scheme=',TRIM(scheme),' err=',err/nn,' DeltaX=',DeltaX,' DeltaT=',DeltaT, &
          ' CFL=',SQRT(Grav*H0)*DeltaT/DeltaX

  END SUBROUTINE Run
!
  SUBROUTINE Finalize()

  END SUBROUTINE Finalize
END PROGRAM  Main
