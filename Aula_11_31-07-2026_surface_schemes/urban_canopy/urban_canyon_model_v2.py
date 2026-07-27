"""
urban_canyon_model_v2.py
=========================

Modelo de canyon urbano - Etapa 2 (prototipo de pesquisa).

Constroi sobre urban_canyon_model.py (Etapa 1) e adiciona, na ordem
pedida:

  1. Estabilidade atmosferica (Louis, 1979), usando o numero de
     Richardson bulk, seguindo a formulacao da Parte 1
     ("Modelo da Camada Superficial").
  2. Esquema numerico IMPLICITO BACKWARD (nao e uma matriz
     tridiagonal - essa e do modelo de solo em camadas da Parte 1;
     aqui e um sistema linear pequeno para (Tr, Tw, Tug), construido
     a partir da Jacobiana dos fluxos Rn, H em relacao as
     temperaturas, exatamente como descrito na secao 2.7 da Parte 4).
     A Jacobiana e obtida por diferencas finitas (em vez de derivar
     analiticamente as ~9+5 derivadas parciais descritas nos slides),
     o que preserva o metodo (Newton / backward implicito) sem risco
     de erro de transcricao das formulas individuais.
  3. Fatores de visao para reflexao unica com as fórmulas de
     "ponto central" (a energia e emitida/recebida a partir do ponto
     central de cada elemento - parede, rua), usando a formula
     classica de fator de forma ponto-a-faixa 2D:
         F(h, d1, d2) = 0.5*(d2/sqrt(d2^2+h^2) - d1/sqrt(d1^2+h^2))
     em vez da reciprocidade aproximada da Etapa 1. Esta e uma
     implementacao geometrica consistente com o conceito descrito
     nos slides (angulos phi, beta, psi, nu como angulos de visao de
     pontos centrais) mas NAO e transcricao literal das equacoes
     2.20-2.21 (que estao embutidas como imagens no PDF e nao pude
     confirmar caractere-a-caractere). Vale conferir contra o
     original se a fidelidade exata importar para publicacao.
  4. Distribuicao multi-story r(k): media ponderada sobre classes de
     altura de edificio, com a(k) (fracao cumulativa de edificios
     com >= k andares) e Ab = soma[r(k)*k] (altura media em unidades
     de andar), seguindo a secao 2.3. z0 e d0 calculados por
     MacDonald et al. (1998) a partir da densidade de area frontal
     (Ss) e altura media do edificio (Ab*h0), como descrito na secao
     2.3.1. O perfil de vento dentro do canyon usa o coeficiente de
     atencao aw = 9.6*Ss (Macdonald, 2000).

     SIMPLIFICACAO MANTIDA: a difusividade turbulenta (Km) com
     continuidade na altura superior do dossel para multiplas alturas
     de edificio (o "eta" mencionado na secao 2.3.1) NAO foi
     implementada - usamos o perfil exponencial simples dentro do
     canyon com aw calculado a partir de Ss. Isso e uma etapa de
     fechamento adicional que pode ser adicionada depois, se
     necessario.
"""

import numpy as np

SIGMA = 5.67e-8
CP = 1004.0
RHO_AIR = 1.2
KVK = 0.4
G = 9.81


# ---------------------------------------------------------------
# 1. Estabilidade (Louis, 1979) - identico ao usado na Parte 1
# ---------------------------------------------------------------
def louis_F(RiB, b=9.4, bprime=4.7, cstar=7.4, zzo=100.0, k=KVK):
    """F(z/z0, RiB) de Louis (1979).

    NOTA IMPORTANTE: a transcricao inicial usada aqui tinha o ramo
    estavel como (1-b'RiB)^-2, que diverge proximo de RiB=1/b'~0.21
    e AUMENTA a troca turbulenta com a estabilidade - o oposto do
    que deveria acontecer fisicamente. Isso foi corrigido para
    (1-b'RiB)^2 (expoente positivo), que decresce suavemente ate
    zero a medida que RiB cresce, como esperado (mais estavel =>
    menos troca turbulenta). Ainda assim, RiB e limitado por
    seguranca numerica.
    """
    RiB = np.clip(RiB, -10.0, 0.2)
    if RiB >= 0.0:
        return max((1.0 - bprime * RiB) ** 2, 1e-4)
    c = cstar * k ** 2 * b * np.sqrt(zzo) / (np.log(zzo)) ** 2
    return 1.0 - (b * RiB) / (1.0 + c * np.sqrt(abs(RiB)))


def louis_exchange_resistance(z, z0, u, theta0v, thetasv):
    """Resistencia de troca (superficie <-> nivel z) com correcao de
    estabilidade de Louis (1979). Retorna a resistencia r tal que
    H = rho*cp*(Ts - T0)/r.
    """
    u = max(u, 0.2)
    zzo = max(z / z0, 1.5)
    RiB = G * z * (theta0v - thetasv) / (thetasv * u ** 2)
    F = louis_F(RiB, zzo=zzo)
    F = max(F, 1e-3)
    Ch = (KVK ** 2) * F / (np.log(zzo)) ** 2
    Ch = max(Ch, 1e-6)
    return 1.0 / (Ch * u)


# ---------------------------------------------------------------
# 4. Geometria de rugosidade multi-story (Macdonald et al., 1998)
# ---------------------------------------------------------------
def macdonald_z0_d0(z_ave, Vuc, Ss, A=4.43, B=1.0, Cd=1.2, k=KVK):
    """z0 e d0 seguindo Macdonald et al. (1998), usando a densidade
    de area planejada (Vuc) e a densidade de area frontal (Ss).
    """
    d0 = z_ave * (1.0 + A ** (-Vuc) * (Vuc - 1.0))
    d0 = np.clip(d0, 0.0, 0.95 * z_ave)
    ratio = (1.0 - d0 / z_ave)
    inner = 0.5 * B * Cd / k ** 2 * ratio * Ss
    inner = max(inner, 1e-6)
    z0 = z_ave * ratio * np.exp(-inner ** (-0.5))
    z0 = max(z0, 0.001)
    return z0, d0


class UrbanCanyonModelV2:
    def __init__(
        self,
        bw=20.0, h0=3.0, Vuc=0.5,
        r_k=None,                     # dict {k: fracao}, default = 1 andar (100%)
        albedo_r=0.20, albedo_w=0.25, albedo_g=0.15,
        emis_r=0.90, emis_w=0.90, emis_g=0.95,
        depth_r=0.10, depth_w=0.20, depth_g=0.30,
        Cs_r=1.0e6, Cs_w=1.5e6, Cs_g=1.5e6,
        tau_d=86400.0,
        Tbi=293.0, Tdeep=291.0,
        zr=50.0,
    ):
        self.bw = bw
        self.h0 = h0
        self.Vuc = Vuc
        self.r_k = r_k if r_k is not None else {1: 1.0}
        assert abs(sum(self.r_k.values()) - 1.0) < 1e-6, "r(k) deve somar 1"
        self.n = max(self.r_k.keys())

        self.w = bw * (1.0 - Vuc) / Vuc

        # a(k): fracao cumulativa de edificios com >= k andares
        self.a_k = {
            k: sum(v for kk, v in self.r_k.items() if kk >= k)
            for k in range(1, self.n + 1)
        }
        # Ab: altura media do edificio em unidades de andar
        self.Ab = sum(k * v for k, v in self.r_k.items())
        self.z_ave = self.Ab * h0

        # densidade de area frontal Ss = (Vuc*h0/bw)*Ab  (derivado na
        # secao 2.3: area frontal total por unidade de area)
        self.Ss = (self.Vuc * self.h0 / self.bw) * self.Ab
        self.Sr = self.Vuc  # area de telhado por unidade de area (independe de k)

        self.z0, self.d0 = macdonald_z0_d0(self.z_ave, self.Vuc, self.Ss)

        self.albedo_r, self.albedo_w, self.albedo_g = albedo_r, albedo_w, albedo_g
        self.emis_r, self.emis_w, self.emis_g = emis_r, emis_w, emis_g

        self.Cr = Cs_r * depth_r
        self.Cw = Cs_w * depth_w
        self.Cg = Cs_g * depth_g

        self.tau_d = tau_d
        self.Tbi = Tbi
        self.Tdeep = Tdeep
        self.zr = zr

        self.Tr = 293.0
        self.Tw = 293.0
        self.Tg = 293.0

    # -----------------------------------------------------------
    # Geometria por classe de altura k
    # -----------------------------------------------------------
    def sky_view_factors(self, kh0):
        H = kh0 / self.w
        sky_g = np.sqrt(H ** 2 + 1.0) - H
        sky_w = 0.5 * (H + 1.0 - np.sqrt(H ** 2 + 1.0)) / H
        return sky_w, sky_g

    @staticmethod
    def _thomas_solve(a, b, c, d):
        """Algoritmo de Thomas para sistema tridiagonal a*T[i-1] +
        b*T[i] + c*T[i+1] = d. a[0] e c[-1] sao ignorados.
        """
        n = len(d)
        cp = np.zeros(n)
        dp = np.zeros(n)
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        for i in range(1, n):
            m = b[i] - a[i] * cp[i - 1]
            cp[i] = c[i] / m if i < n - 1 else 0.0
            dp[i] = (d[i] - a[i] * dp[i - 1]) / m
        x = np.zeros(n)
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    def reflection_view_factors(self, kh0, zenith=None):
        """Fatores de visao EXATOS (eq. 2.11-2.12 da Parte 4), usando
        os angulos phi, beta, psi, nu.

        phi = arctan(2*kh0 / y)              y = w - kh0*tan(theta)
        beta = (pi - sky_g) / 2
        psi = arctan(2*kh0 / (w + kh0*tan(theta)))
        nu = pi - 2*sky_w

        Esses angulos sao convertidos em fracao de energia recebida
        assumindo proporcionalidade angulo/(pi/2) (ponto receptor
        "enxerga" um quadrante de referencia) - ver ressalva no
        cabecalho do modulo: a formula angular em si e transcrita
        exatamente do PDF, a conversao angulo->fracao energetica
        segue a convencao mais usual para esse tipo de formulacao
        geometrica 2D.
        """
        sky_w, sky_g = self.sky_view_factors(kh0)
        w = self.w

        if zenith is not None and zenith < np.pi / 2.0 - 1e-6:
            tan_th = np.tan(zenith)
        else:
            tan_th = 0.0

        y = max(w - kh0 * tan_th, 1e-6)
        phi = np.arctan(2.0 * kh0 / y)
        psi = np.arctan(2.0 * kh0 / (w + kh0 * tan_th))
        beta = (np.pi - sky_g) / 2.0
        nu = np.pi - 2.0 * sky_w

        # fracao de energia refletida da estrada -> parede esquerda
        # (usa phi p/ componente direta e beta p/ componente difusa,
        # conforme a secao 2.2.3)
        vf_g_to_w_dir = np.clip(phi / (np.pi / 2.0), 0.0, 1.0)
        vf_g_to_w_dif = np.clip(beta / (np.pi / 2.0), 0.0, 1.0)

        # fracao de energia refletida (da estrada e parede esquerda)
        # -> parede direita, usando psi e nu
        vf_to_w_right_dir = np.clip(psi / (np.pi / 2.0), 0.0, 1.0)
        vf_w_to_w = np.clip(nu / np.pi, 0.0, 1.0)

        return dict(vf_g_to_w_dir=vf_g_to_w_dir, vf_g_to_w_dif=vf_g_to_w_dif,
                    vf_to_w_right_dir=vf_to_w_right_dir, vf_w_to_w=vf_w_to_w,
                    phi=phi, beta=beta, psi=psi, nu=nu)

    def direct_beam_partition(self, kh0, zenith):
        if zenith >= np.pi / 2.0 - 1e-6:
            return 0.0, 0.0
        x = kh0 * np.tan(zenith)
        x_eff = min(x, self.w)
        y = self.w - x_eff
        return y / self.w, x_eff / kh0

    # -----------------------------------------------------------
    # Radiacao por classe k, depois agregada por r(k)
    # -----------------------------------------------------------
    def radiation_balance(self, SWdir, SWdif, LWdif, zenith, Tr, Tw, Tg):
        Rn_r_sum = Rn_w_sum = Rn_g_sum = 0.0
        for k, rk in self.r_k.items():
            kh0 = k * self.h0
            sky_w, sky_g = self.sky_view_factors(kh0)
            vfs = self.reflection_view_factors(kh0, zenith)
            frac_road, frac_wall = self.direct_beam_partition(kh0, zenith)

            SW_r_in = SWdir + SWdif
            SW_w_in = SWdir * frac_wall + SWdif * sky_w
            SW_g_in = SWdir * frac_road + SWdif * sky_g

            # parede recebe reflexao da estrada: componente direta via
            # phi, componente difusa via beta (eq. 2.11); tratamos a
            # parede "direita" com psi/nu como uma segunda contribuicao
            # media (o modelo e 2D e trata left/right simetricamente
            # ao longo do ciclo diurno)
            SW_w_refl = self.albedo_g * SW_g_in * (
                0.5 * vfs["vf_g_to_w_dir"] + 0.5 * vfs["vf_g_to_w_dif"]
            ) + self.albedo_w * SW_w_in * (0.5 * vfs["vf_w_to_w"])
            SW_g_refl = self.albedo_w * SW_w_in * (0.5 * vfs["vf_to_w_right_dir"])

            Rsw_r = (1.0 - self.albedo_r) * SW_r_in
            Rsw_w = (1.0 - self.albedo_w) * (SW_w_in + SW_w_refl)
            Rsw_g = (1.0 - self.albedo_g) * (SW_g_in + SW_g_refl)

            LW_w_in = LWdif * sky_w + self.emis_g * SIGMA * Tg ** 4 * vfs["vf_g_to_w_dif"]
            LW_g_in = LWdif * sky_g + self.emis_w * SIGMA * Tw ** 4 * vfs["vf_to_w_right_dir"]

            Rlw_r = self.emis_r * (LWdif - SIGMA * Tr ** 4)
            Rlw_w = self.emis_w * (LW_w_in - SIGMA * Tw ** 4)
            Rlw_g = self.emis_g * (LW_g_in - SIGMA * Tg ** 4)

            Rn_r_sum += rk * (Rsw_r + Rlw_r)
            Rn_w_sum += rk * (Rsw_w + Rlw_w)   # ponderado por altura (~area de parede)
            Rn_g_sum += rk * (Rsw_g + Rlw_g)

        return Rn_r_sum, Rn_w_sum, Rn_g_sum

    # -----------------------------------------------------------
    # Resistencias aerodinamicas com estabilidade (Louis, 1979)
    # -----------------------------------------------------------
    def canyon_air_temperature(self, Tr, Tw, Tg, Tref, r_r, r_w, r_g, r_au):
        # mantido por compatibilidade/depuracao pontual; o modelo usa
        # solve_canyon_air_column() como fonte de verdade.
        num = Tr / r_r + Tw / r_w + Tg / r_g + Tref / r_au
        den = 1.0 / r_r + 1.0 / r_w + 1.0 / r_g + 1.0 / r_au
        return num / den

    # -----------------------------------------------------------
    # 4b. Perfil vertical de Km com continuidade no topo do dossel
    #     (secao 2.3.1: Km proporcional ao vento local dentro do
    #     dossel, e proporcional a (z-d0) acima; eta ajusta a
    #     continuidade em z=z2=n*h0)
    # -----------------------------------------------------------
    def vertical_profile(self, u_ref):
        u_ref = max(u_ref, 0.3)
        z2 = self.n * self.h0
        z_above = max(self.zr - self.d0, z2 - self.d0 + 1.0)
        u_star = KVK * u_ref / np.log(z_above / self.z0)

        aw = 9.6 * self.Ss
        k_idx = np.arange(1, self.n + 1)
        u_k = u_ref * np.exp(-aw * (1.0 - k_idx / self.n))  # u(k), k=1..n; u(n)=u_ref=u2

        Km_top = KVK * u_star * (z2 - self.d0)  # Km logo acima do dossel, em z2
        eta = Km_top / u_k[-1]                   # continuidade em k=n (eq. da secao 2.3.1)
        Km_k = eta * u_k

        return dict(k=k_idx, z=k_idx * self.h0, u=u_k, Km=Km_k,
                    u_star=u_star, eta=eta, z2=z2)

    def solve_canyon_air_column(self, Tw, Tg, Tref, u_ref):
        """Resolve o perfil vertical de Tau(k), k=1..n, por balanco
        de difusao turbulenta (sem armazenamento, sistema tridiagonal
        resolvido pelo algoritmo de Thomas), com fontes de parede
        (ponderadas por a(k)) e de rua (nivel 1), e condicao de
        contorno superior via r_au (Louis) no topo do dossel.

        Retorna Tau (array, 1 valor por andar), Hw_avg (fluxo medio
        parede->ar, ponderado por area), Hg (rua->ar, nivel 1),
        Hatm (ar->atmosfera, nivel n).
        """
        prof = self.vertical_profile(u_ref)
        n = self.n
        h0 = self.h0
        Km = prof["Km"]

        # resistencia parede->ar em cada nivel (usa vento local u(k))
        r_w_k = np.array([1.0 / (0.1 * max(uk, 0.15) + 0.01) for uk in prof["u"]])
        r_g = 1.0 / (0.1 * max(prof["u"][0], 0.15) + 0.01)
        r_au = louis_exchange_resistance(max(self.zr - self.d0, 1.0), self.z0, u_ref, Tref, Tref)

        a_k = np.array([self.a_k[k] for k in range(1, n + 1)])  # fracao de area com parede no nivel k

        # monta sistema tridiagonal: A_i*Tau(i-1) + B_i*Tau(i) + C_i*Tau(i+1) = D_i
        A = np.zeros(n)
        B = np.zeros(n)
        C = np.zeros(n)
        D = np.zeros(n)

        cond_w = RHO_AIR * CP / np.where(r_w_k > 0, r_w_k, 1e6) * a_k  # condutancia parede->ar por nivel

        for i in range(n):
            Km_lo = 0.5 * (Km[i - 1] + Km[i]) if i > 0 else Km[0]
            Km_hi = 0.5 * (Km[i] + Km[i + 1]) if i < n - 1 else Km[-1]
            g_lo = RHO_AIR * CP * Km_lo / h0 ** 2
            g_hi = RHO_AIR * CP * Km_hi / h0 ** 2

            B[i] = -(g_lo if i > 0 else 0.0) - (g_hi if i < n - 1 else 0.0) - cond_w[i]
            if i > 0:
                A[i] = g_lo
            if i < n - 1:
                C[i] = g_hi
            D[i] = -cond_w[i] * Tw

            if i == 0:
                cond_g = RHO_AIR * CP / r_g
                B[i] -= cond_g
                D[i] -= cond_g * Tg
            if i == n - 1:
                cond_top = RHO_AIR * CP / r_au
                B[i] -= cond_top
                D[i] -= cond_top * Tref

        Tau_k = self._thomas_solve(A, B, C, D)

        Hw_avg = np.sum(a_k * RHO_AIR * CP * (Tw - Tau_k) / r_w_k) / max(np.sum(a_k), 1e-6)
        Hg = RHO_AIR * CP * (Tg - Tau_k[0]) / r_g
        Hatm = RHO_AIR * CP * (Tau_k[-1] - Tref) / r_au

        return Tau_k, Hw_avg, Hg, Hatm


    def _rhs(self, T, SWdir, SWdif, LWdif, zenith, Tref, u_ref):
        Tr, Tw, Tg = T
        Rn_r, Rn_w, Rn_g = self.radiation_balance(SWdir, SWdif, LWdif, zenith, Tr, Tw, Tg)

        Tau_profile, Hw, Hg, Hatm = self.solve_canyon_air_column(Tw, Tg, Tref, u_ref)
        Tau_top = Tau_profile[-1]

        r_r = louis_exchange_resistance(max(self.zr - self.d0, 1.0), self.z0, u_ref, Tref, Tr)
        Hr = RHO_AIR * CP * (Tr - Tref) / r_r

        omega = 2.0 * np.pi / self.tau_d
        dTr = (Rn_r - Hr) / self.Cr - omega * (Tr - self.Tbi)
        dTw = (Rn_w - Hw) / self.Cw - omega * (Tw - self.Tbi)
        dTg = (Rn_g - Hg) / self.Cg - omega * (Tg - self.Tdeep)

        extras = dict(Tau=Tau_top, Rn_r=Rn_r, Rn_w=Rn_w, Rn_g=Rn_g,
                      Hr=Hr, Hw=Hw, Hg=Hg, Hatm=Hatm)
        return np.array([dTr, dTw, dTg]), extras

    # -----------------------------------------------------------
    # 2. Passo implicito backward (Newton com Jacobiana numerica)
    # -----------------------------------------------------------
    def step_implicit(self, dt, SWdir, SWdif, LWdif, zenith, Tref, u_ref,
                       n_iter=2, eps=1e-3):
        T0 = np.array([self.Tr, self.Tw, self.Tg])
        T = T0.copy()

        for _ in range(n_iter):
            F, extras = self._rhs(T, SWdir, SWdif, LWdif, zenith, Tref, u_ref)

            # Jacobiana numerica dF_i/dT_j
            J = np.zeros((3, 3))
            for j in range(3):
                Tpert = T.copy()
                Tpert[j] += eps
                Fpert, _ = self._rhs(Tpert, SWdir, SWdif, LWdif, zenith, Tref, u_ref)
                J[:, j] = (Fpert - F) / eps

            # sistema: (I/dt - J) * DeltaT = F(T) + J*(T - T0)/dt ... 
            # formulacao padrao do backward implicito:
            # (T_new - T0)/dt = F(T0) + J*(T_new - T0)
            # => (I/dt - J) * (T_new - T0) = F(T0)
            A = np.eye(3) / dt - J
            b = F
            dT = np.linalg.solve(A, b)
            T = T0 + dT

        F_final, extras = self._rhs(T, SWdir, SWdif, LWdif, zenith, Tref, u_ref)
        self.Tr, self.Tw, self.Tg = T
        return dict(Tr=self.Tr, Tw=self.Tw, Tg=self.Tg, **extras)

    def run(self, hours, dt_s, forcing_fn):
        n = int(hours * 3600.0 / dt_s)
        keys = ("t", "Tr", "Tw", "Tg", "Tau", "Tref",
                "Rn_r", "Rn_w", "Rn_g", "Hr", "Hw", "Hg", "Hatm")
        out = {k: np.zeros(n) for k in keys}
        for i in range(n):
            t_h = i * dt_s / 3600.0
            f = forcing_fn(t_h)
            state = self.step_implicit(dt_s, f["SWdir"], f["SWdif"], f["LWdif"],
                                        f["zenith"], f["Tref"], f["u_ref"])
            out["t"][i] = t_h
            out["Tref"][i] = f["Tref"]
            for k, v in state.items():
                out[k][i] = v
        return out
