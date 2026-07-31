# -*- coding: utf-8 -*-
"""
lightning_mod.py
==================

Traducao literal, funcao a funcao, de `mpas_atmphys_lightning.F90` para
Python. Mantem os mesmos nomes de subrotina/funcao, as mesmas
constantes, a mesma logica de ramificacao (branches) e comentarios
originais (traduzidos), para que a correspondencia com o Fortran seja
1:1 e auditavel.

USO (mesma interface do modulo Fortran compilado via f2py, para que
`test_synthetic.py` funcione sem alteracoes):

    import lightning_mod as lm
    drv = lm.mpas_atmphys_lightning.driver_lightning
    fr, fr_cg, fr_ic = drv(z_m, t_c, rho, w, qg, qs, qc,
                            dx_km=2.0, cu_active=False, cldtop_km=0.0)

Diferencas de interface Fortran -> Python (deliberadas, nao mudam a
fisica):
  - `nz` nao e mais um argumento explicito: e obtido de `len(z_m)`
    (o array numpy ja carrega seu proprio tamanho).
  - Argumentos `intent(out)` do Fortran viram valores de retorno em
    tupla, na mesma ordem em que apareciam na assinatura da subrotina.
  - `real(rk), parameter` (Fortran) vira constante de modulo em Python
    (float nativo -- numpy usa float64 por padrao, equivalente ao
    `selected_real_kind(12)` do Fortran).
  - Loops `do k=2,nz` (Fortran, 1-indexado) viram `for k in range(1,nz)`
    (Python, 0-indexado) -- a relacao k <-> k-1 (nivel atual vs.
    anterior) e preservada exatamente, pois o deslocamento relativo
    entre os dois indices nao muda com a base de indexacao.

Referencias (identicas as do modulo original):
    McCaul, E.W. et al. (2009), WAF, 24, 709-729.
    Price, C. & Rind, D. (1992), JGR, 97, 9919-9933.
    Price, C. & Rind, D. (1993), JAM, 32, 170-181.
    Barthe, C. et al. (2010), JGR, 115, D24202.
    Thompson, G. et al. -- esquema de microfisica (N0,g diagnosticado a
        partir de qg, usado no fechamento de tamanho do graupel no EP)

Paulo Kubota / INPE-CPTEC -- esqueleto didatico (MET-756-4) e prototipo
de integracao operacional no MONAN. Esta versao: traducao Python.
"""

import numpy as np

PI = 3.14159265358979


class mpas_atmphys_lightning:
    """
    Namespace que espelha o modulo Fortran `mpas_atmphys_lightning`.
    Todas as funcoes sao `staticmethod` para permitir o acesso no
    estilo `mpas_atmphys_lightning.driver_lightning(...)`, igual ao
    acesso `modulo.subrotina` de um modulo Fortran compilado via f2py.
    """

    # ------------------------------------------------------------------
    # Coeficientes empiricos (expor como namelist na integracao real)
    # ------------------------------------------------------------------
    coef_a_mccaul = 0.042  # McCaul et al. (2009), eq. F1: slope da
                           # regressao linear FR = k1*(w*qg)|_-15C

    coef_alpha_pr = 1.71e-9  # FR_total = alpha * H^beta (H em km)
    coef_beta_pr = 8.7
    # Recalibrado com Barthe et al. (2010, JGR) -- regressao entre altura
    # do topo (cloud top height) e a TAXA DE FLASH TOTAL observada/
    # simulada (r=0.74), a partir de simulacoes WRF de 2 casos
    # (STERAO-A e Alabama do Norte). Substitui os coeficientes originais
    # de Price & Rind (1992) (alpha=3.44e-5, beta=4.9). ATENCAO:
    # calibracao extratropical (EUA), nao tropical -- ainda precisa de
    # recalibracao regional com RINDAT/GLM sobre casos de MCS na Bacia
    # do Prata / Amazonia antes de uso operacional no MONAN.

    t_peak_nic = -15.0  # nivel-alvo do McCaul original (2009): pico de
                        # eficiencia NIC graupel-cristal (Takahashi 1978/
                        # Saunders), onde o paper avalia o fluxo w*qg
                        # (nao integrado)

    # -- transicao scale-aware (sigmoide em dx) --------------------------
    # Em vez de chaveamento rigido if(dx_km > dx_threshold_km), a fracao
    # "convectiva parametrizada" sigma_cu cresce suavemente de 0 (malha
    # fina, convection-permitting) a 1 (malha grossa, cumulus dominante).
    # Evita salto artificial na faixa de resolucao "cinzenta" (~3-10 km)
    # onde o Grell-Freitas ja faz sua propria transicao suave (fallback
    # ate que o sigma real do GF seja exposto no pool de diagnostico).
    dx_threshold_km = 6.0   # centro da transicao (km)
    dx_transition_km = 2.0  # largura da transicao (km)

    # -- Potencial de Eletrificacao (EP) ---------------------------------
    # Proxy fisico que substitui o McCaul (massa simples) na branch de
    # grade explicita/convection-permitting, portado do prototipo Python
    # (ep_prototype.py) ja validado e com os 2 bugs corrigidos:
    #   (1) conversao de EW (agua liquida efetiva) sem fator 1e-3 espurio
    #   (2) magnitude da carga transferida cai perto da temperatura de
    #       reversao (dist_factor), e cresce com a intensidade do updraft
    # Teoria: separacao de carga nao-indutiva (NIC) graupel-cristal
    # (Takahashi 1978; Saunders & Peck 1998; Jayaratne et al. 1983).
    #
    # ATUALIZACAO (curso MET-756-4, apostila do esquema Thompson): a
    # concentracao numerica e o diametro caracteristico do graupel NAO
    # usam mais um diametro fixo assumido -- sao derivados do parametro
    # de interceptacao N0,g diagnosticado a partir do proprio qg,
    # exatamente como o Thompson scheme faz internamente:
    #   N0,g = max(1e4, min(200/qg, 5e6))  [m^-4], qg em kg/kg
    # seguido do fechamento da distribuicao exponencial
    # (rho_graupel=400 kg/m3, igual ao valor usado no Thompson) para
    # obter lambda_g, Ng e o diametro caracteristico Dg=1/lambda_g.
    #
    # ATENCAO: constantes de microfisica assumida (D0_ICE, densidades) e
    # coeficientes da lei de potencia (coef_a_ep, coef_b_ep) ainda sao
    # ORDENS DE GRANDEZA DA LITERATURA, NAO CALIBRADAS -- precisam de
    # ajuste contra dados reais (RINDAT/GLM) antes de uso operacional.
    rho_graupel_ep = 400.0        # kg/m3, densidade do graupel (Thompson)
    d0_ice_ep = 2.0e-4            # m, diametro medio assumido cristal
    rho_ice_ep = 900.0            # kg/m3, densidade assumida do gelo
    v_ice_ep = 0.5                # m/s, veloc. terminal tipica cristal
    d0_graupel_fallback_ep = 2.0e-3  # m, usado so quando qg<=0

    t_top_ep = -40.0  # limite superior camada fase mista (C)
    t_bot_ep = 0.0    # limite inferior camada fase mista (C)

    # lei de potencia FR_EP = a_ep * EP^b_ep -- PLACEHOLDER, calibrar depois
    coef_a_ep = 1.0e-3
    coef_b_ep = 0.5

    # ====================================================================
    # driver_lightning
    # ====================================================================
    @staticmethod
    def driver_lightning(z_m, t_c, rho, w, qg, qs, qc, dx_km,
                          cu_active, cldtop_km):
        """
        Decide o esquema por coluna com base na resolucao local (dx) e
        no esquema de cumulus ativo (cu_active).

        Se cu_active = True (conveccao parametrizada, sem graupel de
        grade confiavel) -> usa Price-Rind (precisa da altura do topo
        convectivo, ja diagnosticada pelo esquema de cumulus, ex.
        Grell-Freitas: cldtop_km).

        Se cu_active = False (convection-permitting, graupel resolvido)
        -> usa o Potencial de Eletrificacao (EP) via integral de carga
        na camada de fase mista.

        Retorna
        -------
        (flash_rate, flash_rate_cg, flash_rate_ic) : taxas de flash
        totais, nuvem-solo e intra-nuvem (flashes/min).
        """
        m = mpas_atmphys_lightning

        # fracao "convectiva parametrizada" (0 = malha fina/convection-
        # permitting -> so EP; 1 = malha grossa/cumulus dominante -> so
        # Price-Rind). Se cu_active=False em toda a coluna (esquema de
        # cumulus desligado no dominio), forca sigma_cu=0 independente de dx.
        if not cu_active:
            sigma_cu = 0.0
        else:
            sigma_cu = m.sigmoid_dx(dx_km)

        # --- via Price-Rind/Barthe (altura do topo convectivo) ---------
        # CORRECAO: flashrate_pricerind retorna o flash rate TOTAL (assim
        # como a regressao de Barthe et al. 2010 contra cloud top height,
        # e a formula original de Price & Rind 1992) -- nao apenas a
        # fracao CG como uma versao anterior deste modulo assumia por
        # engano. O total e dividido entre CG e IC usando a razao IC/CG
        # (Price & Rind 1993).
        h_eff = cldtop_km
        fr_pr_total = m.flashrate_pricerind(h_eff)
        ratio_ic_cg = m.calc_ic_cg_ratio(h_eff)
        fr_pr_cg = fr_pr_total / (1.0 + ratio_ic_cg)
        fr_pr_ic = fr_pr_total - fr_pr_cg
        fr_pr = fr_pr_total

        # --- via EP (potencial de eletrificacao, grade explicita) -------
        # Substitui o proxy de massa simples do McCaul nesta branch: o EP
        # agrega sensibilidade a fase de crescimento do graupel (seco/
        # umido, via reversao de sinal dependente de LWC/qc) que o
        # McCaul nao ve. flashrate_mccaul_column permanece disponivel
        # (publica) para comparacao/diagnostico, mas nao e mais usada
        # pelo driver.
        ep_mag, ep_signed, fr_ep = m.flashrate_ep_column(z_m, t_c, rho, w, qg, qs, qc)
        # sem info explicita de topo, usa particao IC/CG generica (~90/10)
        # ate haver calibracao regional com RINDAT/GLM
        fr_ep_ic = fr_ep * 0.90
        fr_ep_cg = fr_ep * 0.10

        # --- mistura ponderada pela fracao convectiva (scale-aware) -----
        flash_rate = sigma_cu * fr_pr + (1.0 - sigma_cu) * fr_ep
        flash_rate_cg = sigma_cu * fr_pr_cg + (1.0 - sigma_cu) * fr_ep_cg
        flash_rate_ic = sigma_cu * fr_pr_ic + (1.0 - sigma_cu) * fr_ep_ic

        return flash_rate, flash_rate_cg, flash_rate_ic

    # ====================================================================
    # sigmoid_dx
    # ====================================================================
    @staticmethod
    def sigmoid_dx(dx_km):
        """
        Transicao suave 0->1 centrada em dx_threshold_km, com largura
        dx_transition_km. Fallback ate que a fracao de area convectiva
        scale-aware real do Grell-Freitas esteja disponivel no pool de
        diagnostico do MONAN.
        """
        m = mpas_atmphys_lightning
        return 1.0 / (1.0 + np.exp(-(dx_km - m.dx_threshold_km) / m.dx_transition_km))

    # ====================================================================
    # flashrate_ep_column
    # ====================================================================
    @staticmethod
    def flashrate_ep_column(z_m, t_c, rho, w, qg, qs, qc):
        """
        Potencial de Eletrificacao (EP), portado do prototipo Python
        `ep_prototype.py` (ja validado nos 3 experimentos: reversao de
        sinal com LWC, curva Trev(EW), sensibilidade monotonica a
        w/qg). Substitui o McCaul como proxy fisico da branch de grade
        explicita (convection-permitting).

        Teoria (NIC graupel-cristal, Takahashi 1978 / Saunders & Peck
        1998): a taxa de separacao de carga depende da taxa de colisao
        (Ng*Ni*|Vg-Vi|*secao_choque), da eficiencia de colisao Eci(T)
        (pico em ~-15C), e do SINAL/magnitude da carga transferida, que
        depende de quao longe T esta da temperatura de reversao
        Trev(EW) -- que por sua vez desloca com o conteudo de agua
        liquida efetiva (aqui aproximado por qc).

        Unidades de entrada: qg, qs, qc em kg/kg (convertidas
        internamente para g/kg, como no prototipo Python). Saida: EP
        (magnitude, unidade arbitraria ate calibracao) e EP_signed
        (guardado para uso futuro em polaridade CG+/CG- e LNOx), alem
        do flash_rate = a_ep * EP^b_ep.

        Retorna
        -------
        (ep_mag, ep_signed, flash_rate)
        """
        m = mpas_atmphys_lightning
        nz = len(z_m)

        m_ice = (PI / 6.0) * m.rho_ice_ep * m.d0_ice_ep ** 3

        ep_mag = 0.0
        ep_signed = 0.0

        for k in range(1, nz):
            in_layer_k = (t_c[k] <= m.t_bot_ep) and (t_c[k] >= m.t_top_ep)
            in_layer_km1 = (t_c[k - 1] <= m.t_bot_ep) and (t_c[k - 1] >= m.t_top_ep)

            if in_layer_k or in_layer_km1:

                # --- nivel k -------------------------------------------
                ng, dg_char_k = m.graupel_size_ep(qg[k], rho[k])
                qs_gkg = qs[k] * 1000.0
                qc_gkg = qc[k] * 1000.0

                ni = rho[k] * (qs_gkg * 1.0e-3) / m_ice
                vg = m.fallspeed_graupel_ep(dg_char_k * 1000.0)
                dv = abs(vg - m.v_ice_ep)
                xsec_k = PI * (0.5 * dg_char_k + 0.5 * m.d0_ice_ep) ** 2
                ew = max(qc_gkg, 0.0) * rho[k]
                trev = m.reversal_temperature_ep(ew)
                eci = m.collision_efficiency_ep(t_c[k])
                dist_factor = np.tanh(abs(t_c[k] - trev) / 5.0)
                updraft_factor = 1.0 + max(w[k], 0.0) / 5.0

                dqdt_k = ng * ni * dv * xsec_k * eci * dist_factor * updraft_factor
                sign_k = -1.0 if t_c[k] > trev else 1.0

                # --- nivel k-1 ------------------------------------------
                ng, dg_char_km1 = m.graupel_size_ep(qg[k - 1], rho[k - 1])
                qs_gkg = qs[k - 1] * 1000.0
                qc_gkg = qc[k - 1] * 1000.0

                ni = rho[k - 1] * (qs_gkg * 1.0e-3) / m_ice
                vg = m.fallspeed_graupel_ep(dg_char_km1 * 1000.0)
                dv = abs(vg - m.v_ice_ep)
                xsec_km1 = PI * (0.5 * dg_char_km1 + 0.5 * m.d0_ice_ep) ** 2
                ew = max(qc_gkg, 0.0) * rho[k - 1]
                trev = m.reversal_temperature_ep(ew)
                eci = m.collision_efficiency_ep(t_c[k - 1])
                dist_factor = np.tanh(abs(t_c[k - 1] - trev) / 5.0)
                updraft_factor = 1.0 + max(w[k - 1], 0.0) / 5.0

                dqdt_km1 = ng * ni * dv * xsec_km1 * eci * dist_factor * updraft_factor
                sign_km1 = -1.0 if t_c[k - 1] > trev else 1.0

                dz = z_m[k] - z_m[k - 1]

                ep_mag = ep_mag + 0.5 * (abs(dqdt_k) + abs(dqdt_km1)) * dz
                ep_signed = ep_signed + 0.5 * (dqdt_k * sign_k + dqdt_km1 * sign_km1) * dz

        if ep_mag > 0.0:
            flash_rate = m.coef_a_ep * ep_mag ** m.coef_b_ep
        else:
            flash_rate = 0.0

        return ep_mag, ep_signed, flash_rate

    # ====================================================================
    # n0g_thompson
    # ====================================================================
    @staticmethod
    def n0g_thompson(qg_kgkg):
        """
        Parametro de interceptacao do graupel diagnosticado a partir de
        qg (kg/kg), exatamente como no Thompson scheme (curso MET-756-4,
        apostila "Gregory Thompson Scheme"):

            N0,g = max(1e4, min(200/qg, 5e6))  [m^-4]

        Correntes ascendentes fortes (mais qg) -> N0,g menor -> espectro
        desloca para particulas maiores -> queda mais rapida (sem
        precisar aumentar N0,g como um parametro fixo faria).
        """
        qg_safe = max(qg_kgkg, 1.0e-12)
        return max(1.0e4, min(200.0 / qg_safe, 5.0e6))

    # ====================================================================
    # graupel_size_ep
    # ====================================================================
    @staticmethod
    def graupel_size_ep(qg_kgkg, rho_k):
        """
        Concentracao numerica (Ng) e diametro caracteristico
        (Dg=1/lambda_g) do graupel, a partir do fechamento da
        distribuicao exponencial N(D)=N0,g*exp(-lambda_g*D) com N0,g
        diagnosticado por n0g_thompson(qg) e rho_graupel_ep=400 kg/m3
        (igual ao Thompson):

            lambda_g^4 = 6*pi*rho_graupel*N0,g / (rho_ar * qg)

        Retorna
        -------
        (ng, dg_char)
        """
        m = mpas_atmphys_lightning

        if qg_kgkg <= 1.0e-12:
            ng = 0.0
            dg_char = m.d0_graupel_fallback_ep  # sem massa de graupel ->
            # Ng=0 zera a colisao de qualquer forma
            return ng, dg_char

        n0g = m.n0g_thompson(qg_kgkg)
        lambda_g = (6.0 * PI * m.rho_graupel_ep * n0g / (rho_k * qg_kgkg)) ** 0.25
        ng = n0g / lambda_g
        dg_char = 1.0 / lambda_g
        return ng, dg_char

    # ====================================================================
    # fallspeed_graupel_ep
    # ====================================================================
    @staticmethod
    def fallspeed_graupel_ep(dg_mm):
        """
        Velocidade terminal do graupel (m/s), crescendo com o diametro
        caracteristico Dg (mm) -- derivado fisicamente de N0,g(qg)
        (graupel_size_ep) em vez de um diametro fixo assumido. Forma de
        lei de potencia, calibrar depois.
        """
        return 3.0 * max(dg_mm, 1.0e-3) ** 0.15 + 1.5

    # ====================================================================
    # collision_efficiency_ep
    # ====================================================================
    @staticmethod
    def collision_efficiency_ep(t_c_k):
        """
        Eficiencia de colisao/riming graupel-cristal, pico proximo a
        -15C (Takahashi 1978), decaindo para os dois lados.
        """
        return np.exp(-0.5 * ((t_c_k + 15.0) / 8.0) ** 2)

    # ====================================================================
    # reversal_temperature_ep
    # ====================================================================
    @staticmethod
    def reversal_temperature_ep(ew_gm3):
        """
        Temperatura de reversao de sinal de carga (C) em funcao do
        conteudo de agua liquida efetivo (EW, g/m3), forma simplificada
        inspirada em Saunders & Peck (1998). Clampada em [-20, -3] C
        (faixa fisica tipica).
        """
        trev = -15.0 + 6.0 * ew_gm3
        return min(max(trev, -20.0), -3.0)

    # ====================================================================
    # flashrate_mccaul_column
    # ====================================================================
    @staticmethod
    def flashrate_mccaul_column(z_m, t_c, rho, w, qg, qs):
        """
        Forma LITERAL de McCaul et al. (2009), eq. F1 -- descoberta ao
        ler o paper original (curso MET-756-4): a relacao publicada e
        LINEAR (FR = k1*(w*qg)), nao lei de potencia, e o fluxo de
        graupel e avaliado num UNICO NIVEL, exatamente a -15C (nao
        integrado em -5/-40C como uma versao anterior deste modulo
        fazia -- aquela era uma extensao fisica propria, nao o McCaul
        original). Mantida publica so para comparacao/diagnostico; o
        driver usa o EP.
        """
        m = mpas_atmphys_lightning
        nz = len(z_m)

        # localiza o par de niveis que envolve T=-15C (busca a mudanca
        # de sinal de (T - (-15)) entre niveis consecutivos)
        found = False
        k_lo = 0
        k = 0
        for kk in range(1, nz):
            if (t_c[kk] - m.t_peak_nic) * (t_c[kk - 1] - m.t_peak_nic) <= 0.0:
                k_lo = kk - 1
                k = kk
                found = True
                break

        if found:
            if abs(t_c[k] - t_c[k_lo]) > 1.0e-6:
                frac = (m.t_peak_nic - t_c[k_lo]) / (t_c[k] - t_c[k_lo])
            else:
                frac = 0.5
            frac = min(max(frac, 0.0), 1.0)

            w_m15 = w[k_lo] + frac * (w[k] - w[k_lo])
            qgs_m15 = (qg[k_lo] + qs[k_lo]) + frac * ((qg[k] + qs[k]) - (qg[k_lo] + qs[k_lo]))
        else:
            # coluna nao atravessa -15C (rasa/quente ou muito fria em
            # toda a extensao) -- usa o nivel mais proximo do alvo como
            # aproximacao
            k_lo = 0
            for kk in range(nz):
                if abs(t_c[kk] - m.t_peak_nic) < abs(t_c[k_lo] - m.t_peak_nic):
                    k_lo = kk
            w_m15 = w[k_lo]
            qgs_m15 = qg[k_lo] + qs[k_lo]

        # graupel flux = w * qg no nivel de -15C, qg convertido para g/kg
        # (convencao do paper original -- fluxos reportados na ordem de
        # centenas, ex. "graupel flux of 400 m s-1", consistente com qg
        # em g/kg e nao em kg/kg)
        graupel_flux_gkg = max(w_m15, 0.0) * qgs_m15 * 1000.0

        return m.coef_a_mccaul * graupel_flux_gkg

    # ====================================================================
    # flashrate_pricerind
    # ====================================================================
    @staticmethod
    def flashrate_pricerind(h_km):
        """
        FR_total = alpha * H^beta, H em km (topo convectivo).
        Coeficientes recalibrados com Barthe et al. (2010) -- retorna o
        flash rate TOTAL (IC+CG), nao apenas CG (ver nota em
        driver_lightning sobre a correcao desta semantica).
        """
        m = mpas_atmphys_lightning
        if h_km > 0.0:
            return m.coef_alpha_pr * h_km ** m.coef_beta_pr
        return 0.0

    # ====================================================================
    # calc_ic_cg_ratio
    # ====================================================================
    @staticmethod
    def calc_ic_cg_ratio(h_km):
        """
        Razao IC/CG em funcao da altura do topo (Price & Rind 1993),
        valido aproximadamente para H entre 5.5 e 14 km; fora disso,
        clampado.
        """
        h = min(max(h_km, 5.5), 14.0)
        ratio = (0.021 * h ** 4 - 0.648 * h ** 3 + 7.49 * h ** 2
                 - 36.54 * h + 63.09)
        return max(ratio, 0.1)  # evita razao negativa/instavel nos extremos
