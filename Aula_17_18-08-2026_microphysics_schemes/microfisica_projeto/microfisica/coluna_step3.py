# -*- coding: utf-8 -*-
"""
coluna_step3.py
=================

Modelo de COLUNA 1D (vertical) para o PASSO 3: adiciona NEVE (qs, Ns) e
GRAUPEL (qg, Ng) ao modelo do Passo 2, com todas as interacoes de fase
mista (riming, congelamento de chuva, coleta cruzada, Hallett-Mossop,
degelo).

Esta classe HERDA de `ColunaFaseGelo` (Passo 2) e sobrescreve
`_passo_processos_locais` para incluir, na ORDEM abaixo, todos os
processos. A ordem generaliza a logica de Wegener-Bergeron-Findeisen ja
usada no Passo 2 (crescimento por deposicao de TODAS as categorias de
gelo primeiro, condensacao/evaporacao liquida depois) e adiciona as
interacoes de coleta (riming) e congelamento:

    1. Pidsn                          (nucleacao primaria de gelo)
    2. Pidep, Psdep, Pgdep             (deposicao/sublimacao -- gelo, neve, graupel)
    3. Pccnd                          (condensacao/evaporacao liquida -- efeito WBF)
    4. Pifzc, Pgfzr                    (congelamento heterogeneo/homogeneo: qc->qi, qr->qg)
    5. Pi_iacw, Ps_sacw, Pgacw          (riming: gelo/neve/graupel coletam agua de nuvem)
    6. Pispl                          (Hallett-Mossop, usa a soma dos ritmos de riming)
    7. Ps_sacr, Pgacr, Piacr            (neve/graupel/gelo coletam chuva)
    8. Picns                          (autoconversao gelo -> neve)
    9. Pimlt, Psmlt, Pgmlt              (degelo -- gelo->nuvem; neve,graupel->chuva)
   10. Pccnr, Pracw, Pr_self, Prevp      (processos de chuva quente do Passo 1)

Ver `processos_fase_mista.py` para a lista de simplificacoes
deliberadas deste passo (Picng, Pscng, Pg_racs NAO implementados).
"""

import numpy as np

from .constantes import (Rd, g, cp, Lv, Ls, rho_w, rho_i, rho_s, rho_g,
                          MU_ICE, MU_SNOW, QMIN, NMIN, gamma_func)
from .coluna_step2 import ColunaFaseGelo
from .processos_chuva_quente import (
    Pccnd, Pccnr, Pracw, Pr_self, Prevp,
    velocidade_terminal_chuva,
)
from .processos_fase_gelo import (
    Pidsn, Pidep, Pifzc, Pimlt, Pi_iacw,
    velocidade_terminal_gelo,
)
from .processos_fase_mista import (
    Picns, Psdep, Pgdep, Pgfzr, Pmlt, Pispl, colecao_continua,
    velocidade_terminal_neve, velocidade_terminal_graupel,
)
from .distribuicoes import lambda_gama, diametro_medio_numero


class ColunaFaseMista(ColunaFaseGelo):
    """
    Coluna atmosferica 1D com microfisica completa de fase mista:
    Passo 1 (qc,Nc,qr,Nr) + Passo 2 (qi,Ni) + Passo 3 (qs,Ns,qg,Ng).
    """

    def __init__(self, nz=40, dz=100.0, T_base=293.0, p_base=95000.0,
                 lapse_rate=6.5e-3):
        super().__init__(nz=nz, dz=dz, T_base=T_base, p_base=p_base,
                          lapse_rate=lapse_rate)
        self.qs = np.zeros(nz)
        self.Ns = np.zeros(nz)
        self.qg = np.zeros(nz)
        self.Ng = np.zeros(nz)

        self.historico["qs"] = []
        self.historico["Ns"] = []
        self.historico["qg"] = []
        self.historico["Ng"] = []

    def inserir_neve(self, k_base, k_topo, qs_valor=1.0e-4, Ns_valor=1.0e4):
        self.qs[k_base:k_topo + 1] = qs_valor
        self.Ns[k_base:k_topo + 1] = Ns_valor

    def inserir_graupel(self, k_base, k_topo, qg_valor=1.0e-4, Ng_valor=1.0e3):
        self.qg[k_base:k_topo + 1] = qg_valor
        self.Ng[k_base:k_topo + 1] = Ng_valor

    # -------------------------------------------------------------
    # PASSO DE PROCESSOS LOCAIS (sobrescreve o do Passo 2)
    # -------------------------------------------------------------
    def _passo_processos_locais(self, dt):
        for k in range(self.nz):
            T, p, rho = self.T[k], self.p[k], self.rho[k]
            qv, qc, Nc = self.qv[k], self.qc[k], self.Nc[k]
            qr, Nr = self.qr[k], self.Nr[k]
            qi, Ni = self.qi[k], self.Ni[k]
            qs, Ns = self.qs[k], self.Ns[k]
            qg, Ng = self.qg[k], self.Ng[k]

            # ===== 1) Pidsn: nucleacao primaria de gelo =====
            dqi_idsn, dNi_idsn = Pidsn(Ni, T, rho, dt)
            dqi_idsn *= dt; dNi_idsn *= dt
            qi_new = qi + dqi_idsn
            Ni_new = Ni + dNi_idsn
            qv_new = qv - dqi_idsn
            T_new = T + (Ls / cp) * dqi_idsn

            # ===== 2) Pidep, Psdep, Pgdep: deposicao/sublimacao =====
            dqi_idep = 0.0
            if Ni_new > NMIN:
                dqi_idep = max(Pidep(qv_new, qi_new, Ni_new, T_new, p, rho) * dt, -qi_new)
            qi_new += dqi_idep
            qv_new -= dqi_idep
            T_new += (Ls / cp) * dqi_idep

            dqs_sdep = 0.0
            qs_new, Ns_new = qs, Ns
            if Ns_new > NMIN:
                dqs_sdep = max(Psdep(qv_new, qs_new, Ns_new, T_new, p, rho) * dt, -qs_new)
            qs_new += dqs_sdep
            qv_new -= dqs_sdep
            T_new += (Ls / cp) * dqs_sdep

            dqg_gdep = 0.0
            qg_new, Ng_new = qg, Ng
            if Ng_new > NMIN:
                dqg_gdep = max(Pgdep(qv_new, qg_new, Ng_new, T_new, p, rho) * dt, -qg_new)
            qg_new += dqg_gdep
            qv_new -= dqg_gdep
            T_new += (Ls / cp) * dqg_gdep

            # ===== 3) Pccnd: condensacao/evaporacao liquida (efeito WBF) =====
            dqc_ccnd, dqv_ccnd, dT_ccnd = Pccnd(qv_new, qc, T_new, p, dt)
            qc_new = qc + dqc_ccnd
            qv_new = qv_new + dqv_ccnd
            T_new = T_new + dT_ccnd
            Nc_new = Nc
            if dqc_ccnd > 0 and Nc_new <= NMIN:
                Nc_new = 1.0e8

            # ===== 4) Pifzc, Pgfzr: congelamento heterogeneo/homogeneo =====
            dqc_ifzc, dNc_ifzc = Pifzc(qc_new, Nc_new, T_new, dt)
            dqc_ifzc = max(dqc_ifzc * dt, -qc_new)
            dNc_ifzc = max(dNc_ifzc * dt, -Nc_new)
            qc_new += dqc_ifzc; Nc_new += dNc_ifzc
            qi_new += -dqc_ifzc; Ni_new += -dNc_ifzc
            T_new += (Ls - Lv) / cp * (-dqc_ifzc)

            dqr_gfzr, dNr_gfzr = Pgfzr(qr, Nr, T_new, dt)
            dqr_gfzr = max(dqr_gfzr * dt, -qr)
            dNr_gfzr = max(dNr_gfzr * dt, -Nr)
            qr_new = qr + dqr_gfzr
            Nr_new = Nr + dNr_gfzr
            qg_new += -dqr_gfzr
            Ng_new += -dNr_gfzr
            T_new += (Ls - Lv) / cp * (-dqr_gfzr)

            # ===== 5) Pi_iacw, Ps_sacw, Pgacw: riming (coletam agua de nuvem) =====
            Vi, _ = velocidade_terminal_gelo(qi_new, Ni_new, rho)
            Vs, _ = velocidade_terminal_neve(qs_new, Ns_new, rho)
            Vg, _ = velocidade_terminal_graupel(qg_new, Ng_new, rho)

            dqc_iiacw = max(Pi_iacw(qc_new, Ni_new, qi_new, T_new, rho) * dt, -qc_new)
            qc_new += dqc_iiacw
            qi_new += -dqc_iiacw
            T_new += (Ls - Lv) / cp * (-dqc_iiacw)

            D_s = diametro_medio_numero(lambda_gama(qs_new, Ns_new, rho, rho_s, MU_SNOW), MU_SNOW) if qs_new > QMIN and Ns_new > NMIN else 0.0
            dqc_ssacw = max(colecao_continua(qc_new, Ns_new, D_s, Vs, 0.0, 1.0) * dt, -qc_new) if Ns_new > NMIN else 0.0
            qc_new += dqc_ssacw
            qs_new += -dqc_ssacw
            T_new += (Ls - Lv) / cp * (-dqc_ssacw)

            D_g = diametro_medio_numero(lambda_gama(qg_new, Ng_new, rho, rho_g, MU_SNOW), MU_SNOW) if qg_new > QMIN and Ng_new > NMIN else 0.0
            dqc_gacw = max(colecao_continua(qc_new, Ng_new, D_g, Vg, 0.0, 1.0) * dt, -qc_new) if Ng_new > NMIN else 0.0
            qc_new += dqc_gacw
            qg_new += -dqc_gacw
            T_new += (Ls - Lv) / cp * (-dqc_gacw)

            # ===== 6) Pispl: Hallett-Mossop (usa o riming total deste passo) =====
            riming_total = (-dqc_iiacw - dqc_ssacw - dqc_gacw) / dt  # kg/kg/s, >=0
            dNi_ispl = Pispl(riming_total, T_new) * dt
            Ni_new += dNi_ispl  # cria novos cristais (sem consumir massa extra: simplificacao)

            # ===== 7) Ps_sacr, Pgacr, Piacr: coletam chuva =====
            Vr, _ = velocidade_terminal_chuva(qr_new, Nr_new, rho)

            dqr_ssacr = max(colecao_continua(qr_new, Ns_new, D_s, Vs, Vr, 1.0) * dt, -qr_new) if Ns_new > NMIN else 0.0
            qr_new += dqr_ssacr
            qs_new += -dqr_ssacr

            dqr_gacr = max(colecao_continua(qr_new, Ng_new, D_g, Vg, Vr, 1.0) * dt, -qr_new) if Ng_new > NMIN else 0.0
            qr_new += dqr_gacr
            qg_new += -dqr_gacr

            D_i = diametro_medio_numero(lambda_gama(qi_new, Ni_new, rho, rho_i, MU_ICE), MU_ICE) if qi_new > QMIN and Ni_new > NMIN else 0.0
            dqr_iacr = max(colecao_continua(qr_new, Ni_new, D_i, Vi, Vr, 1.0) * dt, -qr_new) if Ni_new > NMIN else 0.0
            qr_new += dqr_iacr
            qg_new += -dqr_iacr  # gelo coletando chuva forma graupel (massa e calor)
            T_new += (Ls - Lv) / cp * (-dqr_iacr)

            # ===== 8) Picns: autoconversao gelo -> neve =====
            dqi_icns, dNi_icns, dNs_icns = Picns(qi_new, Ni_new, T_new)
            dqi_icns = max(dqi_icns * dt, -qi_new)
            dNi_icns = max(dNi_icns * dt, -Ni_new)
            dNs_icns = dNs_icns * dt
            qi_new += dqi_icns
            Ni_new += dNi_icns
            qs_new += -dqi_icns
            Ns_new += dNs_icns

            # ===== 9) Pimlt, Psmlt, Pgmlt: degelo =====
            dqi_imlt, dNi_imlt = Pimlt(qi_new, Ni_new, T_new, dt)
            dqi_imlt = max(dqi_imlt * dt, -qi_new)
            dNi_imlt = max(dNi_imlt * dt, -Ni_new)
            qi_new += dqi_imlt; Ni_new += dNi_imlt
            qc_new += -dqi_imlt; Nc_new += -dNi_imlt
            T_new -= (Ls - Lv) / cp * (-dqi_imlt)

            dqs_smlt, dNs_smlt = Pmlt(qs_new, Ns_new, T_new, dt)
            dqs_smlt = max(dqs_smlt * dt, -qs_new)
            dNs_smlt = max(dNs_smlt * dt, -Ns_new)
            qs_new += dqs_smlt; Ns_new += dNs_smlt
            qr_new += -dqs_smlt; Nr_new += -dNs_smlt
            T_new -= (Ls - Lv) / cp * (-dqs_smlt)

            dqg_gmlt, dNg_gmlt = Pmlt(qg_new, Ng_new, T_new, dt)
            dqg_gmlt = max(dqg_gmlt * dt, -qg_new)
            dNg_gmlt = max(dNg_gmlt * dt, -Ng_new)
            qg_new += dqg_gmlt; Ng_new += dNg_gmlt
            qr_new += -dqg_gmlt; Nr_new += -dNg_gmlt
            T_new -= (Ls - Lv) / cp * (-dqg_gmlt)

            # ===== 10) Processos de chuva quente (Passo 1) =====
            dqc_ccnr, dNc_ccnr = Pccnr(qc_new, Nc_new, rho)
            dqc_ccnr *= dt; dNc_ccnr *= dt
            dqc_ccnr = max(dqc_ccnr, -qc_new)

            dqc_racw = max(Pracw(qc_new, qr_new) * dt, -(qc_new + dqc_ccnr))

            transferencia = -(dqc_ccnr + dqc_racw)
            qc_new = qc_new + dqc_ccnr + dqc_racw
            qr_new = qr_new + transferencia
            Nr_new = Nr_new + (-dNc_ccnr)
            Nc_new = Nc_new + dNc_ccnr

            dNr_self = Pr_self(qr_new, Nr_new, rho) * dt
            Nr_new = max(Nr_new + dNr_self, 0.0)

            dqr_revp_b, dNr_revp_b = Prevp(qr_new, Nr_new, qv_new, T_new, p, rho)
            dqr_revp = max(dqr_revp_b * dt, -qr_new)
            dNr_revp = max(dNr_revp_b * dt, -Nr_new)
            qr_new += dqr_revp
            Nr_new += dNr_revp
            qv_new -= dqr_revp
            T_new += (Lv / cp) * dqr_revp

            # ---- limpeza de valores residuais despreziveis ----
            if qc_new < QMIN: qc_new, Nc_new = 0.0, 0.0
            if qr_new < QMIN: qr_new, Nr_new = 0.0, 0.0
            if qi_new < QMIN: qi_new, Ni_new = 0.0, 0.0
            if qs_new < QMIN: qs_new, Ns_new = 0.0, 0.0
            if qg_new < QMIN: qg_new, Ng_new = 0.0, 0.0

            self.qv[k], self.qc[k], self.Nc[k] = qv_new, qc_new, Nc_new
            self.qr[k], self.Nr[k] = qr_new, Nr_new
            self.qi[k], self.Ni[k] = qi_new, Ni_new
            self.qs[k], self.Ns[k] = qs_new, Ns_new
            self.qg[k], self.Ng[k] = qg_new, Ng_new
            self.T[k] = T_new

    # -------------------------------------------------------------
    # SEDIMENTACAO (sobrescreve para incluir qs, Ns, qg, Ng)
    # -------------------------------------------------------------
    def _passo_sedimentacao(self, dt):
        precip_chuva_mm = super()._passo_sedimentacao(dt)  # ja cuida de qr e qi

        for campo_q, campo_n, vel_func, rho_x, nome in [
            ("qs", "Ns", velocidade_terminal_neve, rho_s, "neve"),
            ("qg", "Ng", velocidade_terminal_graupel, rho_g, "graupel"),
        ]:
            q_arr = getattr(self, campo_q)
            n_arr = getattr(self, campo_n)

            Vq = np.zeros(self.nz)
            Vn = np.zeros(self.nz)
            for k in range(self.nz):
                Vq[k], Vn[k] = vel_func(q_arr[k], n_arr[k], self.rho[k])

            Vmax = max(Vq.max(), Vn.max(), 1.0e-6)
            n_sub = int(np.ceil(Vmax * dt / self.dz)) + 1
            dt_sub = dt / n_sub

            for _ in range(n_sub):
                fluxo_q = self.rho * q_arr * Vq
                fluxo_n = self.rho * n_arr * Vn

                dq = np.zeros(self.nz)
                dn = np.zeros(self.nz)
                for k in range(self.nz):
                    fq_in = fluxo_q[k + 1] if k < self.nz - 1 else 0.0
                    fn_in = fluxo_n[k + 1] if k < self.nz - 1 else 0.0
                    dq[k] = (fq_in - fluxo_q[k]) / (self.rho[k] * self.dz) * dt_sub
                    dn[k] = (fn_in - fluxo_n[k]) / (self.rho[k] * self.dz) * dt_sub

                q_arr = np.maximum(q_arr + dq, 0.0)
                n_arr = np.maximum(n_arr + dn, 0.0)

                for k in range(self.nz):
                    Vq[k], Vn[k] = vel_func(q_arr[k], n_arr[k], self.rho[k])

            setattr(self, campo_q, q_arr)
            setattr(self, campo_n, n_arr)

        return precip_chuva_mm

    # -------------------------------------------------------------
    # LOOP PRINCIPAL (sobrescreve para tambem salvar qs,Ns,qg,Ng)
    # -------------------------------------------------------------
    def integrar(self, tempo_total_s, dt=5.0, salvar_a_cada=30.0):
        n_passos = int(tempo_total_s / dt)
        proxima_gravacao = 0.0
        self.precip_superficie_mm = 0.0

        for passo in range(n_passos):
            t = passo * dt
            self._passo_processos_locais(dt)
            precip = self._passo_sedimentacao(dt)
            self.precip_superficie_mm += precip

            if t >= proxima_gravacao:
                self.historico["t"].append(t)
                self.historico["qc"].append(self.qc.copy())
                self.historico["qr"].append(self.qr.copy())
                self.historico["Nc"].append(self.Nc.copy())
                self.historico["Nr"].append(self.Nr.copy())
                self.historico["qv"].append(self.qv.copy())
                self.historico["T"].append(self.T.copy())
                self.historico["qi"].append(self.qi.copy())
                self.historico["Ni"].append(self.Ni.copy())
                self.historico["qs"].append(self.qs.copy())
                self.historico["Ns"].append(self.Ns.copy())
                self.historico["qg"].append(self.qg.copy())
                self.historico["Ng"].append(self.Ng.copy())
                proxima_gravacao += salvar_a_cada

        return self.historico
