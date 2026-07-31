# -*- coding: utf-8 -*-
"""
coluna_step2.py
=================

Modelo de COLUNA 1D (vertical) para o PASSO 2: adiciona a categoria de
GELO DE NUVEM (qi, Ni) ao esquema de chuva quente do Passo 1.

Esta classe HERDA de `ColunaChuvaQuente` (Passo 1) e:
  1. Adiciona os arrays de estado `qi`, `Ni`;
  2. SOBRESCREVE `_passo_processos_locais` para incluir, na ORDEM
     correta (ver docstring de `processos_fase_gelo.py` sobre o efeito
     Wegener-Bergeron-Findeisen), os processos:

         Pidsn -> Pidep -> Pccnd -> Pifzc -> Pimlt -> (processos de
         chuva quente do Passo 1: Pccnr, Pracw, Pr_self, Prevp)

  3. Adiciona sedimentacao (queda lenta) do gelo de nuvem.

ORDEM DOS PROCESSOS E O EFEITO WBF (Wegener-Bergeron-Findeisen)
-----------------------------------------------------------------
A ordem Pidsn -> Pidep -> Pccnd nao e arbitraria: ela e o que permite
o modelo reproduzir o efeito WBF de forma simples (ver docstring
completa em `processos_fase_gelo.py`). Resolvendo Pidep ANTES de
Pccnd, o vapor consumido pelo crescimento do gelo ja esta refletido em
qv quando calculamos a condensacao/evaporacao da agua liquida -- se
isso empurra qv para abaixo da saturacao liquida, Pccnd literalmente
evapora agua de nuvem para compensar, exatamente o comportamento
fisico esperado em nuvens de fase mista.
"""

import numpy as np

from .constantes import Rd, g, cp, Lv, Ls, rho_w, rho_i, MU_ICE, QMIN, NMIN, gamma_func
from .coluna_step1 import ColunaChuvaQuente
from .processos_chuva_quente import (
    Pccnd, Pccnr, Pracw, Pr_self, Prevp,
    velocidade_terminal_chuva, razao_mistura_saturacao,
)
from .processos_fase_gelo import (
    Pidsn, Pidep, Pifzc, Pimlt,
    razao_mistura_saturacao_gelo, velocidade_terminal_gelo,
)
from .distribuicoes import lambda_gama, N0_gama


class ColunaFaseGelo(ColunaChuvaQuente):
    """
    Coluna atmosferica 1D com microfisica de chuva quente (Passo 1) +
    gelo de nuvem (Passo 2): variaveis adicionais qi, Ni.
    """

    def __init__(self, nz=40, dz=100.0, T_base=293.0, p_base=95000.0,
                 lapse_rate=6.5e-3):
        super().__init__(nz=nz, dz=dz, T_base=T_base, p_base=p_base,
                          lapse_rate=lapse_rate)
        self.qi = np.zeros(nz)
        self.Ni = np.zeros(nz)

        self.historico["qi"] = []
        self.historico["Ni"] = []

    def inserir_gelo(self, k_base, k_topo, qi_valor=1.0e-4, Ni_valor=1.0e6):
        """Insere artificialmente uma camada de gelo de nuvem (util para
        testes controlados; em geral o gelo deve nuclear sozinho via
        Pidsn quando T < -5 graus C)."""
        self.qi[k_base:k_topo + 1] = qi_valor
        self.Ni[k_base:k_topo + 1] = Ni_valor

    # -------------------------------------------------------------
    # PASSO DE PROCESSOS LOCAIS (sobrescreve o do Passo 1)
    # -------------------------------------------------------------
    def _passo_processos_locais(self, dt):
        for k in range(self.nz):
            T, p, rho = self.T[k], self.p[k], self.rho[k]
            qv, qc, Nc = self.qv[k], self.qc[k], self.Nc[k]
            qr, Nr = self.qr[k], self.Nr[k]
            qi, Ni = self.qi[k], self.Ni[k]

            # ============================================================
            # ---- Pidsn: nucleacao primaria de gelo (Cooper 1986) ----
            # ============================================================
            dqi_idsn, dNi_idsn = Pidsn(Ni, T, rho, dt)
            dqi_idsn *= dt
            dNi_idsn *= dt
            qi_new = qi + dqi_idsn
            Ni_new = Ni + dNi_idsn
            qv_new = qv - dqi_idsn  # vapor consumido pela nucleacao (efeito pequeno)
            T_new = T + (Ls / cp) * dqi_idsn  # libera calor latente de deposicao

            # ============================================================
            # ---- Pidep: deposicao/sublimacao sobre gelo existente ----
            # ============================================================
            if Ni_new > NMIN:
                dqi_idep_bruto = Pidep(qv_new, qi_new, Ni_new, T_new, p, rho) * dt
                # limite: sublimacao nao pode remover mais gelo do que existe
                dqi_idep = max(dqi_idep_bruto, -qi_new)
            else:
                dqi_idep = 0.0

            qi_new = qi_new + dqi_idep
            qv_new = qv_new - dqi_idep  # vapor perdido/ganho e o oposto de qi
            T_new = T_new + (Ls / cp) * dqi_idep  # calor latente de sublimacao/deposicao

            # ============================================================
            # ---- Pccnd: condensacao/evaporacao de agua de nuvem ----
            # (USA O qv JA ATUALIZADO pelo gelo -- e isso que produz o
            #  efeito Wegener-Bergeron-Findeisen, ver docstring do modulo)
            # ============================================================
            dqc_ccnd, dqv_ccnd, dT_ccnd = Pccnd(qv_new, qc, T_new, p, dt)
            qc_new = qc + dqc_ccnd
            qv_new = qv_new + dqv_ccnd
            T_new = T_new + dT_ccnd
            Nc_new = Nc
            if dqc_ccnd > 0 and Nc_new <= NMIN:
                Nc_new = 1.0e8  # nucleacao simplificada de goticulas (como no Passo 1)

            # ============================================================
            # ---- Pifzc: congelamento heterogeneo/homogeneo qc -> qi ----
            # ============================================================
            dqc_ifzc, dNc_ifzc = Pifzc(qc_new, Nc_new, T_new, dt)
            dqc_ifzc = dqc_ifzc * dt
            dNc_ifzc = dNc_ifzc * dt
            dqc_ifzc = max(dqc_ifzc, -qc_new)
            dNc_ifzc = max(dNc_ifzc, -Nc_new)

            qc_new = qc_new + dqc_ifzc
            Nc_new = Nc_new + dNc_ifzc
            qi_new = qi_new + (-dqc_ifzc)   # massa congelada vira gelo
            Ni_new = Ni_new + (-dNc_ifzc)   # numero de cristais formados
            T_new = T_new + (Ls - Lv) / cp * (-dqc_ifzc)  # calor latente de fusao liberado

            # ============================================================
            # ---- Pimlt: degelo do gelo de nuvem (T > 0 graus C) ----
            # ============================================================
            dqi_imlt, dNi_imlt = Pimlt(qi_new, Ni_new, T_new, dt)
            dqi_imlt *= dt
            dNi_imlt *= dt
            dqi_imlt = max(dqi_imlt, -qi_new)
            dNi_imlt = max(dNi_imlt, -Ni_new)

            qi_new = qi_new + dqi_imlt
            Ni_new = Ni_new + dNi_imlt
            qc_new = qc_new + (-dqi_imlt)
            Nc_new = Nc_new + (-dNi_imlt)
            T_new = T_new - (Ls - Lv) / cp * (-dqi_imlt)  # consome calor latente de fusao

            # ============================================================
            # ---- Processos de chuva quente do Passo 1 (Pccnr, Pracw,
            #      Pr_self, Prevp) -- exatamente como em coluna_step1.py
            # ============================================================
            dqc_ccnr, dNc_ccnr = Pccnr(qc_new, Nc_new, rho)
            dqc_ccnr *= dt
            dNc_ccnr *= dt
            dqc_ccnr = max(dqc_ccnr, -qc_new)

            dqc_racw = Pracw(qc_new, qr) * dt
            dqc_racw = max(dqc_racw, -(qc_new + dqc_ccnr))

            transferencia_massa = -(dqc_ccnr + dqc_racw)
            qc_new = qc_new + dqc_ccnr + dqc_racw
            qr_new = qr + transferencia_massa
            Nr_new = Nr + (-dNc_ccnr)
            Nc_new = Nc_new + dNc_ccnr

            dNr_self = Pr_self(qr_new, Nr_new, rho) * dt
            Nr_new = max(Nr_new + dNr_self, 0.0)

            dqr_revp_bruto, dNr_revp_bruto = Prevp(qr_new, Nr_new, qv_new, T_new, p, rho)
            dqr_revp_bruto *= dt
            dNr_revp_bruto *= dt
            dqr_revp = max(dqr_revp_bruto, -qr_new)
            dNr_revp = max(dNr_revp_bruto, -Nr_new)
            qr_new = qr_new + dqr_revp
            Nr_new = Nr_new + dNr_revp
            qv_new = qv_new - dqr_revp
            T_new = T_new + (Lv / cp) * dqr_revp

            # ---- limpeza de valores residuais despreziveis ----
            if qc_new < QMIN:
                qc_new, Nc_new = 0.0, 0.0
            if qr_new < QMIN:
                qr_new, Nr_new = 0.0, 0.0
            if qi_new < QMIN:
                qi_new, Ni_new = 0.0, 0.0

            self.qv[k], self.qc[k], self.Nc[k] = qv_new, qc_new, Nc_new
            self.qr[k], self.Nr[k] = qr_new, Nr_new
            self.qi[k], self.Ni[k] = qi_new, Ni_new
            self.T[k] = T_new

    # -------------------------------------------------------------
    # SEDIMENTACAO (sobrescreve para incluir tambem qi, Ni)
    # -------------------------------------------------------------
    def _passo_sedimentacao(self, dt):
        precip_chuva_mm = super()._passo_sedimentacao(dt)

        Vq = np.zeros(self.nz)
        Vn = np.zeros(self.nz)
        for k in range(self.nz):
            Vq[k], Vn[k] = velocidade_terminal_gelo(self.qi[k], self.Ni[k], self.rho[k])

        Vmax = max(Vq.max(), Vn.max(), 1.0e-6)
        n_sub = int(np.ceil(Vmax * dt / self.dz)) + 1
        dt_sub = dt / n_sub

        for _ in range(n_sub):
            fluxo_q = self.rho * self.qi * Vq
            fluxo_n = self.rho * self.Ni * Vn

            dqi = np.zeros(self.nz)
            dNi = np.zeros(self.nz)
            for k in range(self.nz):
                fluxo_entra_q = fluxo_q[k + 1] if k < self.nz - 1 else 0.0
                fluxo_entra_n = fluxo_n[k + 1] if k < self.nz - 1 else 0.0
                dqi[k] = (fluxo_entra_q - fluxo_q[k]) / (self.rho[k] * self.dz) * dt_sub
                dNi[k] = (fluxo_entra_n - fluxo_n[k]) / (self.rho[k] * self.dz) * dt_sub

            self.qi = np.maximum(self.qi + dqi, 0.0)
            self.Ni = np.maximum(self.Ni + dNi, 0.0)

            for k in range(self.nz):
                Vq[k], Vn[k] = velocidade_terminal_gelo(self.qi[k], self.Ni[k], self.rho[k])

        return precip_chuva_mm  # gelo nao conta como "precipitacao liquida" aqui

    # -------------------------------------------------------------
    # LOOP PRINCIPAL (sobrescreve para tambem salvar qi, Ni no historico)
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
                proxima_gravacao += salvar_a_cada

        return self.historico
