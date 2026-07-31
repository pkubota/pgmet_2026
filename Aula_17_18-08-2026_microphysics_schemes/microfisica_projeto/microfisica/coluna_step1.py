# -*- coding: utf-8 -*-
"""
coluna_step1.py
================

Modelo de COLUNA 1D (vertical) para o PASSO 1 (chuva quente).

OBJETIVO DIDATICO
------------------
Simular a evolucao temporal de uma camada de nuvem em uma coluna
atmosferica estatica (perfil de temperatura e pressao prescrito, sem
dinamica -- o foco e 100% na microfisica), permitindo observar:

    1. Condensacao do vapor em excesso -> formacao de agua de nuvem (qc)
    2. Autoconversao: qc -> qr  (formacao da "primeira chuva")
    3. Acrescimo: chuva coleta mais agua de nuvem, crescendo ainda mais
    4. Autocolecao: gotas de chuva se fundem (Nr cai, qr mantem-se)
    5. Sedimentacao: qr e Nr caem atraves dos niveis do modelo
    6. Evaporacao da chuva abaixo da base da nuvem (ar subsaturado)

ESTRUTURA NUMERICA
-------------------
- Grade vertical Arakawa tipo "Lorenz" simples: nz niveis igualmente
  espacados, com dz constante.
- Perfil termodinamico prescrito: atmosfera com lapse rate constante,
  hidrostatica.
- Splitting de processos (aproximacao padrao em microfisica "bulk"):
  em cada passo de tempo dt, primeiro resolvem-se os processos LOCAIS
  (condensacao, autoconversao, acrescimo, autocolecao, evaporacao) em
  cada nivel de forma independente, e depois aplica-se a SEDIMENTACAO
  (processo verticalmente acoplado) com um esquema upwind explicito,
  sub-passado (substepping) para garantir estabilidade (criterio CFL).
"""

import numpy as np

from .constantes import Rd, g, cp, rho_w, QMIN, NMIN
from .processos_chuva_quente import (
    Pccnd,
    Pccnr,
    Pracw,
    Pr_self,
    Prevp,
    velocidade_terminal_chuva,
    razao_mistura_saturacao,
)


class ColunaChuvaQuente:
    """
    Coluna atmosferica 1D com microfisica de chuva quente (Passo 1).

    Variaveis de estado, um valor por nivel vertical (arrays de shape (nz,)):
        T   : temperatura (K)
        p   : pressao (Pa)
        rho : densidade do ar (kg/m^3)
        qv  : razao de mistura de vapor d'agua (kg/kg)
        qc  : razao de mistura de agua de nuvem (kg/kg)
        Nc  : concentracao numerica de goticulas de nuvem (kg^-1)
        qr  : razao de mistura de agua de chuva (kg/kg)
        Nr  : concentracao numerica de gotas de chuva (kg^-1)
    """

    def __init__(self, nz=40, dz=100.0, T_base=293.0, p_base=95000.0,
                 lapse_rate=6.5e-3):
        self.nz = nz
        self.dz = dz
        self.z = np.arange(nz) * dz  # m, nivel 0 = superficie

        # --- perfil termodinamico prescrito (hidrostatico, lapse rate constante) ---
        self.T = T_base - lapse_rate * self.z
        # integracao hidrostatica simples: dp/dz = -rho*g = -p/(Rd*T) * g
        self.p = np.zeros(nz)
        self.p[0] = p_base
        for k in range(1, nz):
            rho_k = self.p[k - 1] / (Rd * self.T[k - 1])
            self.p[k] = self.p[k - 1] - rho_k * g * dz
        self.rho = self.p / (Rd * self.T)

        # --- variaveis de agua (iniciam em zero / ambiente subsaturado) ---
        self.qv = 0.6 * razao_mistura_saturacao(self.T, self.p)  # 60% UR ambiente
        self.qc = np.zeros(nz)
        self.Nc = np.zeros(nz)
        self.qr = np.zeros(nz)
        self.Nr = np.zeros(nz)

        # historico para diagnosticos/plots
        self.historico = {"t": [], "qc": [], "qr": [], "Nc": [], "Nr": [],
                           "qv": [], "T": []}

    def inserir_nuvem(self, k_base, k_topo, qc_valor=1.0e-3, Nc_valor=1.0e8):
        """
        Insere artificialmente uma camada de agua de nuvem entre os
        niveis k_base e k_topo (inclusive), simulando uma nuvem ja
        formada por ascensao adiabatica previa (simplificacao: pulamos
        a etapa de ativacao de CCN, que e tratada como um valor tipico
        de Nc = 10^8 kg^-1 ~ 100 gotas/cm^3, valor continental tipico).
        """
        self.qc[k_base:k_topo + 1] = qc_valor
        self.Nc[k_base:k_topo + 1] = Nc_valor
        # ajusta o vapor para saturacao nesses niveis (consistencia fisica)
        self.qv[k_base:k_topo + 1] = razao_mistura_saturacao(
            self.T[k_base:k_topo + 1], self.p[k_base:k_topo + 1])

    # -------------------------------------------------------------
    # PASSO DE PROCESSOS LOCAIS (ponto a ponto, sem trocas entre niveis)
    # -------------------------------------------------------------
    def _passo_processos_locais(self, dt):
        """
        Resolve, em cada nivel k, todos os processos "locais" (que nao
        envolvem troca com outros niveis) usando a notacao de processos
        do curso (slides MET-756-4, esquema de Thompson): Pccnd, Pccnr,
        Pracw, Prevp. O termo Pr_self (autocolecao da chuva) e uma
        extensao de 2 momentos deste projeto, sem simbolo P
        correspondente no esquema oficial de 1 momento -- ver docstring
        do modulo `processos_chuva_quente.py`.

        Correspondencia com as equacoes do curso (Parte IV, slides):

            (dp*qv/dt) contem o termo p*( -Pccnd + Prevp + ... )
            (dp*qc/dt) contem o termo p*( +Pccnd - Pccnr - Pracw + ... )
            (dp*qr/dt) contem o termo p*( +Pccnr + Pracw - Prevp + ... ) - Prprc

        (os "..." referem-se aos termos de fase gelo/mista, ainda nao
        implementados -- Passos 2 e 3.)
        """
        for k in range(self.nz):
            T, p, rho = self.T[k], self.p[k], self.rho[k]
            qv, qc, Nc, qr, Nr = self.qv[k], self.qc[k], self.Nc[k], self.qr[k], self.Nr[k]

            # ---- Pccnd: condensacao/evaporacao de agua de nuvem ----
            dqc_ccnd, dqv_ccnd, dT_ccnd = Pccnd(qv, qc, T, p, dt)
            qc_new = qc + dqc_ccnd
            qv_new = qv + dqv_ccnd
            T_new = T + dT_ccnd
            # se condensou massa nova, cria numero de goticulas
            # (nucleacao simplificada: assume-se Nc de fundo ja ativado,
            # 10^8 kg^-1, se ainda nao havia goticulas)
            Nc_new = Nc
            if dqc_ccnd > 0 and Nc_new <= NMIN:
                Nc_new = 1.0e8

            # ---- Pccnr: autoconversao (qc,Nc -> qr,Nr) ----
            dqc_ccnr, dNc_ccnr = Pccnr(qc_new, Nc_new, rho)
            dqc_ccnr *= dt
            dNc_ccnr *= dt
            dqc_ccnr = max(dqc_ccnr, -qc_new)  # nao pode remover mais que existe

            # ---- Pracw: acrescimo (chuva coleta nuvem) ----
            dqc_racw = Pracw(qc_new, qr) * dt
            dqc_racw = max(dqc_racw, -(qc_new + dqc_ccnr))

            # atualiza agua de nuvem e chuva (Pccnr + Pracw)
            transferencia_massa = -(dqc_ccnr + dqc_racw)  # >=0, massa que vai para qr
            qc_new = qc_new + dqc_ccnr + dqc_racw
            qr_new = qr + transferencia_massa
            Nr_new = Nr + (-dNc_ccnr)  # numero de novas gotas de chuva formadas
            Nc_new = Nc_new + dNc_ccnr  # (dNc_ccnr <= 0)

            # ---- Pr_self: autocolecao da chuva [extensao 2 momentos] ----
            dNr_self = Pr_self(qr_new, Nr_new, rho) * dt
            Nr_new = max(Nr_new + dNr_self, 0.0)

            # ---- Prevp: evaporacao da chuva (se subsaturado) ----
            #
            # IMPORTANTE (conservacao de massa): a taxa retornada e um
            # limite superior "fisico" do processo, mas pode remover, em
            # um unico passo de tempo dt, mais massa do que qr_new
            # realmente possui. Por isso aplicamos o limite (clipping)
            # APOS multiplicar pela dt, e usamos o MESMO valor ja limitado
            # tanto para reduzir qr quanto para aumentar qv e ajustar T.
            # Se cada variavel fosse limitada separadamente (ex.: qr
            # limitado, mas qv usando o valor bruto nao limitado), o
            # balanco de agua total (qv+qc+qr) deixaria de ser conservado
            # -- exatamente o bug que causava crescimento espurio e
            # nao-fisico de precipitacao neste modelo durante os testes
            # de validacao (ver nota detalhada no docstring de `Prevp`).
            dqr_revp_bruto, dNr_revp_bruto = Prevp(qr_new, Nr_new, qv_new, T_new, p, rho)
            dqr_revp_bruto *= dt
            dNr_revp_bruto *= dt

            dqr_revp = max(dqr_revp_bruto, -qr_new)     # <= 0, nao remove mais que qr_new
            dNr_revp = max(dNr_revp_bruto, -Nr_new)      # <= 0, nao remove mais que Nr_new

            qr_new = qr_new + dqr_revp
            Nr_new = Nr_new + dNr_revp
            qv_new = qv_new - dqr_revp  # vapor aumenta quando chuva evapora (dqr_revp<0)
            T_new = T_new + (2.501e6 / cp) * dqr_revp  # resfriamento latente da evaporacao

            # limpeza de valores residuais despreziveis
            if qc_new < QMIN:
                qc_new, Nc_new = 0.0, 0.0
            if qr_new < QMIN:
                qr_new, Nr_new = 0.0, 0.0

            self.qv[k], self.qc[k], self.Nc[k] = qv_new, qc_new, Nc_new
            self.qr[k], self.Nr[k] = qr_new, Nr_new
            self.T[k] = T_new

    # -------------------------------------------------------------
    # PASSO DE SEDIMENTACAO (acopla os niveis verticalmente)
    # -------------------------------------------------------------
    def _passo_sedimentacao(self, dt):
        """
        Sedimentacao explicita (upwind) de qr e Nr -- os termos "Prprc"
        (fluxo de precipitacao de massa) e "Nrprc" (fluxo de numero,
        extensao de 2 momentos) nas equacoes do curso -- com
        substepping para satisfazer o criterio de estabilidade CFL:
        dt_sub <= dz / V_max.

        A massa/numero que sai do nivel k entra no nivel k-1 (queda),
        e o nivel mais baixo (k=0, "superficie") perde a precipitacao
        que se acumula como chuva na superficie (diagnostico de
        precipitacao acumulada).
        """
        Vq = np.zeros(self.nz)
        Vn = np.zeros(self.nz)
        for k in range(self.nz):
            Vq[k], Vn[k] = velocidade_terminal_chuva(self.qr[k], self.Nr[k], self.rho[k])

        Vmax = max(Vq.max(), Vn.max(), 1.0e-6)
        n_sub = int(np.ceil(Vmax * dt / self.dz)) + 1
        dt_sub = dt / n_sub

        precip_acumulada = 0.0
        for _ in range(n_sub):
            fluxo_q = self.rho * self.qr * Vq   # kg m^-2 s^-1 (fluxo de massa para baixo)
            fluxo_n = self.rho * self.Nr * Vn   # m^-2 s^-1

            dqr = np.zeros(self.nz)
            dNr = np.zeros(self.nz)
            for k in range(self.nz):
                fluxo_entra_q = fluxo_q[k + 1] if k < self.nz - 1 else 0.0
                fluxo_entra_n = fluxo_n[k + 1] if k < self.nz - 1 else 0.0
                dqr[k] = (fluxo_entra_q - fluxo_q[k]) / (self.rho[k] * self.dz) * dt_sub
                dNr[k] = (fluxo_entra_n - fluxo_n[k]) / (self.rho[k] * self.dz) * dt_sub

            self.qr = np.maximum(self.qr + dqr, 0.0)
            self.Nr = np.maximum(self.Nr + dNr, 0.0)

            # precipitacao que sai pela base (k=0) vira precipitacao acumulada (mm)
            precip_acumulada += fluxo_q[0] * dt_sub / rho_w * 1000.0  # mm de lamina d'agua

            # recalcula velocidades para o proximo subpasso (elas mudam com qr,Nr)
            for k in range(self.nz):
                Vq[k], Vn[k] = velocidade_terminal_chuva(self.qr[k], self.Nr[k], self.rho[k])

        return precip_acumulada

    # -------------------------------------------------------------
    # LOOP PRINCIPAL DE INTEGRACAO NO TEMPO
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
                proxima_gravacao += salvar_a_cada

        return self.historico
