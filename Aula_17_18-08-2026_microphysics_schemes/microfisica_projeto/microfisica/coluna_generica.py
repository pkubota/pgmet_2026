# -*- coding: utf-8 -*-
"""
coluna_generica.py
====================

Versao "standalone" (independente de qualquer classe de coluna) da
fisica completa de microfisica do Passo 3 (mesma sequencia de
processos de `coluna_step3.ColunaFaseMista._passo_processos_locais`),
para ser chamada por QUALQUER modelo externo que forneca seus proprios
campos de T, p, rho a cada passo de tempo -- por exemplo um modelo
dinamico 2D (vorticidade-funcao de corrente), em vez de vir de um
perfil hidrostatico interno como as classes ColunaChuvaQuente/
ColunaFaseGelo/ColunaFaseMista fazem.

Isso permite acoplar a microfisica de 6 categorias (qv,qc,qr,qi,qs,qg,
com Nc,Nr,Ni,Ns,Ng prognosticos) a um nucleo dinamico que resolve a
circulacao explicitamente (ex.: nuvem_2d_thompson.py), em vez do
ajuste de saturacao simplificado de 1 momento usado no modelo 2D
original do curso de Conveccao Atmosferica.

USO TIPICO (uma coluna x do dominio 2D, a cada passo de tempo):

    T_novo, qv_novo, qc_novo, Nc_novo, qr_novo, Nr_novo, \\
    qi_novo, Ni_novo, qs_novo, Ns_novo, qg_novo, Ng_novo = \\
        passo_microfisica_coluna(dt, T, p, rho, qv, qc, Nc, qr, Nr,
                                  qi, Ni, qs, Ns, qg, Ng)

Todos os argumentos (exceto dt) sao arrays numpy 1D de mesmo tamanho
(nz,), representando UMA coluna vertical num dado instante de tempo.
"""

import numpy as np

from .constantes import (Rd, g, cp, Lv, Ls, rho_w, rho_i, rho_s, rho_g,
                          MU_ICE, MU_SNOW, QMIN, NMIN, gamma_func)
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


def passo_microfisica_coluna(dt, T, p, rho, qv, qc, Nc, qr, Nr,
                              qi, Ni, qs, Ns, qg, Ng, evap_chuva=True):
    """
    Aplica UM passo de tempo de todos os processos de microfisica do
    Passo 3 a uma coluna vertical, EXATAMENTE na mesma ordem e com a
    mesma fisica de `ColunaFaseMista._passo_processos_locais`
    (coluna_step3.py) -- ver esse arquivo para a documentacao completa
    de cada processo (Pidsn, Pidep, Pccnd, Pifzc, Pgfzr, riming,
    Pispl, Picns, degelo, processos de chuva quente).

    Parametros
    ----------
    dt : passo de tempo (s)
    T, p, rho : temperatura (K), pressao (Pa), densidade do ar (kg/m3)
                -- FORNECIDOS EXTERNAMENTE pelo modelo hospedeiro (ex.:
                diagnosticados de theta+perturbacao num modelo 2D)
    qv, qc, Nc, qr, Nr, qi, Ni, qs, Ns, qg, Ng : arrays (nz,), estado
                atual de cada categoria
    evap_chuva : se False, desliga a evaporacao da chuva (Prevp) --
                util para reproduzir o experimento pedagogico "com/sem
                downdraft" do modelo 2D original (--evap-chuva off)

    Retorna
    -------
    T, qv, qc, Nc, qr, Nr, qi, Ni, qs, Ns, qg, Ng : arrays (nz,)
    atualizados apos o passo de tempo (T ja inclui o aquecimento/
    resfriamento latente de todos os processos).
    """
    nz = len(T)
    T_out = T.copy()
    qv_out = qv.copy()
    qc_out = qc.copy()
    Nc_out = Nc.copy()
    qr_out = qr.copy()
    Nr_out = Nr.copy()
    qi_out = qi.copy()
    Ni_out = Ni.copy()
    qs_out = qs.copy()
    Ns_out = Ns.copy()
    qg_out = qg.copy()
    Ng_out = Ng.copy()

    for k in range(nz):
        Tk, pk, rhok = T[k], p[k], rho[k]
        qvk = qv[k]
        qck, Nck = qc[k], Nc[k]
        qrk, Nrk = qr[k], Nr[k]
        qik, Nik = qi[k], Ni[k]
        qsk, Nsk = qs[k], Ns[k]
        qgk, Ngk = qg[k], Ng[k]

        # ===== 1) Pidsn: nucleacao primaria de gelo =====
        dqi_idsn, dNi_idsn = Pidsn(Nik, Tk, rhok, dt)
        dqi_idsn *= dt; dNi_idsn *= dt
        qi_new = qik + dqi_idsn
        Ni_new = Nik + dNi_idsn
        qv_new = qvk - dqi_idsn
        T_new = Tk + (Ls / cp) * dqi_idsn

        # ===== 2) Pidep, Psdep, Pgdep: deposicao/sublimacao =====
        dqi_idep = 0.0
        if Ni_new > NMIN:
            dqi_idep = max(Pidep(qv_new, qi_new, Ni_new, T_new, pk, rhok) * dt, -qi_new)
        qi_new += dqi_idep
        qv_new -= dqi_idep
        T_new += (Ls / cp) * dqi_idep

        dqs_sdep = 0.0
        qs_new, Ns_new = qsk, Nsk
        if Ns_new > NMIN:
            dqs_sdep = max(Psdep(qv_new, qs_new, Ns_new, T_new, pk, rhok) * dt, -qs_new)
        qs_new += dqs_sdep
        qv_new -= dqs_sdep
        T_new += (Ls / cp) * dqs_sdep

        dqg_gdep = 0.0
        qg_new, Ng_new = qgk, Ngk
        if Ng_new > NMIN:
            dqg_gdep = max(Pgdep(qv_new, qg_new, Ng_new, T_new, pk, rhok) * dt, -qg_new)
        qg_new += dqg_gdep
        qv_new -= dqg_gdep
        T_new += (Ls / cp) * dqg_gdep

        # ===== 3) Pccnd: condensacao/evaporacao liquida (efeito WBF) =====
        dqc_ccnd, dqv_ccnd, dT_ccnd = Pccnd(qv_new, qck, T_new, pk, dt)
        qc_new = qck + dqc_ccnd
        qv_new = qv_new + dqv_ccnd
        T_new = T_new + dT_ccnd
        Nc_new = Nck
        if dqc_ccnd > 0 and Nc_new <= NMIN:
            Nc_new = 1.0e8

        # ===== 4) Pifzc, Pgfzr: congelamento heterogeneo/homogeneo =====
        dqc_ifzc, dNc_ifzc = Pifzc(qc_new, Nc_new, T_new, dt)
        dqc_ifzc = max(dqc_ifzc * dt, -qc_new)
        dNc_ifzc = max(dNc_ifzc * dt, -Nc_new)
        qc_new += dqc_ifzc; Nc_new += dNc_ifzc
        qi_new += -dqc_ifzc; Ni_new += -dNc_ifzc
        T_new += (Ls - Lv) / cp * (-dqc_ifzc)

        dqr_gfzr, dNr_gfzr = Pgfzr(qrk, Nrk, T_new, dt)
        dqr_gfzr = max(dqr_gfzr * dt, -qrk)
        dNr_gfzr = max(dNr_gfzr * dt, -Nrk)
        qr_new = qrk + dqr_gfzr
        Nr_new = Nrk + dNr_gfzr
        qg_new += -dqr_gfzr
        Ng_new += -dNr_gfzr
        T_new += (Ls - Lv) / cp * (-dqr_gfzr)

        # ===== 5) Pi_iacw, Ps_sacw, Pgacw: riming (coletam agua de nuvem) =====
        Vi, _ = velocidade_terminal_gelo(qi_new, Ni_new, rhok)
        Vs, _ = velocidade_terminal_neve(qs_new, Ns_new, rhok)
        Vg, _ = velocidade_terminal_graupel(qg_new, Ng_new, rhok)

        dqc_iiacw = max(Pi_iacw(qc_new, Ni_new, qi_new, T_new, rhok) * dt, -qc_new)
        qc_new += dqc_iiacw
        qi_new += -dqc_iiacw
        T_new += (Ls - Lv) / cp * (-dqc_iiacw)

        D_s = diametro_medio_numero(lambda_gama(qs_new, Ns_new, rhok, rho_s, MU_SNOW), MU_SNOW) if qs_new > QMIN and Ns_new > NMIN else 0.0
        dqc_ssacw = max(colecao_continua(qc_new, Ns_new, D_s, Vs, 0.0, 1.0) * dt, -qc_new) if Ns_new > NMIN else 0.0
        qc_new += dqc_ssacw
        qs_new += -dqc_ssacw
        T_new += (Ls - Lv) / cp * (-dqc_ssacw)

        D_g = diametro_medio_numero(lambda_gama(qg_new, Ng_new, rhok, rho_g, MU_SNOW), MU_SNOW) if qg_new > QMIN and Ng_new > NMIN else 0.0
        dqc_gacw = max(colecao_continua(qc_new, Ng_new, D_g, Vg, 0.0, 1.0) * dt, -qc_new) if Ng_new > NMIN else 0.0
        qc_new += dqc_gacw
        qg_new += -dqc_gacw
        T_new += (Ls - Lv) / cp * (-dqc_gacw)

        # ===== 6) Pispl: Hallett-Mossop =====
        riming_total = (-dqc_iiacw - dqc_ssacw - dqc_gacw) / dt
        dNi_ispl = Pispl(riming_total, T_new) * dt
        Ni_new += dNi_ispl

        # ===== 7) Ps_sacr, Pgacr, Piacr: coletam chuva =====
        Vr, _ = velocidade_terminal_chuva(qr_new, Nr_new, rhok)

        dqr_ssacr = max(colecao_continua(qr_new, Ns_new, D_s, Vs, Vr, 1.0) * dt, -qr_new) if Ns_new > NMIN else 0.0
        qr_new += dqr_ssacr
        qs_new += -dqr_ssacr

        dqr_gacr = max(colecao_continua(qr_new, Ng_new, D_g, Vg, Vr, 1.0) * dt, -qr_new) if Ng_new > NMIN else 0.0
        qr_new += dqr_gacr
        qg_new += -dqr_gacr

        D_i = diametro_medio_numero(lambda_gama(qi_new, Ni_new, rhok, rho_i, MU_ICE), MU_ICE) if qi_new > QMIN and Ni_new > NMIN else 0.0
        dqr_iacr = max(colecao_continua(qr_new, Ni_new, D_i, Vi, Vr, 1.0) * dt, -qr_new) if Ni_new > NMIN else 0.0
        qr_new += dqr_iacr
        qg_new += -dqr_iacr
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
        dqc_ccnr, dNc_ccnr = Pccnr(qc_new, Nc_new, rhok)
        dqc_ccnr *= dt; dNc_ccnr *= dt
        dqc_ccnr = max(dqc_ccnr, -qc_new)

        dqc_racw = max(Pracw(qc_new, qr_new) * dt, -(qc_new + dqc_ccnr))

        transferencia = -(dqc_ccnr + dqc_racw)
        qc_new = qc_new + dqc_ccnr + dqc_racw
        qr_new = qr_new + transferencia
        Nr_new = Nr_new + (-dNc_ccnr)
        Nc_new = Nc_new + dNc_ccnr

        dNr_self = Pr_self(qr_new, Nr_new, rhok) * dt
        Nr_new = max(Nr_new + dNr_self, 0.0)

        if evap_chuva:
            dqr_revp_b, dNr_revp_b = Prevp(qr_new, Nr_new, qv_new, T_new, pk, rhok)
            dqr_revp = max(dqr_revp_b * dt, -qr_new)
            dNr_revp = max(dNr_revp_b * dt, -Nr_new)
            qr_new += dqr_revp
            Nr_new += dNr_revp
            qv_new -= dqr_revp
            T_new += (Lv / cp) * dqr_revp

        # ---- limpeza de valores residuais desprezveis ----
        if qc_new < QMIN: qc_new, Nc_new = 0.0, 0.0
        if qr_new < QMIN: qr_new, Nr_new = 0.0, 0.0
        if qi_new < QMIN: qi_new, Ni_new = 0.0, 0.0
        if qs_new < QMIN: qs_new, Ns_new = 0.0, 0.0
        if qg_new < QMIN: qg_new, Ng_new = 0.0, 0.0

        T_out[k] = T_new
        qv_out[k] = qv_new
        qc_out[k] = qc_new; Nc_out[k] = Nc_new
        qr_out[k] = qr_new; Nr_out[k] = Nr_new
        qi_out[k] = qi_new; Ni_out[k] = Ni_new
        qs_out[k] = qs_new; Ns_out[k] = Ns_new
        qg_out[k] = qg_new; Ng_out[k] = Ng_new

    return (T_out, qv_out, qc_out, Nc_out, qr_out, Nr_out,
            qi_out, Ni_out, qs_out, Ns_out, qg_out, Ng_out)
