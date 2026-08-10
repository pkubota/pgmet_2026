# -*- coding: utf-8 -*-
"""
modelo_nuvens.py
=================
Modelo numerico de parametrizacao de fracao de cobertura de nuvens.
MET-576-4 - Parametrizacao de Nuvens (Paulo Yoshio Kubota).

Script unico: contem tanto as funcoes de fisica (perfil atmosferico,
esquemas diagnosticos, esquema prognostico, overlap) quanto o bloco de
configuracao e a geracao das figuras.

COMO USAR
---------
Edite as opcoes no bloco "CONFIGURACAO" logo abaixo (regime atmosferico,
esquema diagnostico, comprimento de decorrelacao, cenario do ciclo de vida
etc.) e rode:

    python3 modelo_nuvens.py

O script calcula tudo e gera as figuras, salvando em OUTPUT_DIR e (se
SHOW_PLOTS=True) abrindo cada uma em uma janela do matplotlib.

Constantes de calibracao (Xu-Randall, taxas do esquema prognostico) sao
ILUSTRATIVAS, preservando a forma funcional dos esquemas originais; para uso
em pesquisa, devem ser recalibradas contra dados observacionais.
"""

# =============================================================================
# PARTE 1 - FISICA DO MODELO (antigo cloud_model.py)
# =============================================================================

import numpy as np

# ---------------------------------------------------------------------------
# Constantes fisicas
# ---------------------------------------------------------------------------
G = 9.80665          # gravidade (m/s^2)
RD = 287.05           # constante dos gases para o ar seco (J/kg/K)
P0 = 1013.25          # pressao de referencia em superficie (hPa)


# ---------------------------------------------------------------------------
# 1. Termodinamica basica
# ---------------------------------------------------------------------------
def esat_hPa(Tc):
    """Pressao de vapor de saturacao (hPa), formula de Bolton (1980).

    Parameters
    ----------
    Tc : float ou array
        Temperatura em graus Celsius.
    """
    Tc = np.asarray(Tc, dtype=float)
    return 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))


def qsat_kgkg(Tc, p_hPa):
    """Umidade especifica de saturacao (kg/kg)."""
    es = esat_hPa(Tc)
    p_hPa = np.asarray(p_hPa, dtype=float)
    return 0.622 * es / np.maximum(p_hPa - 0.378 * es, 1.0)


# ---------------------------------------------------------------------------
# 2. Grade vertical e perfil atmosferico idealizado
# ---------------------------------------------------------------------------
def build_grid(nlev=30, ztop=14500.0):
    """Retorna a grade de alturas (m), ascendente, 0..ztop."""
    return np.linspace(0.0, ztop, nlev)


REGIME_PRESETS = {
    # T0 (degC), RH0 (%) superficie, zi (m) topo da CLP/inversao,
    # RHabove (%) UR logo acima da inversao, lapse (degC/km) na CLP
    "strato": dict(T0=25, RH0=88, zi=1400, RHabove=28, lapse=6.5),
    "conv":   dict(T0=27, RH0=78, zi=9500, RHabove=62, lapse=7.0),
    "front":  dict(T0=15, RH0=82, zi=6000, RHabove=70, lapse=6.0),
    "clear":  dict(T0=22, RH0=35, zi=2000, RHabove=15, lapse=8.5),
}

REGIME_LABELS = {
    "strato": "Estratocumulo",
    "conv": "Convectivo profundo",
    "front": "Frontal / estratiforme",
    "clear": "Ceu claro",
}


def atmospheric_profile(regime="strato", z=None, T0=None, RH0=None, zi=None,
                         RHabove=None, lapse=None):
    """Constroi um perfil atmosferico idealizado.

    Pode-se usar um regime pre-definido (`REGIME_PRESETS`) e/ou sobrepor
    parametros individualmente (T0, RH0, zi, RHabove, lapse).

    Returns
    -------
    dict com arrays 'z' (m), 'p' (hPa), 'T' (degC), 'RH' (%), 'qs' (kg/kg),
    'q' (kg/kg) e o dicionario de parametros usado em 'params'.
    """
    if z is None:
        z = build_grid()
    preset = dict(REGIME_PRESETS.get(regime, REGIME_PRESETS["strato"]))
    if T0 is not None: preset["T0"] = T0
    if RH0 is not None: preset["RH0"] = RH0
    if zi is not None: preset["zi"] = zi
    if RHabove is not None: preset["RHabove"] = RHabove
    if lapse is not None: preset["lapse"] = lapse

    p_T0, p_RH0, p_zi, p_RHabove, p_lapse = (
        preset["T0"], preset["RH0"], preset["zi"], preset["RHabove"], preset["lapse"]
    )

    T0K = p_T0 + 273.15
    lapse_km = p_lapse / 1000.0          # K/m na CLP
    lapse_free = 6.5 / 1000.0            # K/m na troposfera livre (fixo)
    inv_jump = 6.5 if regime == "strato" else (2.0 if regime == "clear" else 1.0)

    T = np.empty_like(z)
    p = np.empty_like(z)
    RH = np.empty_like(z)

    Tzi = p_T0 - lapse_km * p_zi
    for i, zi_lev in enumerate(z):
        if zi_lev <= p_zi:
            Tc = p_T0 - lapse_km * zi_lev
        else:
            Tc = (Tzi - inv_jump) - lapse_free * (zi_lev - p_zi)
        T[i] = Tc

        Tmean_below = T0K - lapse_km * min(zi_lev, p_zi) / 2.0
        Tmean_above = lapse_free * max(zi_lev - p_zi, 0.0) / 2.0
        TmeanK = max(Tmean_below - Tmean_above, 180.0)
        p[i] = P0 * np.exp(-G * zi_lev / (RD * TmeanK))

        if regime == "clear":
            rh = p_RH0 * np.exp(-zi_lev / 4500.0) * 0.6 + 3.0
        elif zi_lev <= p_zi:
            frac = zi_lev / p_zi if p_zi > 0 else 0.0
            RHtop = min(99.0, p_RH0 + (96.0 - p_RH0) * 0.4)
            rh = p_RH0 + (RHtop - p_RH0) * frac
        else:
            decay_len = 6000.0 if regime == "conv" else (4500.0 if regime == "front" else 3000.0)
            rh = p_RHabove * np.exp(-(zi_lev - p_zi) / decay_len)
        RH[i] = np.clip(rh, 2.0, 99.5)

    qs = qsat_kgkg(T, p)
    q = qs * RH / 100.0

    return dict(z=z, p=p, T=T, RH=RH, qs=qs, q=q, params=preset, regime=regime)


# ---------------------------------------------------------------------------
# 3. Esquemas diagnosticos de fracao de cobertura de nuvem
# ---------------------------------------------------------------------------
def diag_all_or_nothing(profile):
    """C = 1 se q >= q_s, senao C = 0."""
    return (profile["q"] >= profile["qs"]).astype(float)


def diag_sundqvist(profile, RHc_pct=78.0):
    """Esquema de UR critica (tipo Sundqvist, 1989 / Slingo, 1980, 1987).

    C = 1 - sqrt(1 - x),  x = (UR - RHc)/(1 - RHc),  0 <= x <= 1
    """
    RHc = RHc_pct / 100.0
    RH = profile["RH"] / 100.0
    x = np.clip((RH - RHc) / max(1 - RHc, 1e-6), 0.0, 1.0)
    C = 1.0 - np.sqrt(1.0 - x)
    C[RH <= RHc] = 0.0
    return C


def diag_xu_randall(profile, RHc_pct=78.0, p_exp=0.25, beta=100.0, gamma=0.49,
                     lw_coeff=0.003):
    """Xu & Randall (1996), forma ilustrativa.

    C = UR^p * { 1 - exp[ -beta*q_l / ((1-UR)*q_s)^gamma ] }

    q_l e aproximado a partir do excesso de UR sobre a UR critica (proxy do
    condensado presente mesmo quando a media de grade esta sub-saturada --
    situacao tipica de nuvens parcialmente cobrindo a celula de grade).
    """
    RHc = np.clip(RHc_pct / 100.0, 0.3, 0.95)
    RH = np.clip(profile["RH"] / 100.0, 0.02, 0.98)
    qs = profile["qs"]
    ql = lw_coeff * qs * np.maximum(0.0, RH - RHc) / max(1 - RHc, 0.05)
    denom = np.power((1 - RH) * qs, gamma)
    denom = np.maximum(denom, 1e-12)
    term = np.exp(-beta * ql / denom)
    return np.power(RH, p_exp) * (1.0 - term)


def diag_pdf_triangular(profile, frac_b=0.15):
    """Esquema estatistico com PDF triangular de agua total (tipo Smith, 1990).

    A largura da PDF e definida como uma fracao `frac_b` da umidade de
    saturacao local, b(z) = frac_b * q_s(z) -- e nao um valor absoluto fixo.
    Isso evita fracao de nuvem espuria em altitude, onde q_s e muito pequeno
    (um b fixo em g/kg passaria a dominar sobre q e q_s nesses niveis).

    s = q - q_s ,  x = s / b(z)
    C = 0                  se x <= -1
    C = 0.5*(1+x)^2        se -1 < x <= 0
    C = 1 - 0.5*(1-x)^2    se  0 < x < 1
    C = 1                  se x >= 1
    """
    b = np.maximum(frac_b * profile["qs"], 1e-12)
    s = profile["q"] - profile["qs"]
    x = s / b
    C = np.where(
        x <= -1, 0.0,
        np.where(x >= 1, 1.0,
                 np.where(x <= 0, 0.5 * (1 + x) ** 2, 1 - 0.5 * (1 - x) ** 2))
    )
    return C


DIAG_SCHEMES = {
    "allnothing": ("Tudo-ou-nada", diag_all_or_nothing),
    "sundqvist": ("UR critica (Sundqvist/Slingo)", diag_sundqvist),
    "xurandall": ("Xu & Randall (1996)", diag_xu_randall),
    "pdf": ("PDF estatistica (Smith, 1990)", diag_pdf_triangular),
}


def compute_all_diagnostics(profile, RHc_pct=78.0, frac_b=0.15):
    """Calcula os quatro esquemas diagnosticos de uma vez. Retorna dict."""
    return {
        "allnothing": diag_all_or_nothing(profile),
        "sundqvist": diag_sundqvist(profile, RHc_pct),
        "xurandall": diag_xu_randall(profile, RHc_pct),
        "pdf": diag_pdf_triangular(profile, frac_b),
    }


# ---------------------------------------------------------------------------
# 4. Overlap vertical
# ---------------------------------------------------------------------------
def overlap_random(C):
    """C_tot = 1 - prod(1 - C_i)  (camadas estatisticamente independentes)."""
    C = np.asarray(C, dtype=float)
    return 1.0 - np.prod(1.0 - C)


def overlap_maximum(C):
    """C_tot = max_i(C_i)  (alinhamento vertical total)."""
    return float(np.max(C))


def overlap_max_random(C):
    """Maximo-aleatorio (Geleyn & Hollingsworth, 1979), forma recursiva.

    C_tot(1) = C(1)
    C_tot(k) = 1 - [1-C_tot(k-1)]*[1-max(C(k),C(k-1))] / [1-C(k-1)]
    """
    C = np.asarray(C, dtype=float)
    ctot = C[0]
    for k in range(1, len(C)):
        ck, ck1 = C[k], C[k - 1]
        if ck1 < 0.999999:
            ctot = 1 - (1 - ctot) * (1 - max(ck, ck1)) / (1 - ck1)
        else:
            ctot = 1 - (1 - ctot) * (1 - ck)
    return float(np.clip(ctot, 0.0, 1.0))


def overlap_exp_random(C, dz, Ldecorr):
    """Exponencial-aleatorio (Hogan & Illingworth, 2000).

    alpha_k = exp(-dz_k / L_decorr)
    C_tot(k) = alpha_k * (combinacao max-aleatoria) +
               (1-alpha_k) * (combinacao aleatoria)

    alpha -> 1 (L_decorr >> dz)  =>  tende ao maximo-aleatorio
    alpha -> 0 (L_decorr << dz)  =>  tende ao aleatorio
    """
    C = np.asarray(C, dtype=float)
    dz = np.asarray(dz, dtype=float)
    ctot = C[0]
    for k in range(1, len(C)):
        ck, ck1 = C[k], C[k - 1]
        alpha = np.exp(-dz[k - 1] / max(Ldecorr, 1.0))
        if ck1 < 0.999999:
            max_rand = 1 - (1 - ctot) * (1 - max(ck, ck1)) / (1 - ck1)
        else:
            max_rand = 1 - (1 - ctot) * (1 - ck)
        rand = 1 - (1 - ctot) * (1 - ck)
        ctot = alpha * max_rand + (1 - alpha) * rand
    return float(np.clip(ctot, 0.0, 1.0))


OVERLAP_SCHEMES = {
    "random": ("Aleatorio", lambda C, dz, L: overlap_random(C)),
    "max": ("Maximo", lambda C, dz, L: overlap_maximum(C)),
    "maxrandom": ("Maximo-aleatorio", lambda C, dz, L: overlap_max_random(C)),
    "exprandom": ("Exponencial-aleatorio", lambda C, dz, L: overlap_exp_random(C, dz, L)),
}


def compute_all_overlaps(C, z, Ldecorr=2000.0):
    """Calcula os quatro totais de cobertura (0-1) a partir do perfil C(z)."""
    dz = np.diff(z)
    return {
        "random": overlap_random(C),
        "max": overlap_maximum(C),
        "maxrandom": overlap_max_random(C),
        "exprandom": overlap_exp_random(C, dz, Ldecorr),
    }


# ---------------------------------------------------------------------------
# 5. Esquema prognostico (tipo Tiedtke, 1993) e ciclo de vida
# ---------------------------------------------------------------------------
def forcing(t, scenario):
    """Forcante de grande escala do cenario escolhido.

    Retorna (ascent, conv, subsidence) em [0,1] no instante t (horas).
    """
    if scenario == "frontal":
        ascent = max(0.0, np.sin(np.pi * min(t, 18) / 18))
        return ascent, 0.0, 0.0
    if scenario == "conv":
        conv = np.exp(-((t - 11) / 2.6) ** 2)
        ascent = 0.25 * conv
        return ascent, conv, 0.0
    # subsidencia: ascensao breve seguida de subsidencia sustentada (seca)
    ascent = max(0.0, np.sin(np.pi * min(t, 6) / 6)) if t < 6 else 0.0
    subs = min(1.0, (t - 6) / 6) if t > 6 else 0.0
    return ascent, 0.0, subs


def run_prognostic(profile, scenario="frontal", a1=0.9, a2=0.6, a3=0.7, a4=0.35,
                    RHc_pct=78.0, hours=24.0, nt=480, tau_precip=3.0):
    """Integra dC/dt no tempo (Euler explicito) para todos os niveis.

    dC/dt = a1*(1-C)*max(0,UR-RHc)/(1-RHc)        [fonte estratiforme]
          + a2*D_conv(t,z)*(1-C)                  [fonte convectiva]
          - a3*C*max(0,RHc-UR)/RHc                [sumidouro: evaporacao]
          - a4*C^2/tau_precip                     [sumidouro: autoconversao]

    Returns
    -------
    t : array (nt,) tempo em horas
    Chist : array (nt, nlev) fracao de nuvem por nivel e instante
    """
    z = profile["z"]
    nlev = len(z)
    RHc = RHc_pct / 100.0
    RH0 = profile["RH"] / 100.0
    dt = hours / nt

    conv_lev_min = int(nlev * 0.35)
    conv_lev_max = int(nlev * 0.65)

    C = np.full(nlev, 0.02)
    Chist = np.empty((nt, nlev))
    tvec = np.empty(nt)

    for ti in range(nt):
        t = ti * dt
        tvec[ti] = t
        ascent, conv, subs = forcing(t, scenario)

        RH = RH0 + 0.30 * ascent - 0.25 * subs
        RH = np.clip(RH, 0.02, 1.0)

        src_strat = a1 * (1 - C) * np.maximum(0, RH - RHc) / max(1 - RHc, 0.05)

        in_conv_layer = np.zeros(nlev)
        in_conv_layer[conv_lev_min:conv_lev_max + 1] = 1.0
        src_conv = a2 * conv * in_conv_layer * (1 - C)

        sink_evap = a3 * C * np.maximum(0, RHc - RH) / max(RHc, 0.05)
        sink_precip = a4 * C ** 2 / tau_precip

        dC = (src_strat + src_conv - sink_evap - sink_precip) * dt
        C = np.clip(C + dC, 0.0, 1.0)
        Chist[ti] = C

    return tvec, Chist


def overlap_time_series(Chist, z, method="maxrandom", Ldecorr=2000.0):
    """Aplica um esquema de overlap a cada instante de Chist (nt, nlev)."""
    dz = np.diff(z)
    fn = {
        "random": lambda C: overlap_random(C),
        "max": lambda C: overlap_maximum(C),
        "maxrandom": lambda C: overlap_max_random(C),
        "exprandom": lambda C: overlap_exp_random(C, dz, Ldecorr),
    }[method]
    return np.array([fn(C) for C in Chist])


def classify_phase(ctot_series, idx, window=3):
    """Classifica a fase do ciclo de vida no indice `idx` a partir da
    tendencia recente de cobertura total (heuristica simples, pedagogica)."""
    idx = max(0, min(len(ctot_series) - 1, idx))
    d = ctot_series[idx] - ctot_series[max(0, idx - window)] if idx > window else 0.0
    if d > 0.01:
        return "Formacao / Crescimento"
    if d < -0.01:
        return "Dissipacao"
    if ctot_series[idx] > 0.25:
        return "Maduro"
    return "Ceu claro"

# =============================================================================
# PARTE 2 - CONFIGURACAO E SCRIPT PRINCIPAL (antigo main.py)
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# =============================================================================
# CONFIGURACAO -- edite os valores abaixo e rode o script
# =============================================================================

# ---- 1) Perfil atmosferico -------------------------------------------------
REGIME = "strato"          # 'strato' | 'conv' | 'front' | 'clear'

# Sobrescreve parametros do regime escolhido (deixe None para usar o padrao)
T0 = None                  # temperatura de superficie (degC)
RH0 = None                 # UR de superficie (%)
ZI = None                  # altura da inversao / topo da CLP (m)
RHABOVE = None             # UR acima da inversao (%)
LAPSE = None               # lapse rate na CLP (degC/km)

# ---- 2) Esquemas diagnosticos ----------------------------------------------
RHC = 78.0                 # UR critica (%) -- usada em Sundqvist/Slingo e Xu-Randall
FRAC_B = 0.15               # largura da PDF triangular, como fracao de q_s(z)

# ---- 3) Overlap & decorrelacao ---------------------------------------------
SCHEME_FOR_OVERLAP = "sundqvist"   # 'allnothing' | 'sundqvist' | 'xurandall' | 'pdf'
LDECORR = 2000.0                    # comprimento de decorrelacao (m)

# ---- 4) Ciclo de vida (esquema prognostico) --------------------------------
SCENARIO = "frontal"       # 'frontal' | 'conv' | 'subs'
A1 = 0.9                   # taxa de formacao estratiforme (/h)
A2 = 0.6                   # taxa de detrainamento convectivo (/h)
A3 = 0.7                   # taxa de evaporacao / sumidouro (/h)
A4 = 0.35                  # taxa de conversao em precipitacao (/h)
OVERLAP_FOR_LIFECYCLE = "maxrandom"  # overlap usado na serie temporal
MAKE_ANIMATION = False      # True gera GIF do ciclo de vida (mais lento)

# ---- 5) O que rodar ---------------------------------------------------------
RUN_DIAGNOSTIC = True       # Etapa 1: perfil + esquemas diagnosticos
RUN_OVERLAP = True          # Etapa 2: overlap + comprimento de decorrelacao
RUN_LIFECYCLE = True        # Etapa 3: ciclo de vida prognostico

# ---- 6) Saida ----------------------------------------------------------------
OUTPUT_DIR = "saida_modelo"
SHOW_PLOTS = True            # True abre as janelas do matplotlib; False so salva

# =============================================================================
# A partir daqui e so o codigo do modelo -- normalmente nao precisa mexer.
# =============================================================================

DIAG_COLORS = {"allnothing": "#9AA7B2", "sundqvist": "#2E86AB",
               "xurandall": "#E08A2C", "pdf": "#6E4FA3"}
OVERLAP_COLORS = {"random": "#9AA7B2", "max": "#E08A2C",
                   "maxrandom": "#2E86AB", "exprandom": "#6E4FA3"}
SCENARIO_LABELS = {"frontal": "Ascensao frontal (estratiforme)",
                    "conv": "Pulso convectivo diurno",
                    "subs": "Subsidencia (dissipacao rapida)"}


def _finish(fig, filename):
    """Salva a figura em OUTPUT_DIR e, se configurado, mostra na tela."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=140, facecolor="white")
    print(f"  figura salva: {path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# ETAPA 1 -- Perfil atmosferico & esquemas diagnosticos
# -----------------------------------------------------------------------------
def etapa1_diagnostico(prof):
    print("\n[Etapa 1] Perfil atmosferico e esquemas diagnosticos")
    diags = compute_all_diagnostics(prof, RHc_pct=RHC, frac_b=FRAC_B)
    z_km = prof["z"] / 1000.0

    fig, (ax_rh, ax_diag) = plt.subplots(1, 2, figsize=(11, 5.6))

    ax_rh.plot(prof["RH"], z_km, color="#2E86AB", lw=2.2)
    ax_rh.set_xlim(0, 100)
    ax_rh.set_xlabel("UR (%)", color="#2E86AB")
    ax_rh.tick_params(axis="x", labelcolor="#2E86AB")
    ax_rh.set_ylabel("Altura (km)")
    ax_rh.grid(alpha=0.25)
    ax_rh.set_title(f"Perfil -- {REGIME_LABELS[REGIME]}")

    ax_t = ax_rh.twiny()
    ax_t.plot(prof["T"], z_km, color="#E08A2C", lw=2.0, ls="--")
    ax_t.set_xlabel("Temperatura (degC)", color="#E08A2C")
    ax_t.tick_params(axis="x", labelcolor="#E08A2C")

    zi_km = prof["params"]["zi"] / 1000.0
    ax_rh.axhline(zi_km, color="gray", lw=0.8, ls=":")

    for key, C in diags.items():
        label, _ = DIAG_SCHEMES[key]
        ax_diag.plot(C * 100, z_km, color=DIAG_COLORS[key], lw=2.0,
                     ls="--" if key == "allnothing" else "-", label=label)
    ax_diag.set_xlim(0, 100)
    ax_diag.set_xlabel("Fracao de cobertura C (%)")
    ax_diag.axhline(zi_km, color="gray", lw=0.8, ls=":")
    ax_diag.grid(alpha=0.25)
    ax_diag.legend(fontsize=9)
    ax_diag.set_title("Esquemas diagnosticos")

    fig.suptitle(
        f"RHc={RHC:.0f}%   frac_b={FRAC_B:.2f}   "
        f"T0={prof['params']['T0']:.0f}degC  RH0={prof['params']['RH0']:.0f}%  "
        f"zi={prof['params']['zi']:.0f}m",
        fontsize=9, color="#5C6E7D",
    )
    fig.tight_layout()
    _finish(fig, f"1_perfil_diagnostico_{REGIME}.png")
    return diags


# -----------------------------------------------------------------------------
# ETAPA 2 -- Overlap & comprimento de decorrelacao
# -----------------------------------------------------------------------------
def etapa2_overlap(prof, diags):
    print("\n[Etapa 2] Overlap vertical e comprimento de decorrelacao")
    C = diags[SCHEME_FOR_OVERLAP]
    z = prof["z"]
    overlaps = compute_all_overlaps(C, z, Ldecorr=LDECORR)

    print(f"  esquema usado: {DIAG_SCHEMES[SCHEME_FOR_OVERLAP][0]}")
    for key, val in overlaps.items():
        print(f"    {OVERLAP_SCHEMES[key][0]:<24s} C_tot = {val*100:5.1f} %")

    # --- Figura 2a: coluna de nuvem + barras de overlap ---
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4], wspace=0.35)

    ax_col = fig.add_subplot(gs[0, 0])
    ax_col.set_xlim(0, 1)
    ax_col.set_ylim(z.min(), z.max())
    ax_col.set_facecolor("#0B1A28")
    for i in range(len(z) - 1):
        Ci = 0.5 * (C[i] + C[i + 1])
        if Ci > 0.01:
            ax_col.axhspan(z[i], z[i + 1], xmin=0.05, xmax=0.95,
                            color="#F4F8FB", alpha=min(0.9, 0.15 + 0.75 * Ci))
    ax_col.set_ylabel("Altura (m)", color="#152534")
    ax_col.set_title(f"Perfil de nuvem -- {REGIME_LABELS[REGIME]}\n"
                      f"esquema: {DIAG_SCHEMES[SCHEME_FOR_OVERLAP][0]}",
                      color="#152534", fontsize=11, fontweight="bold", pad=10)
    ax_col.tick_params(colors="#152534")
    ax_col.set_xticks([])

    ax_bar = fig.add_subplot(gs[0, 1])
    names = ["Aleatorio", "Maximo", "Maximo-\naleatorio",
             f"Exp.-aleatorio\n(L={LDECORR:.0f} m)"]
    keys = ["random", "max", "maxrandom", "exprandom"]
    vals = [overlaps[k] * 100 for k in keys]
    bars = ax_bar.bar(names, vals, color=[OVERLAP_COLORS[k] for k in keys],
                       edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax_bar.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                    ha="center", fontsize=11, fontweight="bold")
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("Cobertura total de nuvem (%)")
    ax_bar.set_title("Cobertura total sob cada hipotese de overlap", fontsize=11)
    ax_bar.grid(axis="y", alpha=0.3)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.08)
    _finish(fig, f"2a_overlap_{REGIME}_{SCHEME_FOR_OVERLAP}.png")

    # --- Figura 2b: sensibilidade a L_decorr ---
    Ls = np.logspace(np.log10(50), np.log10(20000), 60)
    dz = np.diff(z)
    exp_vals = [overlap_exp_random(C, dz, L) for L in Ls]

    fig2, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(Ls, np.array(exp_vals) * 100, color=OVERLAP_COLORS["exprandom"],
            lw=2.4, label="Exponencial-aleatorio")
    ax.axhline(overlaps["random"] * 100, color=OVERLAP_COLORS["random"],
               ls="--", lw=1.6, label="Limite aleatorio (L-0)")
    ax.axhline(overlaps["maxrandom"] * 100, color=OVERLAP_COLORS["maxrandom"],
               ls="--", lw=1.6, label="Limite maximo-aleatorio (L-)")
    ax.axvline(LDECORR, color="gray", lw=1.0, ls=":",
               label=f"L_decorr atual = {LDECORR:.0f} m")
    ax.set_xscale("log")
    ax.set_xlabel("Comprimento de decorrelacao L$_{decorr}$ (m, escala log)")
    ax.set_ylabel("Cobertura total de nuvem (%)")
    ax.set_title(f"Sensibilidade da cobertura total a L$_{{decorr}}$\n"
                 f"{REGIME_LABELS[REGIME]} -- {DIAG_SCHEMES[SCHEME_FOR_OVERLAP][0]}",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig2.tight_layout()
    _finish(fig2, f"2b_decorrelacao_{REGIME}_{SCHEME_FOR_OVERLAP}.png")

    return overlaps


# -----------------------------------------------------------------------------
# ETAPA 3 -- Ciclo de vida (esquema prognostico)
# -----------------------------------------------------------------------------
def etapa3_ciclo_vida(prof):
    print("\n[Etapa 3] Ciclo de vida -- esquema prognostico")
    t, Chist = run_prognostic(prof, scenario=SCENARIO, a1=A1, a2=A2, a3=A3, a4=A4,
                                  RHc_pct=RHC)
    series = overlap_time_series(Chist, prof["z"], method=OVERLAP_FOR_LIFECYCLE,
                                     Ldecorr=LDECORR)

    idx_max = int(np.argmax(series))
    print(f"  cenario: {SCENARIO_LABELS[SCENARIO]}")
    print(f"  overlap da serie: {OVERLAP_SCHEMES[OVERLAP_FOR_LIFECYCLE][0]}")
    print(f"  cobertura maxima: {series[idx_max]*100:.0f}% em t={t[idx_max]:.1f} h")
    print(f"  cobertura final (t=24h): {series[-1]*100:.0f}%")

    # --- Figura 3a: serie temporal para os 3 cenarios (comparacao) ---
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for key, label in SCENARIO_LABELS.items():
        tk, Ck = run_prognostic(prof, scenario=key, a1=A1, a2=A2, a3=A3, a4=A4,
                                    RHc_pct=RHC)
        sk = overlap_time_series(Ck, prof["z"], method=OVERLAP_FOR_LIFECYCLE,
                                     Ldecorr=LDECORR)
        lw = 3.0 if key == SCENARIO else 1.6
        alpha = 1.0 if key == SCENARIO else 0.55
        ax.plot(tk, sk * 100, lw=lw, alpha=alpha,
                color={"frontal": "#2E86AB", "conv": "#E08A2C", "subs": "#6E4FA3"}[key],
                label=label + ("  (cenario atual)" if key == SCENARIO else ""))
    ax.set_xlabel("Tempo (h)")
    ax.set_ylabel("Cobertura total de nuvem (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Ciclo de vida da fracao de cobertura de nuvem -- {REGIME_LABELS[REGIME]}\n"
                 f"(overlap: {OVERLAP_SCHEMES[OVERLAP_FOR_LIFECYCLE][0]})", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _finish(fig, f"3a_ciclo_vida_series_{REGIME}.png")

    # --- Figura 3b: snapshots do perfil vertical em 4 instantes-chave ---
    # adaptativo ao pico real (idx_max), para capturar formacao/crescimento/
    # maduro/dissipacao de forma significativa em qualquer cenario
    nt = len(t)
    idx_dissip = idx_max + max(1, (nt - 1 - idx_max) // 2)
    snap_idx = sorted(set([0, max(1, idx_max // 2), idx_max, min(nt - 1, idx_dissip)]))
    while len(snap_idx) < 4:
        snap_idx.append(nt - 1)

    fig3, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
    z_km = prof["z"] / 1000.0
    for ax, idx in zip(axes, snap_idx):
        ax.plot(Chist[idx] * 100, z_km, color="#2E86AB", lw=2.2)
        ax.fill_betweenx(z_km, 0, Chist[idx] * 100, color="#2E86AB", alpha=0.18)
        ax.set_xlim(0, 100)
        phase = classify_phase(series, idx)
        ax.set_title(f"t = {t[idx]:.1f} h\n{phase}\nC_tot = {series[idx]*100:.0f}%",
                     fontsize=10)
        ax.set_xlabel("C (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Altura (km)")
    fig3.suptitle(f"Perfil vertical em instantes-chave do ciclo de vida -- "
                  f"{SCENARIO_LABELS[SCENARIO]}", fontsize=12)
    fig3.tight_layout(rect=[0, 0, 1, 0.94])
    _finish(fig3, f"3b_snapshots_{REGIME}_{SCENARIO}.png")

    # --- animacao opcional ---
    if MAKE_ANIMATION:
        print("  gerando animacao (GIF)...")
        _animar_ciclo_vida(prof, t, Chist, series)

    return t, Chist, series


def _animar_ciclo_vida(prof, t, Chist, series, n_frames=60, fps=10):
    z_km = prof["z"] / 1000.0
    fig, (ax_prof, ax_ts) = plt.subplots(1, 2, figsize=(10, 5.2),
                                          gridspec_kw={"width_ratios": [1, 1.4]})
    frame_idx = np.linspace(0, len(t) - 1, n_frames).astype(int)

    line_prof, = ax_prof.plot([], [], color="#2E86AB", lw=2.4)
    ax_prof.set_xlim(0, 100)
    ax_prof.set_ylim(z_km.min(), z_km.max())
    ax_prof.set_xlabel("C (%)")
    ax_prof.set_ylabel("Altura (km)")
    ax_prof.set_title("Perfil vertical")
    ax_prof.grid(alpha=0.3)

    ax_ts.plot(t, series * 100, color="#9AA7B2", lw=1.6)
    marker, = ax_ts.plot([], [], "o", color="#E08A2C", markersize=9)
    ax_ts.set_xlim(0, 24)
    ax_ts.set_ylim(0, 100)
    ax_ts.set_xlabel("Tempo (h)")
    ax_ts.set_ylabel("Cobertura total (%)")
    ax_ts.set_title(SCENARIO_LABELS[SCENARIO])
    ax_ts.grid(alpha=0.3)

    phase_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    def init():
        line_prof.set_data([], [])
        marker.set_data([], [])
        return line_prof, marker

    def update(frame):
        idx = frame_idx[frame]
        C = Chist[idx] * 100
        line_prof.set_data(C, z_km)
        for coll in ax_prof.collections:
            coll.remove()
        ax_prof.fill_betweenx(z_km, 0, C, color="#2E86AB", alpha=0.18)
        marker.set_data([t[idx]], [series[idx] * 100])
        phase = classify_phase(series, idx)
        phase_text.set_text(f"t = {t[idx]:.1f} h   |   fase: {phase}   |   "
                             f"C_total = {series[idx]*100:.0f}%")
        return line_prof, marker

    ani = animation.FuncAnimation(fig, update, frames=len(frame_idx), init_func=init,
                                   blit=False, interval=1000 / fps)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"3c_animacao_{REGIME}_{SCENARIO}.gif")
    ani.save(path, writer="pillow", fps=fps, dpi=110)
    plt.close(fig)
    print(f"  animacao salva: {path}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("MODELO NUMERICO DE PARAMETRIZACAO DE FRACAO DE COBERTURA DE NUVENS")
    print("=" * 70)
    print(f"Regime: {REGIME_LABELS[REGIME]}  |  Saida: {OUTPUT_DIR}/")

    prof = atmospheric_profile(REGIME, T0=T0, RH0=RH0, zi=ZI,
                                   RHabove=RHABOVE, lapse=LAPSE)

    diags = None
    if RUN_DIAGNOSTIC:
        diags = etapa1_diagnostico(prof)

    if RUN_OVERLAP:
        if diags is None:
            diags = compute_all_diagnostics(prof, RHc_pct=RHC, frac_b=FRAC_B)
        etapa2_overlap(prof, diags)

    if RUN_LIFECYCLE:
        etapa3_ciclo_vida(prof)

    print("\nConcluido. Figuras em:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
