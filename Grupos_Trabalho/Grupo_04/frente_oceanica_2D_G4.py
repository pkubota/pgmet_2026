# -*- coding: utf-8 -*-
"""
MET-579 - Grupo 4 (G4) - VERSAO 2D (lat x lon)
Simulacao idealizada de frentes oceanicas: sensibilidade ao esquema numerico
e a parametrizacao de difusao horizontal.

Extensao 2D do modelo 1D original: a frente agora MEANDRA em latitude (como
uma frente oceanica real, ex. regiao da Confluencia Brasil-Malvinas) e e
advectada zonalmente. Isso permite dois diagnosticos complementares:

  (A) DIFUSAO TRANSVERSAL A FRENTE (como no caso 1D): quanto do alargamento
      da frente e K_fis (parametrizacao fisica) vs K_num (erro de truncamento
      do esquema de adveccao)? K_efetivo = K_fis + K_num.

  (B) AMORTECIMENTO DO MEANDRO (NOVO, especifico do 2D): a amplitude do
      meandro da frente decai com o tempo por difusao meridional. Um esquema
      excessivamente difusivo "apaga" o meandro mais rapido do que a difusao
      fisica prescrita justificaria -- analogo directo ao problema conhecido
      em modelos oceanicos operacionais de difusao numerica amortecendo
      vortices/meandros de mesoescala que deveriam ser resolvidos pela
      dinamica (nao pela parametrizacao).

Dominio idealizado inspirado na regiao da Confluencia Brasil-Malvinas
(~38S), usado apenas para dar escalas realistas de grade e velocidade -
o dominio e periodico em longitude (canal zonal) e periodico em latitude
(um comprimento de onda completo do meandro cabe no dominio).

Autor: material de apoio MET-579 (INPE) - Grupo 4
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# 1. DOMINIO GEOGRAFICO (lat x lon) E CONVERSAO PARA METROS
# ---------------------------------------------------------------------------

R_TERRA = 6.371e6  # raio da Terra (m)

lat_min, lat_max = -42.0, -34.0     # graus - faixa meridional (~ Confluencia BM)
lon_min, lon_max = -58.0, -46.0     # graus - faixa zonal

nlat, nlon = 100, 120
lat = np.linspace(lat_min, lat_max, nlat, endpoint=False)   # graus
lon = np.linspace(lon_min, lon_max, nlon, endpoint=False)   # graus

lat0_ref = 0.5 * (lat_min + lat_max)   # latitude de referencia p/ conversao lon->m

dlat = (lat_max - lat_min) / nlat
dlon = (lon_max - lon_min) / nlon

dy = R_TERRA * np.deg2rad(dlat)                          # resolucao meridional (m)
dx = R_TERRA * np.cos(np.deg2rad(lat0_ref)) * np.deg2rad(dlon)  # resolucao zonal (m)

Ly = R_TERRA * np.deg2rad(lat_max - lat_min)              # extensao meridional (m)
Lx = R_TERRA * np.cos(np.deg2rad(lat0_ref)) * np.deg2rad(lon_max - lon_min)  # extensao zonal (m)

# coordenadas cartesianas locais (m), para os calculos numericos
x_m = np.arange(nlon) * dx     # distancia zonal (m)
y_m = np.arange(nlat) * dy     # distancia meridional (m)

print(f"Grade: dx={dx/1e3:.2f} km, dy={dy/1e3:.2f} km | Lx={Lx/1e3:.0f} km, Ly={Ly/1e3:.0f} km")

# ---------------------------------------------------------------------------
# 2. FRENTE IDEALIZADA COM MEANDRO
# ---------------------------------------------------------------------------

x0      = Lx / 4          # posicao zonal inicial da "linha media" da frente (m)
L0      = 3 * dx          # largura inicial da frente (m)
A_meandro = 0.06 * Lx     # amplitude do meandro (m)
n_meandro = 5              # numero de comprimentos de onda do meandro no dominio
                           # (mesoescala, ~180 km/onda; precisa ser inteiro p/ manter
                           # periodicidade em y). n=1 (uma unica onda larga) tem escala
                           # de decaimento difusivo de MILHARES de dias (ver LEIA-ME) -
                           # inviavel de observar numa integracao curta.
k_y     = 2*np.pi*n_meandro / Ly

u_adv   = 0.20            # velocidade de adveccao zonal da frente (m/s)

def posicao_frente(y, t, u=None, A=None):
    """Posicao zonal (m) da frente em funcao da latitude (y, em m) e do tempo."""
    if u is None:
        u = u_adv
    if A is None:
        A = A_meandro
    return x0 + u * t + A * np.sin(k_y * y)

def campo_frente(t, u=None, A=None):
    """Campo 2D (nlat x nlon) do tracador/temperatura no instante t."""
    Xf = posicao_frente(y_m, t, u=u, A=A)[:, None]   # (nlat, 1) - centro da frente por linha
    d = (x_m[None, :] - Xf + Lx/2) % Lx - Lx/2        # distancia zonal periodica (nlat, nlon)
    return 0.5 * (1.0 + np.tanh(d / L0))


# ---------------------------------------------------------------------------
# 3. ESQUEMA NUMERICO 2D (splitting simples: adveccao so em x; difusao em x e y)
# ---------------------------------------------------------------------------

def passo_2d(T, u, K, dt, esquema):
    """Um passo de tempo explicito. Adveccao zonal (u) discretizada pelo
    esquema escolhido (fonte do K_num); difusao (K, isotropica) sempre via
    diferencas centradas de 2a ordem em x E em y."""
    Txp = np.roll(T, -1, axis=1); Txm = np.roll(T, 1, axis=1)   # vizinhos em x (lon)
    Typ = np.roll(T, -1, axis=0); Tym = np.roll(T, 1, axis=0)   # vizinhos em y (lat)
    lap = (Txp - 2*T + Txm) / dx**2 + (Typ - 2*T + Tym) / dy**2

    if esquema == "upwind":
        adv_x = -u * (T - Txm) / dx
        return T + dt * (adv_x + K * lap)
    elif esquema == "centrado":
        adv_x = -u * (Txp - Txm) / (2*dx)
        return T + dt * (adv_x + K * lap)
    elif esquema == "lax_wendroff":
        C = u * dt / dx
        adv_lw = T - 0.5*C*(Txp - Txm) + 0.5*C**2*(Txp - 2*T + Txm)
        return adv_lw + K * dt * lap
    else:
        raise ValueError(esquema)


def simular_2d(esquema, K_fis, u=None, A=None, dt=None, T_total=None, n_snap=6):
    if u is None:
        u = u_adv
    if dt is None:
        dt_cfl_adv  = 0.5 * dx / u if u > 0 else np.inf
        dt_cfl_diff = 0.4 * min(dx, dy)**2 / K_fis if K_fis > 0 else np.inf
        dt = min(dt_cfl_adv, dt_cfl_diff, 3*3600.0)
    if T_total is None:
        T_total = 12 * 24 * 3600.0   # 12 dias

    nt = int(T_total / dt)
    snap_steps = np.linspace(0, nt, n_snap, dtype=int)

    T = campo_frente(0.0, u=u, A=A)
    tempos, campos = [], []
    for n in range(nt + 1):
        if n in snap_steps:
            tempos.append(n * dt)
            campos.append(T.copy())
        T = passo_2d(T, u, K_fis, dt, esquema)

    return np.array(tempos), np.array(campos), dt, (u*dt/dx if u > 0 else 0.0)


# ---------------------------------------------------------------------------
# 4. DIAGNOSTICO (A): LARGURA TRANSVERSAL DA FRENTE, MEDIA SOBRE AS LATITUDES
# ---------------------------------------------------------------------------

def perfil_tanh_1d(xx, xc, L):
    d = (xx - xc + Lx/2) % Lx - Lx/2
    return 0.5 * (1.0 + np.tanh(d / L))

def ajustar_linha(T_linha, xc_esperado, janela):
    d = (x_m - xc_esperado + Lx/2) % Lx - Lx/2
    mask = np.abs(d) < janela
    x_loc, T_loc = x_m[mask], T_linha[mask]
    if len(x_loc) < 5:
        return np.nan, np.nan
    try:
        p0 = [xc_esperado, L0 * 1.5]
        bounds = ([xc_esperado - janela, dx/2], [xc_esperado + janela, janela])
        popt, _ = curve_fit(perfil_tanh_1d, x_loc, T_loc, p0=p0, bounds=bounds, maxfev=5000)
        return popt[0], abs(popt[1])
    except Exception:
        return np.nan, np.nan

def diagnosticar_campo(campo, t, u=None, A=None):
    """Para um campo 2D num instante t, ajusta a largura L_j e o centro xc_j
    da frente em CADA linha de latitude. Retorna (xc_por_linha, L_por_linha)."""
    janela = Lx / 4
    xcs, Ls = np.zeros(nlat), np.zeros(nlat)
    for j in range(nlat):
        xc_esp = posicao_frente(y_m[j], t, u=u, A=A)
        xc, L = ajustar_linha(campo[j, :], xc_esp, janela)
        xcs[j], Ls[j] = xc, L
    return xcs, Ls


# ---------------------------------------------------------------------------
# 5. DIAGNOSTICO (B): AMPLITUDE DO MEANDRO (decaimento por difusao meridional)
# ---------------------------------------------------------------------------

def amplitude_meandro(xcs, t, u=None):
    """A partir das posicoes ajustadas da frente por latitude (xcs), remove a
    translacao media (u*t) e projeta o residuo sobre sin(k_y*y) para extrair
    a amplitude do meandro (m) -- decomposicao de Fourier de 1 modo."""
    if u is None:
        u = u_adv
    residuo = xcs - (x0 + u * t)
    # projecao (produto interno) sobre a base sin(k_y*y): A = 2*media(residuo*sin)
    proj_sin = 2 * np.mean(residuo * np.sin(k_y * y_m))
    proj_cos = 2 * np.mean(residuo * np.cos(k_y * y_m))
    A = np.hypot(proj_sin, proj_cos)
    return A


# ---------------------------------------------------------------------------
# 6. EXECUCAO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv

    # =======================================================================
    # EXPERIMENTO A: frente RETA (sem meandro) -- quantificacao precisa de
    # K_num, mesmo metodo validado no modelo 1D (calibracao empirica de alpha
    # + regressao de L^2 vs t). O meandro introduz efeitos geometricos
    # (inclinacao local da frente) que degradam a precisao quantitativa desse
    # diagnostico -- por isso ele e feito SEM meandro, exatamente como no 1D.
    # =======================================================================
    print("="*70)
    print("EXPERIMENTO A: quantificacao de K_num (frente reta, A_meandro=0)")
    print("="*70)

    def calibrar_alpha_2d(K_calib=50.0):
        tempos, campos, dt, _ = simular_2d("centrado", K_fis=K_calib, u=0.0, A=0.0,
                                            dt=3600.0, T_total=8*24*3600.0, n_snap=8)
        Lmeans = np.array([np.nanmean(diagnosticar_campo(c, t, u=0.0, A=0.0)[1])
                           for c, t in zip(campos, tempos)])
        mask = tempos > 0
        slope, _ = np.polyfit(tempos[mask], Lmeans[mask]**2 - L0**2, 1)
        return slope / K_calib

    alpha_2d = calibrar_alpha_2d()
    print(f"alpha_2D calibrado = {alpha_2d:.3f}  (nota: depende da resolucao/dominio;"
          f" NAO e o mesmo valor do modelo 1D -- por isso recalibramos aqui)")

    resultados_A = []
    K_fis_lista = (0.0, 20.0, 80.0)
    dados_snap_A = {}
    for esquema in ["upwind", "centrado", "lax_wendroff"]:
        for K_fis in K_fis_lista:
            if esquema == "centrado" and K_fis == 0.0:
                continue
            tempos, campos, dt, C = simular_2d(esquema, K_fis=K_fis, A=0.0, T_total=12*24*3600.0, n_snap=6)
            Lmeans = np.array([np.nanmean(diagnosticar_campo(c, t, A=0.0)[1])
                              for c, t in zip(campos, tempos)])
            mask = np.isfinite(Lmeans) & (tempos > 0)
            if mask.sum() < 2:
                K_eff = np.nan
            else:
                slope, _ = np.polyfit(tempos[mask], Lmeans[mask]**2 - L0**2, 1)
                K_eff = slope / alpha_2d
            K_num = K_eff - K_fis if np.isfinite(K_eff) else np.nan
            K_num_teo = 0.5*u_adv*dx*(1-C) if esquema == "upwind" else np.nan
            resultados_A.append(dict(esquema=esquema, K_fis=K_fis, Courant=C,
                                      K_efetivo=K_eff, K_num_diagnosticado=K_num,
                                      K_num_teorico=K_num_teo))
            dados_snap_A[(esquema, K_fis)] = (tempos, campos, Lmeans)

    print("\n{:<14s}{:>10s}{:>10s}{:>12s}{:>14s}{:>16s}".format(
        "esquema", "K_fis", "Courant", "K_efetivo", "K_num_diag", "K_num_teorico"))
    for r in resultados_A:
        teo = f'{r["K_num_teorico"]:.1f}' if np.isfinite(r["K_num_teorico"]) else "  -"
        print("{:<14s}{:>10.1f}{:>10.3f}{:>12.1f}{:>14.1f}{:>16s}".format(
            r["esquema"], r["K_fis"], r["Courant"], r["K_efetivo"],
            r["K_num_diagnosticado"], teo))

    with open("resultados_2D_experimentoA.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(resultados_A[0].keys()))
        w.writeheader(); w.writerows(resultados_A)

    # =======================================================================
    # EXPERIMENTO B: frente MEANDRANTE -- visualizacao + decaimento do meandro
    # =======================================================================
    print("\n" + "="*70)
    print("EXPERIMENTO B: frente meandrante - visualizacao e decaimento (tau)")
    print("="*70)

    K_fis_B = 80.0
    T_total_B = 200*24*3600.0
    resultados_B = []
    dados_snap_B = {}
    for esquema in ["upwind", "lax_wendroff"]:   # 'centrado' excluido: instavel
                                                   # nessa integracao longa com K_fis=80
                                                   # (mesmo achado do 1D - ver LEIA-ME)
        tempos, campos, dt, C = simular_2d(esquema, K_fis=K_fis_B, T_total=T_total_B, n_snap=10)
        As = np.array([amplitude_meandro(diagnosticar_campo(c, t)[0], t)
                       for c, t in zip(campos, tempos)])
        mask = As > 0
        slope, _ = np.polyfit(tempos[mask], np.log(As[mask]), 1)
        tau_diag = -1/slope/86400
        tau_teo = 1/(K_fis_B * k_y**2)/86400
        resultados_B.append(dict(esquema=esquema, K_fis=K_fis_B, tau_diagnosticado_dias=tau_diag,
                                  tau_teorico_dias=tau_teo))
        dados_snap_B[esquema] = (tempos, campos, As)
        print(f"{esquema:15s} tau_diagnosticado={tau_diag:.1f} dias   tau_teorico={tau_teo:.1f} dias")

    with open("resultados_2D_experimentoB.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(resultados_B[0].keys()))
        w.writeheader(); w.writerows(resultados_B)

    # =======================================================================
    # FIGURA 1: mapas lat-lon - evolucao da frente meandrante (upwind, K_fis=0)
    # =======================================================================
    tempos_vis, campos_vis, dt_vis, _ = simular_2d("upwind", K_fis=0.0, T_total=12*24*3600.0, n_snap=4)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, t, campo in zip(axes, tempos_vis, campos_vis):
        im = ax.pcolormesh(lon, lat, campo, cmap="RdBu_r", vmin=0, vmax=1, shading="auto")
        ax.set_title(f"t = {t/86400:.1f} dias")
        ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Traçador (adim.)")
    fig.suptitle("Figura 1 — Frente meandrante advectada e difundida (upwind, K_fís=0)\n"
                 "difusão puramente numérica alarga a frente ao longo do tempo", fontsize=11)
    fig.savefig("fig1_2D_mapas_evolucao.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =======================================================================
    # FIGURA 2: comparacao upwind vs lax_wendroff no instante final (K_fis=0)
    # =======================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, esquema in zip(axes, ["upwind", "lax_wendroff"]):
        tempos_e, campos_e, _, _ = simular_2d(esquema, K_fis=0.0, T_total=12*24*3600.0, n_snap=2)
        im = ax.pcolormesh(lon, lat, campos_e[-1], cmap="RdBu_r", vmin=-0.2, vmax=1.2, shading="auto")
        ax.set_title(f"{esquema}  (t=12 dias, K_fís=0)")
        ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle("Figura 2 — Upwind (difusivo) vs Lax-Wendroff (dispersivo/oscilatório)", fontsize=11)
    fig.savefig("fig2_2D_upwind_vs_lw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =======================================================================
    # FIGURA 3: tabela visual (barras) K_num diagnosticado por esquema (K_fis=0)
    # =======================================================================
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = [r["esquema"] for r in resultados_A if r["K_fis"] == 0.0]
    valores = [r["K_num_diagnosticado"] for r in resultados_A if r["K_fis"] == 0.0]
    ax.bar(labels, valores, color=["tab:red", "tab:green"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(r"$K_{num}$ diagnosticado (m²/s)")
    ax.set_title("Figura 3 — Difusão numérica espúria por esquema (2D, frente reta)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("fig3_2D_Knum_por_esquema.png", dpi=150)
    plt.close(fig)

    # =======================================================================
    # FIGURA 4: decaimento da amplitude do meandro (upwind vs lax_wendroff)
    # =======================================================================
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for esquema, cor in zip(["upwind", "lax_wendroff"], ["tab:red", "tab:green"]):
        tempos_b, campos_b, As_b = dados_snap_B[esquema]
        ax.semilogy(tempos_b/86400, As_b/1e3, "o-", color=cor, label=esquema)
    ax.set_xlabel("tempo (dias)")
    ax.set_ylabel("Amplitude do meandro (km, escala log)")
    ax.set_title(f"Figura 4 — Decaimento do meandro (K_fís={K_fis_B:.0f} m²/s)\n"
                 "praticamente idêntico entre esquemas — depende só de K_fís\n"
                 "(a difusão numérica do esquema em x não afeta a difusão em y aqui)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("fig4_2D_decaimento_meandro.png", dpi=150)
    plt.close(fig)

    print("\nConcluido. Arquivos gerados:")
    print("  resultados_2D_experimentoA.csv, resultados_2D_experimentoB.csv")
    print("  fig1_2D_mapas_evolucao.png, fig2_2D_upwind_vs_lw.png,")
    print("  fig3_2D_Knum_por_esquema.png, fig4_2D_decaimento_meandro.png")
