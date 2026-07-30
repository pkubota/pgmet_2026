# -*- coding: utf-8 -*-
"""
MET-579 - Grupo 4 (G4)
Simulacao idealizada de frentes oceanicas: sensibilidade ao esquema numerico
e a parametrizacao de difusao horizontal.

ENTREGAVEL DIFERENCIAL:
    Quantificar, para cada esquema numerico de adveccao, o quanto da difusao
    "efetiva" observada na simulacao e devida a difusao FISICA prescrita
    (K_fis, parametrizacao de viscosidade/difusao turbulenta horizontal)
    e o quanto e DIFUSAO NUMERICA espuria introduzida pelo proprio esquema
    (erro de truncamento), isto e:

        K_efetivo(t) = K_fis + K_numerico(esquema, u, dx, dt)

    A ideia pedagogica central: a difusao numerica pode MASCARAR ou
    SUBSTITUIR INDEVIDAMENTE a difusao fisica que o modelo deveria estar
    representando explicitamente. Um esquema muito difusivo (ex: upwind de
    1a ordem, C longe de 1) pode "fazer o papel" da parametrizacao de
    difusao horizontal mesmo com K_fis = 0 -- o que e um erro conceitual
    grave em modelagem de frentes oceanicas (a frente relaxa/alarga por
    razoes numericas, nao fisicas).

Autor: material de apoio MET-579 (INPE) - Grupo 4
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # compatibilidade com ambientes sem display (ex. JupyterLite)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# 1. CONFIGURACAO DO DOMINIO E DA FRENTE IDEALIZADA
# ---------------------------------------------------------------------------

Lx      = 400e3        # comprimento do dominio (m)  -> 400 km
nx      = 400           # numero de pontos de grade
dx      = Lx / nx       # resolucao espacial (m)
x       = np.linspace(0, Lx, nx, endpoint=False)

x0      = Lx / 4        # posicao inicial da frente (m)
L0      = 3 * dx        # largura inicial da frente (m) - poucos pontos de grade
                         # (frente "afiada", tipica de simulacao idealizada)

u_adv   = 0.30          # velocidade de advecao da frente (m/s) - jato geostrofico idealizado

def perfil_tanh(x, xc, L):
    """Perfil de frente em tangente hiperbolica (salto suavizado de T/rho),
    genuinamente PERIODICO em [0, Lx). Um unico tanh (x-xc)/L NAO e
    periodico (vai de 0 a 1 sem retornar) -- isso criava uma segunda
    frente espuria e descontinua exatamente na borda do dominio quando
    combinado com o esquema periodico (np.roll), contaminando o
    diagnostico de largura. Aqui usamos a distancia com sinal, respeitando
    a periodicidade (d em [-Lx/2, Lx/2)); o resultado e uma unica frente
    "principal" em xc e, automaticamente, uma segunda frente simetrica no
    ponto antipodal xc+Lx/2 -- exatamente como em canais periodicos
    idealizados (duas frentes, fisicamente consistente e sem descontinuidade
    numerica introduzida por acidente)."""
    d = (x - xc + Lx/2) % Lx - Lx/2
    return 0.5 * (1.0 + np.tanh(d / L))


# ---------------------------------------------------------------------------
# 2. ESQUEMAS NUMERICOS DE ADVECCAO-DIFUSAO (dominio periodico)
# ---------------------------------------------------------------------------

def passo_upwind(T, u, K, dx, dt):
    """Upwind 1a ordem (assume u > 0). Erro de truncamento ~ difusivo."""
    adv  = -u * (T - np.roll(T, 1)) / dx
    diff =  K * (np.roll(T, -1) - 2*T + np.roll(T, 1)) / dx**2
    return T + dt * (adv + diff)

def passo_centrado(T, u, K, dx, dt):
    """Centrado 2a ordem. Sem difusao numerica de 2a ordem, mas dispersivo
    e instavel se K nao for suficiente para conter o modo 2dx (grade)."""
    adv  = -u * (np.roll(T, -1) - np.roll(T, 1)) / (2*dx)
    diff =  K * (np.roll(T, -1) - 2*T + np.roll(T, 1)) / dx**2
    return T + dt * (adv + diff)

def passo_lax_wendroff(T, u, K, dx, dt):
    """Lax-Wendroff (2a ordem, adveccao + difusao numerica de 4a ordem
    fraca, mas dispersivo perto de gradientes fortes)."""
    C  = u * dt / dx
    Tp = np.roll(T, -1)
    Tm = np.roll(T,  1)
    adv_lw = T - 0.5*C*(Tp - Tm) + 0.5*C**2*(Tp - 2*T + Tm)
    diff   = K * dt * (Tp - 2*T + Tm) / dx**2
    return adv_lw + diff

ESQUEMAS = {
    "upwind":        passo_upwind,
    "centrado":      passo_centrado,
    "lax_wendroff":  passo_lax_wendroff,
}


def simular(esquema, K_fis, u=u_adv, dx=dx, dt=None, T_total=None, n_snap=6):
    """Roda a simulacao e retorna os instantes de tempo e os campos salvos."""
    if dt is None:
        # dt escolhido para respeitar CFL advectivo e difusivo com folga
        dt_cfl_adv  = 0.5 * dx / u if u > 0 else np.inf
        dt_cfl_diff = 0.4 * dx**2 / K_fis if K_fis > 0 else np.inf
        dt = min(dt_cfl_adv, dt_cfl_diff, 3600.0)  # tetos de 1h
    if T_total is None:
        T_total = 6 * 24 * 3600.0  # 6 dias

    nt = int(T_total / dt)
    snap_steps = np.linspace(0, nt, n_snap, dtype=int)

    T = perfil_tanh(x, x0, L0)
    passo = ESQUEMAS[esquema]

    tempos, campos = [], []
    for n in range(nt + 1):
        if n in snap_steps:
            tempos.append(n * dt)
            campos.append(T.copy())
        T = passo(T, u, K_fis, dx, dt)

    return np.array(tempos), np.array(campos), dt, (u*dt/dx)


# ---------------------------------------------------------------------------
# 3. DIAGNOSTICO: extrair a largura da frente L(t) por ajuste de tanh
# ---------------------------------------------------------------------------

def ajustar_largura(T_campo, xc_esperado):
    """Ajusta perfil tanh(x; xc, L) LOCALMENTE (janela em torno de
    xc_esperado = x0 + u*t, considerando a periodicidade) e retorna L (m).

    Usar uma janela local e essencial: o dominio tem DUAS frentes
    periodicas (principal em xc e sua simetrica antipodal em xc+Lx/2);
    sem restringir a regiao do ajuste, o curve_fit pode "confundir" as
    duas ou ser atraido pela frente errada, gerando fits espurios.
    """
    janela = Lx / 4  # metade da distancia entre as duas frentes periodicas
    d = (x - xc_esperado + Lx/2) % Lx - Lx/2
    mask = np.abs(d) < janela
    x_loc, T_loc = x[mask], T_campo[mask]
    if len(x_loc) < 5:
        return np.nan
    try:
        p0 = [xc_esperado, L0 * 1.5]
        bounds = ([xc_esperado - janela, dx/2], [xc_esperado + janela, janela])
        popt, _ = curve_fit(lambda xx, xc, L: perfil_tanh(xx, xc, L),
                             x_loc, T_loc, p0=p0, bounds=bounds, maxfev=5000)
        return abs(popt[1])
    except Exception:
        return np.nan


def calibrar_constante_alpha(K_fis_calib=50.0):
    """
    Experimento de CONTROLE: u = 0, difusao pura (esquema centrado e exato
    nesse caso: sem adveccao nao ha erro de truncamento advectivo).
    Serve para calibrar empiricamente a relacao L(t)^2 = L0^2 + alpha*K*t,
    em vez de assumir a priori a constante teorica do erf.
    """
    tempos, campos, dt, _ = simular("centrado", K_fis=K_fis_calib, u=0.0,
                                     T_total=8*24*3600.0, n_snap=8)
    # u=0 -> a frente nao se desloca; xc_esperado = x0 sempre
    Ls = np.array([ajustar_largura(c, x0) for c in campos])
    mask = np.isfinite(Ls) & (tempos > 0)
    slope, _ = np.polyfit(tempos[mask], Ls[mask]**2 - L0**2, 1)
    alpha = slope / K_fis_calib
    return alpha


# ---------------------------------------------------------------------------
# 4. EXPERIMENTO PRINCIPAL: varrer esquemas x K_fis prescrito
# ---------------------------------------------------------------------------

def rodar_experimento(alpha, K_fis_lista=(0.0, 20.0, 80.0)):
    resultados = []
    dados_snapshots = {}

    for esquema in ESQUEMAS:
        for K_fis in K_fis_lista:
            if esquema == "centrado" and K_fis == 0.0:
                continue  # instavel sem difusao minima (modo 2dx) - registrar a parte

            tempos, campos, dt, C = simular(esquema, K_fis)
            xc_esp = (x0 + u_adv * tempos) % Lx
            Ls = np.array([ajustar_largura(c, xc) for c, xc in zip(campos, xc_esp)])
            mask = np.isfinite(Ls) & (tempos > 0)

            if mask.sum() < 2:
                K_efetivo = np.nan
            else:
                slope, _ = np.polyfit(tempos[mask], Ls[mask]**2 - L0**2, 1)
                K_efetivo = slope / alpha

            K_num = K_efetivo - K_fis if np.isfinite(K_efetivo) else np.nan
            K_num_teorico_upwind = 0.5 * u_adv * dx * (1 - C) if esquema == "upwind" else np.nan

            resultados.append(dict(
                esquema=esquema, K_fis=K_fis, dt=dt, Courant=C,
                K_efetivo=K_efetivo, K_num_diagnosticado=K_num,
                K_num_teorico=K_num_teorico_upwind,
            ))
            dados_snapshots[(esquema, K_fis)] = (tempos, campos, Ls)

    return resultados, dados_snapshots


# ---------------------------------------------------------------------------
# 5. EXECUCAO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Calibrando constante alpha (experimento de controle, difusao pura)...")
    alpha = calibrar_constante_alpha()
    print(f"  alpha calibrado = {alpha:.3f}  (teorico p/ erf ~ 4.0 a 5.3, dependendo da definicao de L)")

    print("Rodando experimento principal (esquemas x K_fis)...")
    resultados, dados = rodar_experimento(alpha, K_fis_lista=(0.0, 20.0, 80.0))

    # ---- Tabela resumo ----
    print("\n{:<14s}{:>10s}{:>10s}{:>12s}{:>14s}{:>16s}".format(
        "esquema", "K_fis", "Courant", "K_efetivo", "K_num_diag", "K_num_teorico"))
    for r in resultados:
        print("{:<14s}{:>10.1f}{:>10.3f}{:>12.2f}{:>14.2f}{:>16s}".format(
            r["esquema"], r["K_fis"], r["Courant"], r["K_efetivo"],
            r["K_num_diagnosticado"],
            f'{r["K_num_teorico"]:.2f}' if np.isfinite(r["K_num_teorico"]) else "  -"))

    # salvar tabela em csv para uso no relatorio/lista de exercicios
    import csv
    with open("resultados_G4.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
        writer.writeheader()
        writer.writerows(resultados)

    # =======================================================================
    # FIGURA 1: evolucao da frente (snapshots) - upwind vs lax_wendroff, K_fis=0
    # =======================================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, esquema in zip(axes, ["upwind", "lax_wendroff"]):
        tempos, campos, Ls = dados[(esquema, 0.0)]
        cmap = plt.cm.viridis(np.linspace(0, 1, len(tempos)))
        for t, campo, c in zip(tempos, campos, cmap):
            ax.plot(x/1e3, campo, color=c, lw=1.6, label=f"t={t/86400:.1f} d")
        ax.set_title(f"Esquema: {esquema}  (K_fís = 0)")
        ax.set_xlabel("x (km)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Traçador / temperatura (adim.)")
    axes[0].legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle("Figura 1 — Alargamento espúrio da frente com K_fís = 0\n"
                 "(evidência de difusão puramente numérica)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig("fig1_snapshots_Kfis0.png", dpi=150)
    plt.close(fig)

    # =======================================================================
    # FIGURA 2: L^2 vs t (ajuste linear) para os 3 esquemas, K_fis = 20
    # =======================================================================
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cores = {"upwind": "tab:red", "centrado": "tab:blue", "lax_wendroff": "tab:green"}
    K_FIS_FIG2 = 80.0  # K_fis=20 deixa o esquema centrado marginalmente instavel (ver Figura 5)
    for esquema in ESQUEMAS:
        key = (esquema, K_FIS_FIG2)
        if key not in dados:
            continue
        tempos, campos, Ls = dados[key]
        mask = np.isfinite(Ls)
        ax.plot(tempos[mask]/86400, Ls[mask]**2/1e6, "o-", color=cores[esquema],
                label=esquema)
    ax.set_xlabel("tempo (dias)")
    ax.set_ylabel(r"$L(t)^2$  (km$^2$)")
    ax.set_title(f"Figura 2 — Crescimento de L² com K_fís = {K_FIS_FIG2:.0f} m²/s\n"
                 "(inclinação ∝ K_efetivo; compare entre esquemas)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig2_L2_vs_t_Kfis80.png", dpi=150)
    plt.close(fig)

    # =======================================================================
    # FIGURA 3: barras K_num diagnosticado por esquema (K_fis=0)
    # =======================================================================
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels, valores = [], []
    for r in resultados:
        if r["K_fis"] == 0.0:
            labels.append(r["esquema"])
            valores.append(r["K_num_diagnosticado"])
    ax.bar(labels, valores, color=["tab:red", "tab:green"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(r"$K_{num}$ diagnosticado (m²/s)")
    ax.set_title("Figura 3 — Difusão numérica espúria por esquema\n"
                 "(deveria ser 0; qualquer valor >0 é artefato do esquema)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("fig3_Knum_por_esquema.png", dpi=150)
    plt.close(fig)

    # =======================================================================
    # FIGURA 4: sensibilidade de K_num (upwind) ao numero de Courant
    #           + comparacao com a formula teorica K_num = 0.5 u dx (1-C)
    # =======================================================================
    Courants_teste = [0.9, 0.7, 0.5, 0.3, 0.15]
    K_num_diag, K_num_teo, Cs = [], [], []
    for C_alvo in Courants_teste:
        dt_c = C_alvo * dx / u_adv
        tempos, campos, dt_real, C_real = simular("upwind", K_fis=0.0, dt=dt_c,
                                                    T_total=6*24*3600.0, n_snap=6)
        xc_esp = (x0 + u_adv * tempos) % Lx
        Ls = np.array([ajustar_largura(c, xc) for c, xc in zip(campos, xc_esp)])
        mask = np.isfinite(Ls) & (tempos > 0)
        slope, _ = np.polyfit(tempos[mask], Ls[mask]**2 - L0**2, 1)
        K_eff = slope / alpha
        K_num_diag.append(K_eff)
        K_num_teo.append(0.5 * u_adv * dx * (1 - C_real))
        Cs.append(C_real)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(Cs, K_num_diag, "o-", label="K_num diagnosticado (ajuste de L(t))")
    ax.plot(Cs, K_num_teo, "s--", label=r"K_num teórico = ½ u Δx (1−C)")
    ax.set_xlabel("Número de Courant (C = uΔt/Δx)")
    ax.set_ylabel(r"$K_{num}$ (m²/s)")
    ax.set_title("Figura 4 — Esquema upwind: difusão numérica vs Courant\n"
                 "(validação do diagnóstico contra a teoria)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig4_Knum_vs_Courant_upwind.png", dpi=150)
    plt.close(fig)

    # =======================================================================
    # FIGURA 5 (ACHADO PEDAGOGICO): instabilidade do esquema centrado quando
    # a difusao (fisica + numerica) e insuficiente para satisfazer a
    # condicao de estabilidade de Peclet de malha: 2*K*dt/dx^2 >= (u*dt/dx)^2
    # =======================================================================
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for K_fis_teste, cor in zip([20.0, 80.0], ["tab:orange", "tab:blue"]):
        key = ("centrado", K_fis_teste)
        if key not in dados:
            continue
        tempos, campos, Ls = dados[key]
        mask = np.isfinite(Ls)
        ax.plot(tempos[mask]/86400, Ls[mask]/1e3, "o-", color=cor,
                label=f"K_fís = {K_fis_teste:.0f} m²/s")
    ax.set_xlabel("tempo (dias)")
    ax.set_ylabel("L(t)  (km)")
    ax.set_title("Figura 5 — Esquema centrado: instabilidade com difusão insuficiente\n"
                 "(K_fís=20 viola a condição de estabilidade de Péclet de malha;\n"
                 "o 'ruído' resultante é indistinguível de difusão espúria extrema)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig5_instabilidade_centrado.png", dpi=150)
    plt.close(fig)

    print("\nConcluido. Arquivos gerados:")
    print("  resultados_G4.csv")
    print("  fig1_snapshots_Kfis0.png")
    print("  fig2_L2_vs_t_Kfis80.png")
    print("  fig3_Knum_por_esquema.png")
    print("  fig4_Knum_vs_Courant_upwind.png")
    print("  fig5_instabilidade_centrado.png")
