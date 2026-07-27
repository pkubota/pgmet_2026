"""
run_demo_v2b.py
================

Demonstra o fechamento de Km multi-nivel (item b): mostra o perfil
vertical de temperatura do ar dentro do canyon, Tau(k), para um
bairro de alturas mistas, em diferentes horarios do dia. Isso e algo
que o modelo anterior (no unico, Etapa 2 sem fechamento de Km) nao
conseguia representar.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, ".")
from urban_canyon_model_v2 import UrbanCanyonModelV2
from run_demo_v2 import make_forcing


def main():
    forcing_fn = make_forcing()
    m = UrbanCanyonModelV2(bw=20, h0=3.0, Vuc=0.5, r_k={2: 0.3, 5: 0.4, 10: 0.3})
    m.Tr = m.Tw = m.Tg = 295.0

    # roda ate 24h (1o dia) so para chegar a um estado de ciclo diurno razoavel
    out = m.run(hours=24.0, dt_s=300.0, forcing_fn=forcing_fn)

    horarios = [6, 9, 12, 15, 18, 21]
    fig, ax = plt.subplots(figsize=(6, 7))
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(horarios)))

    for h, color in zip(horarios, cmap):
        f = forcing_fn(h)
        Tw = np.interp(h, out["t"], out["Tw"])
        Tg = np.interp(h, out["t"], out["Tg"])
        Tau_k, Hw, Hg, Hatm = m.solve_canyon_air_column(Tw, Tg, f["Tref"], f["u_ref"])
        z = np.arange(1, m.n + 1) * m.h0
        ax.plot(Tau_k - 273.15, z, "o-", color=color, label=f"{h:02d}:00")

    ax.set_xlabel("Tau(k) - temperatura do ar no nível k (°C)")
    ax.set_ylabel("Altura z (m)")
    ax.set_title("Perfil vertical de Tau(k) dentro do canyon\n(bairro misto: 2/5/10 andares)")
    ax.legend(title="Hora")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_v2b_perfil_vertical.png", dpi=130)

    print("z2 (topo do dossel, altura media ponderada por Ab):", m.Ab * m.h0, "m")
    for h in horarios:
        f = forcing_fn(h)
        Tw = np.interp(h, out["t"], out["Tw"])
        Tg = np.interp(h, out["t"], out["Tg"])
        Tau_k, Hw, Hg, Hatm = m.solve_canyon_air_column(Tw, Tg, f["Tref"], f["u_ref"])
        print(f"{h:02d}h: Tau(1)={Tau_k[0]-273.15:5.1f}C  Tau(topo)={Tau_k[-1]-273.15:5.1f}C  "
              f"Hatm={Hatm:7.1f} W/m2")


if __name__ == "__main__":
    main()
