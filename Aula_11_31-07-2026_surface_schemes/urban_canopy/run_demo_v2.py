"""
run_demo_v2.py
==============

Demonstra as 4 melhorias da Etapa 2:

  A) Validacao: Etapa 2 (implicito) com dt pequeno (60 s) deve
     reproduzir de perto a Etapa 1 (explicito) no mesmo cenario.
  B) Robustez numerica: Etapa 2 com dt GRANDE (600 s = 10 min)
     permanece estavel, enquanto o esquema explicito da Etapa 1
     tende a oscilar/divergir no mesmo dt.
  C) Efeito da distribuicao multi-story r(k): compara um bairro de
     predios uniformes (5 andares) com um bairro de alturas mistas
     (mesma altura MEDIA), mostrando o efeito da densidade de area
     frontal (Ss) sobre a rugosidade (z0) e os fluxos.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, ".")
from urban_canyon_model import UrbanCanyonModel
from urban_canyon_model_v2 import UrbanCanyonModelV2


def make_forcing(amp_sw=700.0, t_sunrise=6.0, t_sunset=18.0,
                  Tref_mean=298.0, Tref_amp=6.0, u_ref=3.0):
    daylen = t_sunset - t_sunrise

    def forcing_fn(t_hours):
        t_mod = t_hours % 24.0
        if t_sunrise < t_mod < t_sunset:
            frac = (t_mod - t_sunrise) / daylen
            sun_elev = max(np.sin(np.pi * frac), 1e-3)
            zenith = np.pi / 2.0 * (1.0 - sun_elev)
            SWdir = amp_sw * sun_elev
            SWdif = 0.15 * amp_sw * sun_elev
        else:
            zenith = np.pi / 2.0
            SWdir = 0.0
            SWdif = 0.0
        Tref = Tref_mean + Tref_amp * np.sin(2 * np.pi * (t_mod - 9.0) / 24.0)
        LWdif = 0.75 * 5.67e-8 * Tref ** 4
        return dict(SWdir=SWdir, SWdif=SWdif, LWdif=LWdif,
                    zenith=zenith, Tref=Tref, u_ref=u_ref)
    return forcing_fn


# ---------------------------------------------------------------
# A) e B): validacao + robustez numerica
# ---------------------------------------------------------------
def compare_explicit_vs_implicit():
    forcing_fn = make_forcing()

    # --- dt pequeno: v1 explicito vs v2 implicito devem concordar ---
    m1 = UrbanCanyonModel(bw=20, kh0=30, Vuc=0.5)
    m1.Tr = m1.Tw = m1.Tg = 295.0
    out1 = m1.run(hours=48, dt_s=60.0, forcing_fn=forcing_fn)

    m2 = UrbanCanyonModelV2(bw=20, h0=30, Vuc=0.5, r_k={1: 1.0})
    m2.Tr = m2.Tw = m2.Tg = 295.0
    out2 = m2.run(hours=48, dt_s=60.0, forcing_fn=forcing_fn)

    mask = out1["t"] >= 24.0
    diff = np.abs(out1["Tg"][mask] - out2["Tg"][mask])
    print(f"[Validacao dt=60s] diferenca media |Tg_v1-Tg_v2|: {diff.mean():.3f} K "
          f"(max: {diff.max():.3f} K)")

    # --- dt grande: testar robustez ---
    m1b = UrbanCanyonModel(bw=20, kh0=30, Vuc=0.5)
    m1b.Tr = m1b.Tw = m1b.Tg = 295.0
    out1b = m1b.run(hours=48, dt_s=600.0, forcing_fn=forcing_fn)

    m2b = UrbanCanyonModelV2(bw=20, h0=30, Vuc=0.5, r_k={1: 1.0})
    m2b.Tr = m2b.Tw = m2b.Tg = 295.0
    out2b = m2b.run(hours=48, dt_s=600.0, forcing_fn=forcing_fn)

    print(f"\n[Robustez dt=600s] Etapa 1 (explicito): "
          f"Tg min={out1b['Tg'].min()-273.15:.1f}C max={out1b['Tg'].max()-273.15:.1f}C")
    print(f"[Robustez dt=600s] Etapa 2 (implicito):  "
          f"Tg min={out2b['Tg'].min()-273.15:.1f}C max={out2b['Tg'].max()-273.15:.1f}C")

    # --- dt ainda maior (1800s): forcar o limite do esquema explicito ---
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m1c = UrbanCanyonModel(bw=20, kh0=30, Vuc=0.5)
        m1c.Tr = m1c.Tw = m1c.Tg = 295.0
        out1c = m1c.run(hours=48, dt_s=1800.0, forcing_fn=forcing_fn)

    m2c = UrbanCanyonModelV2(bw=20, h0=30, Vuc=0.5, r_k={1: 1.0})
    m2c.Tr = m2c.Tw = m2c.Tg = 295.0
    out2c = m2c.run(hours=48, dt_s=1800.0, forcing_fn=forcing_fn)

    v1_nan = np.isnan(out1c["Tg"]).any()
    print(f"\n[Robustez dt=1800s] Etapa 1 (explicito): "
          f"{'DIVERGIU (NaN / overflow)' if v1_nan else 'Tg min=%.1fC max=%.1fC' % (out1c['Tg'].min()-273.15, out1c['Tg'].max()-273.15)}")
    print(f"[Robustez dt=1800s] Etapa 2 (implicito):  "
          f"Tg min={out2c['Tg'].min()-273.15:.1f}C max={out2c['Tg'].max()-273.15:.1f}C  (permanece estável)")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    t = out1b["t"][out1b["t"] >= 24.0] - 24.0
    ax.plot(t, out1b["Tg"][out1b["t"] >= 24.0] - 273.15,
            label="Etapa 1 (explícito), dt=600s", color="tab:red")
    ax.plot(t, out2b["Tg"][out2b["t"] >= 24.0] - 273.15,
            label="Etapa 2 (implícito), dt=600s", color="tab:blue")
    ax.set_xlabel("Hora do dia (h)")
    ax.set_ylabel("Tg - temperatura da rua (°C)")
    ax.set_title("Robustez numérica: dt grande (600 s)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_v2_robustez.png", dpi=130)


# ---------------------------------------------------------------
# C) Efeito da distribuicao multi-story
# ---------------------------------------------------------------
def compare_multistory():
    forcing_fn = make_forcing()

    # bairro uniforme: todos os predios com 5 andares (h0=3m -> 15m)
    uniform = UrbanCanyonModelV2(bw=20, h0=3.0, Vuc=0.5, r_k={5: 1.0})
    uniform.Tr = uniform.Tw = uniform.Tg = 295.0

    # bairro misto: mesma altura MEDIA (Ab=5), mas com predios de
    # 2, 5 e 10 andares (maior densidade de area frontal Ss)
    mixed = UrbanCanyonModelV2(bw=20, h0=3.0, Vuc=0.5,
                                r_k={2: 0.3, 5: 0.4, 10: 0.3})
    mixed.Tr = mixed.Tw = mixed.Tg = 295.0

    print(f"\n[Multi-story] Uniforme (5 andares): Ab={uniform.Ab:.2f}, "
          f"Ss={uniform.Ss:.3f}, z0={uniform.z0:.2f} m, d0={uniform.d0:.2f} m")
    print(f"[Multi-story] Misto (2/5/10 andares): Ab={mixed.Ab:.2f}, "
          f"Ss={mixed.Ss:.3f}, z0={mixed.z0:.2f} m, d0={mixed.d0:.2f} m")

    out_u = uniform.run(hours=48, dt_s=300.0, forcing_fn=forcing_fn)
    out_m = mixed.run(hours=48, dt_s=300.0, forcing_fn=forcing_fn)

    mask = out_u["t"] >= 24.0
    t = out_u["t"][mask] - 24.0

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax = axes[0]
    ax.plot(t, out_u["Tau"][mask] - 273.15, label="Uniforme (5 andares)", color="tab:blue")
    ax.plot(t, out_m["Tau"][mask] - 273.15, label="Misto (2/5/10 andares, mesma Ab)", color="tab:red")
    ax.set_ylabel("Tau - ar do canyon (°C)")
    ax.set_title("Efeito da distribuição multi-story (mesma altura média)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, out_u["Hatm"][mask], label="H p/ atmosfera - uniforme", color="tab:blue")
    ax.plot(t, out_m["Hatm"][mask], label="H p/ atmosfera - misto", color="tab:red")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Hora do dia (h)")
    ax.set_ylabel("H (W m$^{-2}$)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("fig_v2_multistory.png", dpi=130)


if __name__ == "__main__":
    compare_explicit_vs_implicit()
    compare_multistory()
