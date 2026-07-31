# -*- coding: utf-8 -*-
"""
simulacao_passo3_relampago.py
================================

Acopla o diagnostico de taxa de relampago (`lightning_mod.py`, traducao
Python de `mpas_atmphys_lightning.F90`) ao modelo de microfisica de
nuvens do Passo 3 (`ColunaFaseMista`).

COMO FUNCIONA O ACOPLAMENTO
-----------------------------
O esquema de relampago (Potencial de Eletrificacao, EP) precisa, a
cada nivel vertical, de:

    z (m), T ( graus C), rho (kg/m3), w (m/s), qg, qs, qc (kg/kg)

Todas essas variaveis JA EXISTEM no `ColunaFaseMista`, EXCETO `w`
(velocidade vertical): o modelo de microfisica construido nos Passos
1-3 e uma COLUNA ESTATICA -- ele nao resolve a dinamica (equacao de
momento vertical), so a microfisica. Por isso, `w` precisa ser
PRESCRITO externamente aqui, como uma aproximacao do updraft convectivo
que sustentaria a nuvem simulada (num acoplamento operacional real
--MONAN/MPAS--, `w` viria do nucleo dinamico do modelo, resolvido a
cada passo de tempo, e o relampago seria diagnosticado "online" sem
precisar de nenhuma prescricao).

O perfil de `w` prescrito aqui e uma gaussiana simples, centrada na
camada de nuvem (mesma camada usada em `simulacao_passo3.py`), com pico
de 8 m/s -- compativel com a intensidade usada nos testes sinteticos de
`test_synthetic.py` do proprio modulo de relampago.

Como a microfisica aqui e EXPLICITAMENTE resolvida (nao ha esquema de
cumulus parametrizado atuando nesta coluna), usamos `cu_active=False`
e `dx_km` pequeno (2 km, resolucao convection-permitting) -- isso forca
o driver a usar exclusivamente o EP (nao o Price-Rind, que e a branch
para conveccao parametrizada em malha grossa).

Gera:
    (a) Serie temporal da taxa de flash (total, CG, IC) ao longo da
        simulacao microfisica do Passo 3
    (b) Comparacao lado a lado com a evolucao de qi, qs, qg (para ver a
        correlacao entre crescimento do gelo/graupel e a taxa de flash)

Para rodar:  python3 simulacao_passo3_relampago.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from microfisica.coluna_step3 import ColunaFaseMista
from microfisica.constantes import T0, QMIN
import lightning_mod as lm

driver_lightning = lm.mpas_atmphys_lightning.driver_lightning

# ----------------------------------------------------------------------
# 1) Configuracao da coluna (identica a simulacao_passo3.py)
# ----------------------------------------------------------------------
coluna = ColunaFaseMista(nz=80, dz=100.0, T_base=293.0, p_base=95000.0)

k_base = int(1500 / coluna.dz)
k_topo = int(6000 / coluna.dz)
coluna.inserir_nuvem(k_base, k_topo, qc_valor=1.0e-3, Nc_valor=2.0e8)

k_fria = int(4500 / coluna.dz)
coluna.qr[k_fria] = 5.0e-4
coluna.Nr[k_fria] = 5.0e5

# ----------------------------------------------------------------------
# 2) Perfil de velocidade vertical PRESCRITO (ver docstring acima)
# ----------------------------------------------------------------------
Z_PICO_W = 4000.0   # m, centro do updraft (meio da camada de nuvem)
SIGMA_W = 2000.0    # m, largura do updraft
W_PICO = 8.0        # m/s, intensidade do updraft (compativel com
                     # test_synthetic.py do modulo de relampago)

w_prescrito = W_PICO * np.exp(-0.5 * ((coluna.z - Z_PICO_W) / SIGMA_W) ** 2)

# ----------------------------------------------------------------------
# 3) Integracao no tempo + diagnostico de relampago em cada instante salvo
# ----------------------------------------------------------------------
TEMPO_TOTAL = 1800.0
DT = 2.0
historico = coluna.integrar(TEMPO_TOTAL, dt=DT, salvar_a_cada=60.0)

flash_rate_total = []
flash_rate_cg = []
flash_rate_ic = []

for idx in range(len(historico["t"])):
    T_c = np.array(historico["T"][idx]) - T0          # K ->  graus C
    qc_perfil = np.array(historico["qc"][idx])
    qs_perfil = np.array(historico["qs"][idx])
    qg_perfil = np.array(historico["qg"][idx])
    qi_perfil = np.array(historico["qi"][idx])

    # topo de nuvem diagnosticado: nivel mais alto com massa de
    # condensado (qc+qi+qs+qg) acima de um limiar desprezivel
    massa_condensado = qc_perfil + qi_perfil + qs_perfil + qg_perfil
    acima_limiar = np.where(massa_condensado > QMIN)[0]
    cldtop_km = coluna.z[acima_limiar[-1]] / 1000.0 if len(acima_limiar) > 0 else 0.0

    fr, fr_cg, fr_ic = driver_lightning(
        coluna.z, T_c, coluna.rho, w_prescrito,
        qg_perfil, qs_perfil, qc_perfil,
        dx_km=2.0, cu_active=False, cldtop_km=cldtop_km,
    )
    flash_rate_total.append(fr)
    flash_rate_cg.append(fr_cg)
    flash_rate_ic.append(fr_ic)

t_min = np.array(historico["t"]) / 60.0

print("Taxa de flash (flashes/min) ao longo da simulacao:")
for tm, fr in zip(t_min, flash_rate_total):
    print(f"  t={tm:5.1f} min  ->  FR_total={fr:.4f}")

# ----------------------------------------------------------------------
# 4) Grafico: taxa de flash x evolucao do gelo/neve/graupel
# ----------------------------------------------------------------------
qi_total = [np.sum(np.array(v) * coluna.rho) * coluna.dz for v in historico["qi"]]
qs_total = [np.sum(np.array(v) * coluna.rho) * coluna.dz for v in historico["qs"]]
qg_total = [np.sum(np.array(v) * coluna.rho) * coluna.dz for v in historico["qg"]]

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

axes[0].plot(t_min, flash_rate_total, label="Total (IC+CG)", color="black", lw=2)
axes[0].plot(t_min, flash_rate_cg, label="CG (nuvem-solo)", color="tab:red", ls="--")
axes[0].plot(t_min, flash_rate_ic, label="IC (intra-nuvem)", color="tab:blue", ls="--")
axes[0].set_ylabel("Taxa de flash (flashes/min)")
axes[0].set_title("Diagnostico de relampago (EP) acoplado ao Passo 3")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(t_min, qi_total, label="Gelo de nuvem (qi)", color="tab:cyan")
axes[1].plot(t_min, qs_total, label="Neve (qs)", color="tab:purple")
axes[1].plot(t_min, qg_total, label="Graupel (qg)", color="tab:gray")
axes[1].set_xlabel("Tempo (min)")
axes[1].set_ylabel("Massa total na coluna (kg/m$^2$)")
axes[1].set_title("Evolucao das categorias de gelo (microfisica)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./fig_passo3_relampago.png", dpi=130)
plt.close()

print("\nFigura salva: ./fig_passo3_relampago.png")

# ----------------------------------------------------------------------
# 5) Dados numericos (series de taxa de flash + gelo/neve/graupel), em ./
# ----------------------------------------------------------------------
np.savez_compressed(
    "./resultados_passo3_relampago.npz",
    t_s=np.array(historico["t"]),
    z_m=coluna.z,
    w_prescrito=w_prescrito,
    flash_rate_total=np.array(flash_rate_total),
    flash_rate_cg=np.array(flash_rate_cg),
    flash_rate_ic=np.array(flash_rate_ic),
    qi_total_kgm2=np.array(qi_total),
    qs_total_kgm2=np.array(qs_total),
    qg_total_kgm2=np.array(qg_total),
)

with open("./series_passo3_relampago.csv", "w") as f:
    f.write("tempo_min,flash_rate_total,flash_rate_cg,flash_rate_ic,"
            "qi_total_kgm2,qs_total_kgm2,qg_total_kgm2\n")
    for i in range(len(t_min)):
        f.write(f"{t_min[i]:.4f},{flash_rate_total[i]:.6e},{flash_rate_cg[i]:.6e},"
                f"{flash_rate_ic[i]:.6e},{qi_total[i]:.6e},{qs_total[i]:.6e},{qg_total[i]:.6e}\n")

print("\nDados salvos:")
print(" - ./resultados_passo3_relampago.npz")
print(" - ./series_passo3_relampago.csv")
