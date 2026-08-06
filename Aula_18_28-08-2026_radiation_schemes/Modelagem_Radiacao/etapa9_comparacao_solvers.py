"""
ETAPA 9 - Comparacao dos solvers de nuvem McICA, Tripleclouds e SPARTACUS
=============================================================================

Este script gera 3 variantes de configuracao do ecRad (mudando so o
solver de nuvem), roda o ecRad real para cada uma sobre o mesmo arquivo
de entrada, escolhe uma coluna com nebulosidade fracionaria interessante,
e compara os fluxos LW/SW resultantes entre os 3 solvers.

PRE-REQUISITO: assim como a Etapa 8, este script precisa do executavel
`ecrad` ja compilado a partir do codigo-fonte oficial do ECMWF (ver
Capitulo 6 do relatorio) e do arquivo de entrada `era5slice.nc` (tutorial
pratico do ecRad). NAO compila o ecRad nem baixa dados sozinho.

COMO EXECUTAR
--------------
    python3 etapa9_comparacao_solvers.py

Rode a partir do diretorio practical/ do ecRad compilado (onde estao o
executavel `ecrad`, o `config.nam` base, e `era5slice.nc`), ou ajuste os
caminhos nas constantes no topo do arquivo. Requer o pacote `netCDF4` e o
script auxiliar `change_namelist.sh` do proprio ecRad (em
test/common/change_namelist.sh no codigo-fonte).

Para comparar outros solvers ou trocar a coluna escolhida, edite
`SOLVERS` e os criterios de selecao de coluna em `escolhe_coluna_nublada`.

SAIDA: tabela no terminal com OLR, SW refletido no topo e SW/LW na
superficie para cada solver, mais a figura
`etapa9_comparacao_solvers.png`.
"""

import subprocess
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

CHANGENAM = "../test/common/change_namelist.sh"
CONFIG_BASE = "config.nam"
ARQUIVO_ENTRADA = "era5slice.nc"
SOLVERS = ["McICA", "Tripleclouds", "SPARTACUS"]


def gera_configs_solver(solvers, config_base=CONFIG_BASE, changenam=CHANGENAM):
    """Gera um config_<solver>.nam por solver, mudando sw_solver_name e
    lw_solver_name (usa o script change_namelist.sh do proprio ecRad)."""
    arquivos = {}
    for s in solvers:
        saida = f"config_{s}.nam"
        subprocess.run([changenam, config_base, saida,
                         f'sw_solver_name="{s}"', f'lw_solver_name="{s}"'],
                        check=True)
        arquivos[s] = saida
    return arquivos


def roda_ecrad_para_cada_solver(solvers, arquivo_entrada, configs, executavel="./ecrad"):
    saidas = {}
    for s in solvers:
        saida = f"era5slice_{s}_out.nc"
        print(f"Rodando ecRad com solver {s}...")
        subprocess.run([executavel, configs[s], arquivo_entrada, saida],
                        check=True, capture_output=True)
        saidas[s] = saida
    return saidas


def escolhe_coluna_nublada(arquivo_entrada, frac_min=0.15, frac_max=0.6,
                            n_camadas_min=5, n_camadas_max=40):
    """Escolhe uma coluna com nebulosidade fracionaria (nem ceu limpo, nem
    totalmente encoberto) para tornar a diferenca entre solvers visivel."""
    ds = nc.Dataset(arquivo_entrada)
    cf = ds.variables['cloud_fraction'][:]
    lat = ds.variables['latitude'][:]
    frac_media = cf.mean(axis=1)
    n_nubladas = (cf > 0.05).sum(axis=1)
    candidatos = np.where((frac_media > frac_min) & (frac_media < frac_max) &
                           (n_nubladas > n_camadas_min) & (n_nubladas < n_camadas_max))[0]
    if len(candidatos) == 0:
        raise RuntimeError("Nenhuma coluna candidata encontrada com esses criterios.")
    icol = int(candidatos[len(candidatos) // 2])
    return icol, float(lat[icol]), cf[icol, :]


def compara_solvers(solvers, saidas, icol):
    resultados = {}
    for s in solvers:
        ds = nc.Dataset(saidas[s])
        resultados[s] = dict(
            OLR=float(ds.variables['flux_up_lw'][icol, 0]),
            SWup_toa=float(ds.variables['flux_up_sw'][icol, 0]),
            SWdn_sfc=float(ds.variables['flux_dn_sw'][icol, -1]),
            LWdn_sfc=float(ds.variables['flux_dn_lw'][icol, -1]),
            flux_up_lw=ds.variables['flux_up_lw'][icol, :],
            flux_dn_sw=ds.variables['flux_dn_sw'][icol, :],
        )
    return resultados


if __name__ == "__main__":
    configs = gera_configs_solver(SOLVERS)
    saidas = roda_ecrad_para_cada_solver(SOLVERS, ARQUIVO_ENTRADA, configs)
    icol, lat_col, cf_col = escolhe_coluna_nublada(ARQUIVO_ENTRADA)
    print(f"\nColuna escolhida: {icol} (lat={lat_col:.1f})\n")

    resultados = compara_solvers(SOLVERS, saidas, icol)

    print(f"{'Solver':15s}{'OLR':>10s}{'SWup_toa':>10s}{'SWdn_sfc':>10s}{'LWdn_sfc':>10s}")
    for s in SOLVERS:
        r = resultados[s]
        print(f"{s:15s}{r['OLR']:10.2f}{r['SWup_toa']:10.2f}"
              f"{r['SWdn_sfc']:10.2f}{r['LWdn_sfc']:10.2f}")

    ds_in = nc.Dataset(ARQUIVO_ENTRADA)
    p_hl = ds_in.variables['pressure_hl'][icol, :] / 100.0
    p_centros = 0.5 * (p_hl[:-1] + p_hl[1:])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    cores = ["tab:blue", "tab:orange", "tab:green"]

    ax = axes[0]
    ax.plot(cf_col, p_centros, color="k")
    ax.fill_betweenx(p_centros, 0, cf_col, alpha=0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Fracao de nuvem")
    ax.set_ylabel("Pressao (hPa)")
    ax.set_title(f"Perfil de nebulosidade\ncoluna lat={lat_col:.0f}")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for s, c in zip(SOLVERS, cores):
        ax.plot(resultados[s]['flux_up_lw'], p_hl, label=s, color=c)
    ax.invert_yaxis()
    ax.set_xlabel("Fluxo LW ascendente (W/m2)")
    ax.set_ylabel("Pressao (hPa)")
    ax.set_title("LW: solvers quase coincidem")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for s, c in zip(SOLVERS, cores):
        ax.plot(resultados[s]['flux_dn_sw'], p_hl, label=s, color=c)
    ax.invert_yaxis()
    ax.set_xlabel("Fluxo SW descendente (W/m2)")
    ax.set_ylabel("Pressao (hPa)")
    ax.set_title("SW: SPARTACUS diverge\n(efeitos 3D de nuvem quebrada)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("./etapa9_comparacao_solvers.png", dpi=150)
    print("\nFigura salva em etapa9_comparacao_solvers.png")
