"""
ETAPA 8a - Extracao de uma coluna real do ERA5 (auxiliar da Etapa 8)
========================================================================

Este script extrai UMA coluna atmosferica (por padrao, 45N) do arquivo de
entrada do tutorial pratico do ecRad (era5slice.nc: 361 colunas x 137
niveis, corte meridional do ERA5) e as saidas REAIS do ecRad ja executado
sobre esse arquivo (era5slice_out.nc), salvando tudo num unico .npz que o
`etapa8_benchmark.py` consome.

PRE-REQUISITO: nao gera esses dois arquivos .nc sozinho. Eles vem de:
  1. o ecRad compilado a partir do codigo-fonte oficial do ECMWF (ver
     Capitulo 6 do relatorio -- inclui a correcao de um bug de compilacao
     encontrado em radiation_interface.F90 e a obtencao dos dados
     espectrais que vinham vazios no pacote distribuido);
  2. rodar, dentro do diretorio practical/ do ecRad compilado:
         ./ecrad config.nam era5slice.nc era5slice_out.nc
     (era5slice.nc ja vem no tutorial pratico do ecRad, em practical/).

COMO EXECUTAR
--------------
    python3 etapa8a_extrai_coluna_era5.py

Rode a partir do diretorio practical/ do ecRad (onde estao era5slice.nc e
era5slice_out.nc), ou ajuste os caminhos no bloco `if __name__ ==
"__main__":` abaixo. Requer o pacote `netCDF4` (`pip install netCDF4`).

Para mudar a coluna extraida, edite `LATITUDE_ALVO` (usa a coluna mais
proxima dessa latitude).

SAIDA: um arquivo `coluna_45N.npz` (ou nome correspondente a latitude
escolhida) com os perfis de entrada (pressao, temperatura, umidade,
ozonio, CO2) e as saidas reais do ecRad (OLR, fluxos de superficie,
perfil completo de fluxo LW ascendente) para essa coluna -- consumido
pelo `etapa8_benchmark.py`.
"""

import numpy as np
import netCDF4 as nc

LATITUDE_ALVO = 45.0  # graus; a coluna mais proxima desta latitude e usada
ARQUIVO_ENTRADA = "era5slice.nc"
ARQUIVO_SAIDA_ECRAD = "era5slice_out.nc"
ARQUIVO_NPZ = "/home/pkubota/coluna_45N.npz"


def extrai_coluna(arquivo_entrada, arquivo_saida_ecrad, latitude_alvo, arquivo_npz):
    ds_in = nc.Dataset(arquivo_entrada)
    ds_out = nc.Dataset(arquivo_saida_ecrad)

    lat = ds_in.variables['latitude'][:]
    icol = int(np.argmin(np.abs(lat - latitude_alvo)))

    p_hl = ds_in.variables['pressure_hl'][icol, :] / 100.0  # Pa -> hPa
    T_hl = ds_in.variables['temperature_hl'][icol, :]
    q = ds_in.variables['q'][icol, :]           # umidade especifica, kg/kg
    o3 = ds_in.variables['o3_mmr'][icol, :]      # razao de mistura de massa de O3
    co2 = ds_in.variables['co2_vmr'][icol, :]    # razao de mistura de volume
    Ts_skin = float(ds_in.variables['skin_temperature'][icol])
    albedo = float(ds_in.variables['sw_albedo'][icol])
    emiss = float(ds_in.variables['lw_emissivity'][icol])
    coszen = float(ds_in.variables['cos_solar_zenith_angle'][icol])

    OLR_real = float(ds_out.variables['flux_up_lw_clear'][icol, 0])
    LWdn_sfc_real = float(ds_out.variables['flux_dn_lw_clear'][icol, -1])
    SWdn_sfc_real = float(ds_out.variables['flux_dn_sw_clear'][icol, -1])
    SWup_toa_real = float(ds_out.variables['flux_up_sw_clear'][icol, 0])
    SWdn_toa_real = float(ds_out.variables['flux_dn_sw_clear'][icol, 0])
    flux_up_lw_clear = ds_out.variables['flux_up_lw_clear'][icol, :]
    flux_dn_lw_clear = ds_out.variables['flux_dn_lw_clear'][icol, :]

    print(f"Coluna {icol} (lat={lat[icol]:.1f}): {len(p_hl)} niveis")
    print(f"Ts (skin) = {Ts_skin:.2f} K, albedo SW = {albedo:.3f}, "
          f"emissividade LW = {emiss:.3f}")
    print(f"OLR real = {OLR_real:.2f} W/m2, LW descendente real na "
          f"superficie = {LWdn_sfc_real:.2f} W/m2")

    np.savez(arquivo_npz, p_hl=p_hl, T_hl=T_hl, q=q, o3=o3, co2=co2,
             Ts_skin=Ts_skin, albedo=albedo, emiss=emiss, coszen=coszen,
             OLR_real=OLR_real, LWdn_sfc_real=LWdn_sfc_real,
             SWdn_sfc_real=SWdn_sfc_real, SWup_toa_real=SWup_toa_real,
             SWdn_toa_real=SWdn_toa_real,
             flux_up_lw_clear=flux_up_lw_clear,
             flux_dn_lw_clear=flux_dn_lw_clear)
    print(f"Salvo em {arquivo_npz}")
    return arquivo_npz


if __name__ == "__main__":
    extrai_coluna(ARQUIVO_ENTRADA, ARQUIVO_SAIDA_ECRAD, LATITUDE_ALVO, ARQUIVO_NPZ)
