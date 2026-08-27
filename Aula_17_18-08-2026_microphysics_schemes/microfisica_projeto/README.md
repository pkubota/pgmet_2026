# Modelo 2D de Nuvem Convectiva + Esquema de Microfisica Thompson Completo

Este pacote acopla o esquema de microfisica de 6 categorias (Passos 1-3
do projeto `microfisica_project`: qv, qc+Nc, qr+Nr, qi+Ni, qs+Ns,
qg+Ng, com Pccnd, Pccnr, Pracw, Pidsn, Pidep, Pifzc, Pgfzr, riming,
Hallett-Mossop, Picns, degelo, etc.) ao modelo 2D explicito de nuvem
convectiva (vorticidade-funcao de corrente, aproximacao de Boussinesq
-- Equacao 2.6 do relatorio "Modelo Conceitual de Fluxo de Massa para a
Transicao Convectiva Rasa-Profunda") fornecido pelo curso de Conveccao
Atmosferica.

## O que mudou em relacao ao `nuvem_2d.py` original

| | `nuvem_2d.py` (original) | `nuvem_2d_thompson.py` (este pacote) |
|---|---|---|
| Categorias de agua | qc, qi, qr, qsnow (1 momento) | qc, qr, qi, qs, qg (2 momentos: Nc,Nr,Ni,Ns,Ng) |
| Condensacao/evaporacao | ajuste de saturacao simplificado | Pccnd (ajuste de saturacao + particao WBF via ordem dos processos) |
| Autoconversao | limiar unico (tipo Tiedtke) | Pccnr, Pracw (Khairoutdinov e Kogan 2000) + Picns (gelo->neve) |
| Nucleacao de gelo | particao linear por temperatura (ice_fraction) | Pidsn (Cooper 1986) |
| Congelamento | nao ha (so particao liquido/gelo por T) | Pifzc (gotas, Bigg 1953), Pgfzr (chuva->graupel, Bigg 1953) |
| Riming | nao ha | Pi_iacw, Ps_sacw, Pgacw (coleta continua) |
| Multiplicacao de gelo | nao ha | Pispl (Hallett-Mossop) |
| Velocidade terminal | constantes fixas (VT_RAIN=5, VT_SNOW=1.5 m/s) | diagnosticada da distribuicao gama (q,N) -- varia com o tamanho real das particulas |
| Graupel | nao existe | categoria completa (qg,Ng) |

O nucleo dinamico (Equacao 2.6: vorticidade-funcao de corrente) **nao
muda**. A microfisica entra em exatamente dois pontos, como no modelo
original:
1. Fontes/sumidouros de agua e calor latente (agora calculados por
   `microfisica/coluna_generica.py::passo_microfisica_coluna()`);
2. O termo de empuxo `thv`, que agora soma todas as 5 categorias de
   agua condensada (qc+qi+qr+qs+qg).

## Estrutura do pacote

```
nuvem2d_thompson/
|-- nuvem_2d_thompson.py     # script principal (roda o modelo acoplado)
|-- nuvem_2d.py               # copia do script ORIGINAL (microfisica simples,
|                              # 1 momento) -- mantido como referencia/baseline
|                              # para comparacao direta com nuvem_2d_thompson.py
`-- microfisica/
    |-- coluna_generica.py    # funcao passo_microfisica_coluna() --
    |                          # mesma fisica do Passo 3, empacotada
    |                          # para uso "standalone" por coluna
    |-- constantes.py
    |-- distribuicoes.py
    |-- processos_chuva_quente.py
    |-- processos_fase_gelo.py
    |-- processos_fase_mista.py
    `-- coluna_step1/2/3.py    # nao usados diretamente aqui, mas
                                # mantidos para referencia/compatibilidade
```

## Como rodar

```bash
# teste rapido (poucos minutos simulados, para verificar que roda)
python3 nuvem_2d_thompson.py --bolha 5.0 --tempo 10 --cenario teste

# cenario raso
python3 nuvem_2d_thompson.py --bolha 2.5 --cenario rasa --tempo 30

# cenario profundo (mais tempo simulado para a nuvem glaciar de verdade)
python3 nuvem_2d_thompson.py --bolha 7.0 --cenario profunda --tempo 40
```

Flags disponiveis (identicas ao `nuvem_2d.py` original, mais `--nx`/`--nz`):
`--bolha`, `--cenario`, `--microfisica {nenhuma,thompson}`,
`--evap-chuva {on,off}`, `--radiacao {on,off}`, `--ciclo-diurno {on,off}`,
`--tempo`, `--nx`, `--nz`.

## Custo computacional (leia antes de rodar o cenario completo)

A fisica de reacao (10 grupos de processos com varias ramificacoes:
Pidsn, Pidep/Psdep/Pgdep, Pccnd, Pifzc/Pgfzr, riming, Pispl, coleta de
chuva, Picns, degelo, chuva quente) e resolvida com um loop Python
explicito por **coluna** (nao vetorizado com NumPy), porque as
ramificacoes condicionais de cada processo (limiares de temperatura,
clipping de massa, etc.) sao dificeis de vetorizar sem reescrever cada
funcao de processo do zero. Isso foi uma escolha deliberada
(fidelidade fisica > velocidade). Como referencia de custo, medido
neste ambiente:

- Grade padrao (nx=90, nz=110), dt=1.5s: **~8 segundos de CPU por
  minuto simulado**.
- Um cenario de 40-60 min simulados leva, portanto, ~5-8 minutos de
  CPU.
- O modo `--ciclo-diurno on` (12h simuladas por padrao) levaria
  **~1h40 de CPU** -- rode com `--tempo` reduzido (ex.: `--tempo 300`
  para simular so das 6h as 11h) para testes.

A sedimentacao (velocidade terminal de queda de qr,qi,qs,qg) **e**
vetorizada com NumPy (formulas algebricas puras, sem ramificacao), e
por isso nao contribui de forma relevante para o custo.

## Observacao sobre os testes realizados

Nos testes de verificacao feitos durante o desenvolvimento (bolha de
7K, 20 min simulados), a nuvem atingiu ~2600 m de altura e comecou a
dissipar antes de atingir o nivel de congelamento -- ou seja, o
graupel ainda nao se formou nesse cenario especifico dentro do tempo
simulado (comportamento consistente com o relatorio original, que
tambem reporta que fase gelo e downdraft "ficam dormentes" com a
calibracao padrao). Para observar glaciacao e formacao de graupel,
recomenda-se aumentar `--bolha` (ex.: 8-10 K) e/ou `--tempo` (40-60
min), reproduzindo o teste sugerido no proprio relatorio original
(Secao 4.2: aumentar SHF_max/LHF_max ou a amplitude da termica).
