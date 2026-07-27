# Modelo de Superficie  Solo Nu  Ciclo Diurno (didatico)

Implementacao modular do modelo simples de superficie apresentado nos
slides *MET-576-4  Conceitos / Surface Model*, seguindo **exatamente
a mesma sequencia** em que os processos foram expostos em aula.

## Estrutura dos modulos

| Arquivo | Slide correspondente | Conteudo |
|---|---|---|
| `modulo0_forcantes.py` | variaveis marcadas "(dado)" nos esquemas | SWd, LWd, Tr, er, Ur  ciclos diurnos prescritos |
| `modulo1_radiacao.py` | "Saldo de radiacao [Rn]" | Rn = (1-)SWd + LWd  T4 |
| `modulo2_fluxo_calor_solo.py` | "Fluxo de calor no solo [G]" | metodo force-restore: G = (CsD/d)(TTd) |
| `modulo3_resistencia_aerodinamica.py` | "Resistencia aerodinamica [ra]" | modelo de Verma-Rosenberg, CDN |
| `modulo4_fluxo_momentum.py` | "Fluxo de momentum []" |  = Ur/ra |
| `modulo5_calor_sensivel.py` | "Fluxo de calor sensivel [H]" | H = cp(TTr)/ra |
| `modulo6_calor_latente.py` | "Fluxo de calor latente  Caso II (solo umido)" | LE = (cp/)(hes(T)er)/(ra+rsoil) |
| `modulo7_balanco_hidrico.py` | "Balanco de agua do solo" | bucket model: Dd/dt = P  E  R |
| `modulo8_difusao_umidade_solo.py` | "Fluxo de calor no solo [G] - c. Modelo de umidade do solo" (Mahrt e Pan, 1984) | difusao multicamadas: d/dt = d/dz[D()d/dz] + dK()/dz |
| `modulo9_difusao_calor_solo.py` | "Fluxo de calor no solo [G] - Modelo de temperatura do solo" / "O modelo termodinamico de duas camadas de solo" | difusao multicamadas: C()dT/dt = d/dz[K_T()dT/dz], esquema implicito (algoritmo de Thomas) |
| `main_modelo_diurno.py` | "Hipoteses*" (fechamento do sistema, 1 camada) | integra os modulos 0-7 com force-restore |
| `main_modelo_multicamadas.py` | "Modelo com Varias Camadas de Solo" (conjunto de slides) | integra os modulos 0,1,3,4,5,6,8,9 com perfis T(z,t) e (z,t) |

## Dois modelos disponiveis

### 1) `main_modelo_diurno.py` - solo homogeneo (force-restore)
Uma unica camada de solo, G parametrizado por force-restore (modulo 2).
Mais simples, mais rapido, bom para entender o balanco de energia
"de cima para baixo".

### 2) `main_modelo_multicamadas.py` - solo em camadas (difusao) - NOVO
Varias camadas de solo (6, por padrao: 5, 10, 15, 25, 35, 50 cm),
com T e  evoluindo por DIFUSAO REAL (modulos 8 e 9), nao mais por
uma formula empirica de relaxacao. A temperatura da superficie (Ts)
deixa de ser a propria variavel prognostica e passa a ser
DIAGNOSTICADA a cada passo de tempo, resolvendo o balanco de energia
da superficie:

```
(1-alpha)*SWd + LWd - epsilon*sigma*Ts^4 = G(Ts) + H(Ts) + LE(Ts)
```

com G(Ts) = K_T*(Ts - T_camada0)/dz_tilde0 (fluxo difusivo, nao mais
force-restore). Essa equacao e nao linear (por causa de Ts^4 e de
es(Ts)) e e resolvida por um metodo de NEWTON simples (derivada
numerica, poucas iteracoes) a cada passo - o mesmo Ts entao vira a
condicao de contorno de Dirichlet do modulo 9.

Gera, alem do CSV/PNG do balanco de energia, dois MAPAS PROFUNDIDADE
x HORA (T(z,t) e (z,t)) - o equivalente numerico da figura "Soil
Temperature vs Depth, O'Neill Nebraska" mostrada no slide "Fluxo de
calor no solo [G]" (Hartmann, 1994).

### Calibracao importante (modulo 9)

A formula empirica de Al Nakshabandi e Kohnke (1965) citada no slide
gera, sem calibracao, uma condutividade termica K_T com faixa
dinamica MUITO maior que a observada em solos reais. Por isso
`condutividade_termica()` limita K_T ao intervalo [0,15; 0,7] W/mK
por padrao, calibrado para reproduzir, em SOLO NU, uma particao
G/Rn ~0,3 no pico solar (consistente com literatura de campo para
solo descoberto). Ajuste `K_T_min`/`K_T_max` se for representar
outro tipo de solo ou superficie.

## Como rodar

```bash
python3 main_modelo_diurno.py
```

Gera em `output/`:
- `ciclo_diurno_solo_nu.png`  4 paineis (forcantes/T, balanco de
  energia, umidade do solo/resistencias, momentum);
- `ciclo_diurno_ultimo_dia.csv`  serie temporal completa (5 em 5 min)
  do ultimo dia simulado, com todas as variaveis intermediarias.

## Metodo numerico

- Integracao explicita de Euler, `dt = 300 s`;
- Roda 5 dias (spin-up de 4 dias + 1 dia plotado), pois a condicao
  inicial de T e  e arbitraria  o spin-up garante que o ciclo
  diurno mostrado ja e o "regime estabelecido";
- `Td` (solo profundo) e mantido aproximadamente constante, coerente
  com a ideia de "force-restore": a camada profunda varia numa escala
  de tempo muito mais longa que 1 dia.

## Constantes usadas (slide "Hipoteses*  valores para solo nu")

```
 (albedo)        = 0,30
 (emissividade)  = 0,97
Cs                = 1e6 J K-1 m-3
d                = 1 dia / 2
s                = 0,50
z0                = 0,01 m
```

## Simplificacoes didaticas assumidas (documentadas no codigo)

1. **Estabilidade atmosferica**: `ra` calculado em condicoes neutras
   (CD = CDN), sem correcao de Richardson  consistente com a
   hipotese explicita do slide.
2. **Umidade relativa do solo `h()`**: usada a relacao linear
   `h = /s` em vez da curva completa `h = exp[g/(RvT)]`, que
   exigiria parametros de textura do solo (Clapp & Hornberger, 1978)
   nao fornecidos no exercicio.
3. **LWd**: parametrizado como `atmTr4` (nao e um dado observado
   real, apenas uma forma de fechar o ciclo diurno de forma fisicamente
   razoavel).
4. **Sem precipitacao** neste experimento  o solo so perde agua por
   evaporacao, mostrando o feedback "seca  menos LE  mais H  T
   mais alta" ao longo dos dias de spin-up.

## Proximos passos sugeridos (para expandir o modelo)

- Acoplar o **modulo com vegetacao** (interceptacao, transpiracao,
  resistencia estomatica, resistencia da copa `rc`, IAF)  slides
  "BALANCO DE AGUA COM VEGETACAO" em diante;
- Trocar o esquema de integracao por Runge-Kutta ou Crank-Nicolson
  (como no modelo de duas camadas de solo do slide "Fluxo de calor no
  solo [G]  modelo termodinamico de duas camadas");
- Incluir correcao de estabilidade (Louis, 1979) no calculo de `ra`
  e do coeficiente de troca `Ch`, como detalhado no slide "Modelo da
  Camada Superficial".
