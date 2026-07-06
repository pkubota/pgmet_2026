# Guia de Uso: Equação de Advecção com Método Semi-Lagrangiano em Fortran

Este documento fornece um guia detalhado para compilar, executar e entender o código Fortran que implementa a solução da equação de advecção utilizando o Método Semi-Lagrangiano.

## 1. Visão Geral do Código

O programa `advection_semi_lagrangian.f90` simula o transporte de uma quantidade escalar (concentração) em uma dimensão, utilizando o Método Semi-Lagrangiano. A principal característica deste método é sua estabilidade numérica, permitindo passos de tempo maiores em comparação com métodos Eulerianos tradicionais. O código utiliza uma interpolação linear simples para determinar a concentração na posição de partida, mas em aplicações mais complexas, interpolações de ordem superior (e.g., cúbica) seriam mais apropriadas para maior precisão e conservação.

### Parâmetros Configuráveis:

Os seguintes parâmetros podem ser ajustados no início do arquivo `advection_semi_lagrangian.f90` para modificar o comportamento da simulação:

*   `nx`: Número de pontos na grade espacial (padrão: 100).
*   `dx`: Espaçamento da grade espacial (padrão: 1.0).
*   `dt`: Passo de tempo (padrão: 0.1). Note que, devido à estabilidade incondicional do método Semi-Lagrangiano para advecção pura, `dt` pode ser maior que o limite de Courant-Friedrichs-Lewy (CFL) de métodos explícitos.
*   `u`: Velocidade de advecção (padrão: 1.0).
*   `nsteps`: Número de passos de tempo da simulação (padrão: 100).

## 2. Compilação do Código

Para compilar o código Fortran, você precisará de um compilador Fortran, como o `gfortran` (GNU Fortran Compiler), que é amplamente disponível e de código aberto.

### Pré-requisitos:

Certifique-se de que o `gfortran` esteja instalado em seu sistema. Se não estiver, você pode instalá-lo em sistemas baseados em Debian/Ubuntu com o seguinte comando:

```bash
sudo apt-get update
sudo apt-get install gfortran
```

### Processo de Compilação:

Navegue até o diretório onde o arquivo `advection_semi_lagrangian.f90` está salvo e execute o seguinte comando no terminal:

```bash
gfortran advection_semi_lagrangian.f90 -o advection_semi_lagrangian
```

*   `gfortran`: Invoca o compilador Fortran.
*   `advection_semi_lagrangian.f90`: É o arquivo de código fonte Fortran.
*   `-o advection_semi_lagrangian`: Especifica o nome do arquivo executável de saída. Você pode escolher qualquer nome para o executável.

Se a compilação for bem-sucedida, um arquivo executável chamado `advection_semi_lagrangian` (ou o nome que você especificou) será criado no mesmo diretório.

## 3. Execução do Programa

Após a compilação, você pode executar o programa a partir do terminal:

```bash
./advection_semi_lagrangian
```

O programa imprimirá a concentração final em cada ponto da grade espacial no terminal. Para salvar a saída em um arquivo de texto, você pode redirecionar a saída padrão:

```bash
./advection_semi_lagrangian > advection_output.txt
```

Isso criará um arquivo chamado `advection_output.txt` contendo os resultados da simulação.

## 4. Exemplo de Saída

A saída do programa consiste em duas colunas de dados: a primeira coluna representa a coordenada espacial `x`, e a segunda coluna representa a concentração `C` naquele ponto após `nsteps` passos de tempo. Abaixo está um exemplo de como a saída pode se parecer (os valores exatos dependerão dos parâmetros e da função inicial):

```
Simulação concluída.
Concentração final:
0.00000000       0.00000000
1.00000000       0.00000000
2.00000000       0.00000000
...
49.0000000       0.00000000
50.0000000       0.00000000
51.0000000       0.00000000
...
99.0000000       0.00000000
```

Note que a saída completa terá `nx` linhas de dados, correspondendo a cada ponto da grade. A função inicial gaussiana se moverá para a direita (assumindo `u > 0`), e a forma da onda será preservada (ou levemente difundida, dependendo do esquema de interpolação).

## 5. Visualização dos Resultados

Para uma melhor compreensão dos resultados, é altamente recomendável plotar os dados de saída. Você pode usar ferramentas como `matplotlib` em Python, `gnuplot`, ou planilhas eletrônicas para visualizar a distribuição da concentração ao longo do espaço. Por exemplo, em Python, você pode ler o arquivo `advection_output.txt` e plotar os dados:

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("advection_output.txt", skiprows=2) # Ignora as duas primeiras linhas de texto
x = data[:, 0]
C = data[:, 1]

plt.plot(x, C)
plt.xlabel("Posição (x)")
plt.ylabel("Concentração (C)")
plt.title("Distribuição de Concentração após Advecção")
plt.grid(True)
plt.show()
```

Este guia deve ser suficiente para que você possa compilar, executar e analisar os resultados do exemplo da equação de advecção com o Método Semi-Lagrangiano em Fortran.


