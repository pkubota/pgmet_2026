
# Teoria do Método Semi-Lagrangiano para Equações de Advecção

## Introdução

O Método Semi-Lagrangiano (MSL) é uma técnica numérica amplamente utilizada na modelagem atmosférica e oceanográfica para resolver equações de transporte, como a equação de advecção. Ele combina características dos métodos Eulerianos e Lagrangianos para superar as limitações de cada abordagem individualmente, oferecendo maior estabilidade numérica e permitindo passos de tempo maiores do que os métodos puramente Eulerianos.

## Princípios Fundamentais

A ideia central do MSL é rastrear as parcelas de fluido para trás no tempo, a partir de um ponto de grade fixo no instante atual, até sua posição de partida no instante anterior. A concentração (ou outra variável transportada) no ponto de grade atual é então determinada pela interpolação dos valores da variável na posição de partida. Isso contrasta com os métodos Eulerianos, que calculam as mudanças na variável em um ponto de grade fixo com base nos fluxos através das fronteiras da célula.

### Derivada Material

A equação de advecção pode ser expressa em termos da derivada material (ou derivada Lagrangiana), que representa a taxa de variação de uma propriedade de uma parcela de fluido à medida que ela se move com o fluxo. Para uma variável escalar C, a derivada material é dada por:

```
DC/Dt = ∂C/∂t + u ⋅ ∇C
```

Onde `u` é o vetor velocidade do fluxo e `∇C` é o gradiente de C. O MSL discretiza essa derivada material diretamente, o que contribui para sua estabilidade.

### Rastreamento de Trajetórias

No MSL, para determinar o valor de C em um ponto de grade (x, z) no tempo t + Δt, é necessário encontrar a posição de partida (x_origem, z_origem) da parcela de fluido que chegará a (x, z) em t + Δt. Essa posição de partida é calculada retrocedendo a trajetória da parcela no tempo, usando a velocidade do fluxo. A equação de trajetória pode ser aproximada por:

```
x_origem = x - u * Δt
z_origem = z - w * Δt
```

Onde `u` e `w` são as componentes da velocidade nas direções x e z, respectivamente, e Δt é o passo de tempo. Em implementações mais sofisticadas, a velocidade pode ser avaliada em um ponto médio da trajetória ou um esquema iterativo pode ser usado para maior precisão.

### Interpolação

Uma vez que a posição de partida (x_origem, z_origem) geralmente não coincide com um ponto de grade, é necessária uma interpolação para estimar o valor de C nessa posição a partir dos valores conhecidos nos pontos de grade vizinhos no tempo t. Métodos de interpolação comuns incluem interpolação linear, bilineares, cúbicas ou cúbicas de Lagrange. A escolha do esquema de interpolação afeta a precisão e a conservação do método.

## Vantagens do Método Semi-Lagrangiano

*   **Estabilidade Incondicional:** Uma das maiores vantagens do MSL é sua estabilidade incondicional em relação ao termo de advecção, o que significa que ele não é limitado pelo critério de Courant-Friedrichs-Lewy (CFL). Isso permite o uso de passos de tempo maiores, reduzindo o custo computacional para simulações de longo prazo.
*   **Precisão:** Com esquemas de interpolação de ordem superior, o MSL pode alcançar alta precisão na representação do transporte advectivo.
*   **Manutenção de Gradientes:** O MSL é conhecido por sua capacidade de manter gradientes acentuados e evitar a difusão numérica excessiva que pode ocorrer em métodos Eulerianos de baixa ordem.

## Desvantagens e Desafios

*   **Conservação:** A conservação de massa ou outras propriedades pode ser um desafio no MSL, especialmente com esquemas de interpolação de baixa ordem. Métodos de interpolação conservativos ou correções pós-processamento podem ser necessários.
*   **Custo Computacional da Interpolação:** A interpolação em cada passo de tempo pode ser computacionalmente intensiva, especialmente em dimensões mais altas e com esquemas de interpolação de ordem superior.
*   **Determinação da Posição de Partida:** O cálculo preciso da posição de partida pode ser complexo, especialmente em campos de velocidade variáveis no tempo e no espaço.

## Aplicação na Equação de Advecção-Difusão

Para equações que incluem termos de difusão e/ou fontes/sumidouros, o MSL é frequentemente aplicado apenas ao termo de advecção. Os outros termos são então tratados por métodos Eulerianos, resultando em um esquema híbrido. Por exemplo, o termo de difusão pode ser discretizado implicitamente para manter a estabilidade geral do esquema.

## Referências

[1] Meneses, L. R., & Almeida, R. C. (2016). Modelo Semi-Lagrangeano de Dispersão Atmosférica - Avaliação. Ciência e Natura, 38(Ed. Especial), 418–425. Disponível em: [https://periodicos.ufsm.br/cienciaenatura/article/download/20307/pdf/110145](https://periodicos.ufsm.br/cienciaenatura/article/download/20307/pdf/110145)


