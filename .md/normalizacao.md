# Explicação das Técnicas de Normalização, Padronização e Transformação de Potência

## 1. Normalização (MinMaxScaler)

**Fórmula:**  
x_norm = (x - x_min) / (x_max - x_min)

**Exemplo:**  
Dados: [10, 20, 30, 40, 100]  
- x_min = 10  
- x_max = 100

Calcule para x=20:  
(20 - 10) / (100 - 10) = 10 / 90 ≈ 0,111

Calcule para x=40:  
(40 - 10) / (100 - 10) = 30 / 90 ≈ 0,333

Calcule para x=100:  
(100 - 10) / (100 - 10) = 90 / 90 = 1

---

## 2. Padronização (StandardScaler)

**Fórmula:**  
x_std = (x - média) / desvio_padrão

onde:  
- média é a média dos dados  
- desvio_padrão é o desvio padrão dos dados

**Exemplo:**  
Dados: [10, 20, 30, 40, 100]  
- Média = (10 + 20 + 30 + 40 + 100) / 5 = 40  
- Desvio padrão ≈ 32,4

Calcule para x=20:  
(20 - 40) / 32,4 ≈ -0,62

Calcule para x=100:  
(100 - 40) / 32,4 ≈ 1,85

---

## 3. Transformação de potência (PowerTransformer com Yeo-Johnson)

O Yeo-Johnson é uma transformação que “achata” a distribuição dos dados.

Para x ≥ 0, a fórmula é:

- Se λ ≠ 0:  
  T(x, λ) = ((x + 1)^λ - 1) / λ  
- Se λ = 0:  
  T(x, λ) = log(x + 1)

Para x < 0, a fórmula é:

- Se λ ≠ 2:  
  T(x, λ) = - [(-x + 1)^(2 - λ) - 1] / (2 - λ)  
- Se λ = 2:  
  T(x, λ) = - log(-x + 1)

**Exemplo simplificado:**  
Suponha x=3 e λ=0,5:

T(3, 0,5) = ((3 + 1)^0,5 - 1) / 0,5 = (2 - 1) / 0,5 = 2

---

## Resumo

| Técnica           | Fórmula principal                  | O que faz com o dado                         |
|-------------------|----------------------------------|----------------------------------------------|
| Normalização      | (x - x_min) / (x_max - x_min)    | Escala para intervalo [0,1]                   |
| Padronização      | (x - média) / desvio_padrão      | Centraliza e escala para média 0 e desvio 1  |
| Yeo-Johnson       | Fórmulas acima                   | Transforma para aproximar distribuição normal|

