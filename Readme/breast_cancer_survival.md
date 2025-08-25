# Projeto de Modelo Preditivo de Sobrevivência - Haberman Dataset

## Objetivo

Desenvolver um modelo de machine learning para prever a sobrevivência de pacientes submetidos a cirurgia de câncer de mama, usando o conjunto de dados Haberman. O foco é criar um modelo confiável, interpretável e com boa capacidade preditiva.

## Dados

- **Fonte:** Haberman Dataset
- **Características:**
  - Idade do paciente na cirurgia
  - Ano da cirurgia
  - Número de linfonodos positivos detectados
- **Classe alvo:** Sobrevivência por 5 anos ou mais (0) ou menos que 5 anos (1)
- **Distribuição das classes:**  
  - Classe 0: 73,53%  
  - Classe 1: 26,47% (problema desbalanceado)

## Metodologia

1. **Pré-processamento:**
   - Codificação das classes com LabelEncoder
   - Análise da distribuição das classes
   - Tratamento de dados desbalanceados e avaliação com métricas adequadas

2. **Modelagem:**
   - Teste de vários modelos probabilísticos:  
     Regressão Logística, LDA, QDA, Naive Bayes, Gaussian Process
   - Uso de validação cruzada estratificada para avaliação robusta
   - Métrica principal: Brier Skill Score (BSS), que compara o modelo com uma baseline ingênua

3. **Transformações aplicadas:**
   - Escalonamento dos dados com StandardScaler
   - Transformação de potência com PowerTransformer e MinMaxScaler para aproximar distribuição normal
   - Uso de Pipeline para garantir que transformações sejam aplicadas corretamente sem vazamento

4. **Avaliação:**
   - Comparação do desempenho dos modelos com e sem transformações
   - Escolha do modelo final baseado no melhor BSS (Regressão Logística com transformações)
   - Interpretação dos resultados e análise da confiança do modelo

5. **Previsões:**
   - Treinamento do modelo final com todo o conjunto de dados
   - Uso do pipeline para fazer previsões probabilísticas em novos dados
   - Interpretação das probabilidades como confiança nas previsões

## Resultados

- Baseline (chute burro) Brier Score: 19,46%
- Melhor modelo (Regressão Logística com transformações) Brier Skill Score: ~0,10 (10% melhor que baseline)
- Modelos com transformações de potência apresentaram melhora significativa no desempenho
- Modelo final fornece probabilidades confiáveis para cada nova previsão

## Como usar

1. Treine o modelo com o conjunto de dados completo usando o pipeline.
2. Faça previsões probabilísticas para novas amostras.
3. Interprete a probabilidade da classe positiva como a confiança na previsão.

## Próximos passos

- Explorar outras métricas complementares (AUC-ROC, precisão, recall)
- Testar técnicas para lidar com desbalanceamento (SMOTE, ajuste de pesos)
- Avaliar calibração do modelo para melhorar a confiança nas probabilidades
- Criar visualizações e animações para facilitar o entendimento dos conceitos (ex: com Manim)

---

Este projeto foi desenvolvido com foco em clareza, simplicidade e didática, para facilitar o aprendizado e aplicação prática em problemas reais de classificação com dados desbalanceados.

---

A fonte do projeto é o livro **Imbalanced Classification with Python Choose Better Metrics, Balance Skewed Classes, and Apply Cost-Sensitive Learning. Machine Learning Mastery. Jason Brownlee, 2021**
