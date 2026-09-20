# Detecção de Anomalias em Transações (Detecção de Fraude)

Este projeto foi desenvolvido como parte do desafio "Detecção de Anomalias em Transações em Python" da DIO. O objetivo principal é construir modelos de Machine Learning capazes de identificar transações fraudulentas em uma base de dados de cartões de crédito altamente desbalanceada.

##  O que foi implementado e melhorado

Partindo das instruções básicas do projeto, o código foi refatorado e expandido para incluir as melhores práticas de Ciência de Dados, com foco especial em evitar vazamento de dados (*data leakage*) e melhorar a explicabilidade do modelo.

### Principais Técnicas Aplicadas:
* **Prevenção de Data Leakage:** Separação estrita dos dados em treino e teste (`train_test_split` com `stratify`) **antes** da aplicação de qualquer técnica de escalonamento ou balanceamento.
* **Feature Engineering e Escalonamento:** Uso do `StandardScaler` apenas nas variáveis `Time` e `Amount` (as demais já estavam transformadas via PCA), ajustando o scaler exclusivamente nos dados de treino.
* **Balanceamento de Dados (SMOTE):** Aplicação de *Oversampling* gerando dados sintéticos da classe minoritária apenas na base de treino, garantindo que os dados de teste permaneçam reais e inalterados.
* **Modelagem e Comparação:**
  1. **Regressão Logística** (Modelo Baseline)
  2. **Random Forest** (Treinado com dados balanceados via SMOTE)
  3. **XGBoost** (Com `scale_pos_weight` calculado dinamicamente para lidar com o desbalanceamento nativamente).
* **Threshold Tuning (Ajuste de Limiar):** Ajuste do limiar de decisão (*custom threshold* = 0.3) no XGBoost para priorizar o *Recall*, métrica crucial em detecção de fraudes.
* **Otimização de Hiperparâmetros:** Uso de `GridSearchCV` para encontrar a melhor combinação de parâmetros para o modelo XGBoost.
* **Explicabilidade (XAI):** Integração com a biblioteca `SHAP` para visualizar o impacto de cada variável na decisão final do modelo.

##  Tecnologias Utilizadas
* **Linguagem:** Python
* **Manipulação e Análise de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
* **Visualização e Explicabilidade:** Matplotlib, SHAP

##  Como executar

1. Clone o repositório.
2. Instale as dependências necessárias:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib
