import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, roc_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

# 1. Carga dos Dados
url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
df = pd.read_csv(url)

# 2. Separação de Features e Target (Trata divergência de caixa no nome da coluna)
target_col = 'Class' if 'Class' in df.columns else 'class'
X = df.drop(columns=[target_col])
y = df[target_col]

# Separar dados de treino e teste ANTES de qualquer transformação (Evita Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Engenharia de Features / Escalonamento
# Apenas 'Time' e 'Amount' precisam de escalonamento; V1-V28 já são resultado de PCA
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# 4. Experimento 1: Regressão Logística (Modelo Baseline)
lr_base = LogisticRegression(max_iter=1000, random_state=42)
lr_base.fit(X_train_scaled, y_train)
y_pred_base = lr_base.predict(X_test_scaled)

print("=== 1. Regressão Logística (Baseline) ===")
print(classification_report(y_test, y_pred_base))

# 5. Experimento 2: SMOTE no Treino + Random Forest
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test_scaled)

print("=== 2. Random Forest com SMOTE ===")
print(classification_report(y_test, y_pred_rf))

# 6. Experimento 3: XGBoost com Peso de Classe Dinâmico e Limiar Ajustado
scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)

xgb = XGBClassifier(
    scale_pos_weight=scale_weight,
    eval_metric='logloss',
    random_state=42
)
xgb.fit(X_train_scaled, y_train)

# Predição de probabilidades e aplicação de limiar customizado (Threshold Tuning)
y_probs_xgb = xgb.predict_proba(X_test_scaled)[:, 1]
custom_threshold = 0.3
y_pred_xgb_custom = (y_probs_xgb >= custom_threshold).astype(int)

print(f"=== 3. XGBoost (Limiar Ajustado = {custom_threshold}) ===")
print(classification_report(y_test, y_pred_xgb_custom))
print("AUC-ROC XGBoost:", roc_auc_score(y_test, y_probs_xgb))

# 7. Otimização de Hiperparâmetros (GridSearchCV)
param_grid = {
    'max_depth': [3, 5],
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1]
}

grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid,
    scoring='recall',
    cv=3,
    n_jobs=-1
)
grid.fit(X_train_scaled, y_train)

print("=== 4. Otimização via GridSearchCV ===")
print("Melhores Parâmetros:", grid.best_params_)

# 8. Explicabilidade do Modelo (SHAP)
explainer = shap.Explainer(grid.best_estimator_)
shap_values = explainer(X_test_scaled[:100])

plt.figure()
shap.plots.bar(shap_values)