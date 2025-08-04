# ==========================================
# PREVISÃO DE RETORNOS PERCENTUAIS - PETR4.SA
# ==========================================

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import torch
import yfinance as yf
from sklearn import metrics
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %%
# ---------------------------
# 1. Reprodutibilidade
# ---------------------------
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.experimental.enable_op_determinism()
    print(f"Seeds definidas para {seed} - Reprodutibilidade garantida!")

set_seeds(42)

# ---------------------------
# 2. Carregar os dados
# ---------------------------
# Use o arquivo local para garantir reprodutibilidade
df = pd.read_csv('PETR4_SA_yahoo.csv')
df = df[['Date', 'Close']].copy()
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.sort_index()

# ---------------------------
# 3. Calcular o retorno percentual (log-return)
# ---------------------------
# log-return: log(Close_t / Close_{t-1})
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()  # Remove o primeiro valor (NaN)

# ---------------------------
# 4. Criar features de lag dos retornos
# ---------------------------
def create_lag_features(df, col, lags):
    df_lag = df.copy()
    for lag in lags:
        df_lag[f'{col}_lag_{lag}'] = df_lag[col].shift(lag)
    df_lag = df_lag.dropna()
    return df_lag

lags = list(range(1, 16))  # 15 lags
df_lagged = create_lag_features(df, 'log_return', lags)

# ---------------------------
# 5. Split temporal: treino, validação, teste
# ---------------------------
# Usar apenas dados recentes para evitar regime shift
# Exemplo: últimos 5 anos para treino+validação+teste
split_start = df_lagged.index >= (df_lagged.index.max() - pd.DateOffset(years=5))
df_lagged = df_lagged[split_start]

n = len(df_lagged)
test_size = 0.05
valid_size = 0.05

test_start = int(np.floor(n * (1 - test_size)))
valid_start = int(np.floor((n - (n * test_size)) * (1 - valid_size)))

df_test = df_lagged.iloc[test_start:]
df_train_plus_valid = df_lagged.iloc[:test_start]
df_train = df_train_plus_valid.iloc[:valid_start]
df_valid = df_train_plus_valid.iloc[valid_start:]

# ---------------------------
# 6. Separar features e alvo
# ---------------------------
feature_cols = [f'log_return_lag_{lag}' for lag in lags]
target_col = 'log_return'

X_train, y_train = df_train[feature_cols], df_train[target_col]
X_valid, y_valid = df_valid[feature_cols], df_valid[target_col]
X_test, y_test = df_test[feature_cols], df_test[target_col]

# ---------------------------
# 7. Normalização (StandardScaler)
# ---------------------------
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

X_train_scaled = feature_scaler.fit_transform(X_train)
X_valid_scaled = feature_scaler.transform(X_valid)
X_test_scaled  = feature_scaler.transform(X_test)

y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1))
y_test_scaled  = target_scaler.transform(y_test.values.reshape(-1, 1))

# ---------------------------
# 8. Construção do modelo
# ---------------------------
input_layer = Input(shape=(len(lags),), dtype='float32')
dense1 = Dense(64, activation='relu')(input_layer)
dense2 = Dense(32, activation='relu')(dense1)
dropout_layer = Dropout(0.3)(dense2)
output_layer = Dense(1, activation='linear')(dropout_layer)

model = Model(inputs=input_layer, outputs=output_layer)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss='mse', optimizer=optimizer)
model.summary()

# ---------------------------
# 9. Treinamento
# ---------------------------
history = model.fit(
    X_train_scaled, y_train_scaled,
    batch_size=32,
    epochs=100,
    validation_data=(X_valid_scaled, y_valid_scaled),
    shuffle=False,
    verbose=1
)

# ---------------------------
# 10. Avaliação no conjunto de teste
# ---------------------------
y_pred_scaled = model.predict(X_test_scaled)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_test_real = target_scaler.inverse_transform(y_test_scaled)

# DataFrame para análise
df_resultados = pd.DataFrame({
    'Actual': y_test_real.flatten(),
    'Predicted': y_pred.flatten()
}, index=y_test.index)

print(df_resultados.head())

# ---------------------------
# 11. Métricas de avaliação
# ---------------------------
mae = metrics.mean_absolute_error(y_test_real, y_pred)
mse = metrics.mean_squared_error(y_test_real, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test_real, y_pred)
ev = metrics.explained_variance_score(y_test_real, y_pred)

results = {
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2,
    "Explained variance": ev
}
df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
print(df_results)

# ---------------------------
# 12. Visualização dos resultados
# ---------------------------
plt.figure(figsize=(14, 6))
plt.plot(df_resultados['Actual'], color='red', label='Actual')
plt.plot(df_resultados['Predicted'], linestyle='dashed', color='navy', label='Predicted')
plt.legend(loc='best')
plt.ylabel('Log-Return')
plt.xlabel('Test Set Day no.')
plt.title('Comparação: Log-Return Real vs. Previsto')
plt.tight_layout()
plt.show()

# ---------------------------
# 13. Previsão para os próximos dias
# ---------------------------
n_forecast = 5
input_seq = df_lagged[feature_cols].iloc[-1].values.tolist()
previsoes = []

for _ in range(n_forecast):
    input_scaled = feature_scaler.transform(np.array(input_seq).reshape(1, -1))
    pred_scaled = model.predict(input_scaled)
    pred_real = target_scaler.inverse_transform(pred_scaled)[0][0]
    previsoes.append(pred_real)
    # Atualiza a sequência de entrada para a próxima previsão
    input_seq = input_seq[1:] + [pred_real]

# Gera as datas futuras
ultima_data = df_lagged.index[-1]
datas_futuras = pd.bdate_range(start=ultima_data + pd.Timedelta(days=1), periods=n_forecast)

# DataFrame de previsões
df_previsoes = pd.DataFrame({
    'Data': datas_futuras.strftime('%d/%m/%Y'),
    'Log-Return Previsto': previsoes
})

print(df_previsoes)

# ---------------------------
# 14. (Opcional) Converter previsão de log-return para preço futuro
# ---------------------------
# Começa do último preço conhecido
ultimo_preco = df['Close'].iloc[-1]
precos_previstos = [ultimo_preco]

for log_ret in previsoes:
    novo_preco = precos_previstos[-1] * np.exp(log_ret)
    precos_previstos.append(novo_preco)

# Remove o primeiro (é o preço atual)
precos_previstos = precos_previstos[1:]

df_previsoes['Preço Previsto'] = precos_previstos

print(df_previsoes)

# Visualização das previsões de preço
plt.figure(figsize=(10, 5))
sns.lineplot(x='Data', y='Preço Previsto', data=df_previsoes, marker='o', color='#1a73e8')
plt.title('Previsão de Preço Futuro da PETR4.SA')
plt.xlabel('Data')
plt.ylabel('Preço Previsto (R$)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# %%