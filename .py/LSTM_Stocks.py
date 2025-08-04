
# %%
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

# %%
def set_seeds(seed=42):
    """Define seeds para garantir reprodutibilidade"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.experimental.enable_op_determinism()
    print(f"Seeds definidas para {seed} - Reprodutibilidade garantida!")

# CHAME IMEDIATAMENTE
set_seeds(42)

# %%
start_date = '2015-01-01'
end_date = '2025-07-12'
ticker = 'PETR4.SA'

# df = yf.download(ticker, start=start_date, end=end_date)
df = pd.read_csv('PETR4_SA_yahoo.csv')

# %%
df_close = df[['Date', 'Close']].copy()
df_close = df_close.set_index('Date')  # Agora 'Date' é o índice
df_close.index = pd.to_datetime(df_close.index)
df_close.head()

# %%
print(df_close.index)
print(type(df_close.index[-1]))

# %%
def create_lag_features(df, columns, lags):
    """
    Cria features de lag para machine learning
    
    Parâmetros:
    - df: DataFrame com os dados
    - columns: lista de colunas para criar lags (ex: ['Close', 'Volume'])
    - lags: lista de períodos anteriores (ex: [1, 2, 3, 5])
    
    Retorna:
    - DataFrame com features de lag
    """
    # Cria uma cópia do DataFrame
    df_result = df.copy()
    
    # Para cada coluna
    for col in columns:
        # Para cada lag
        for lag in lags:
            # Cria nova coluna com valores deslocados
            df_result[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Remove linhas com valores NaN (causados pelos lags)
    max_lag = max(lags)
    df_result = df_result.iloc[max_lag:].copy()
    
    return df_result

# %%
list_of_attributes = ['Close']
list_of_prev_t_instants = list(range(1, 16))
df_new = create_lag_features(df_close, list_of_attributes, list_of_prev_t_instants)


# %% Desenhar a arquitetura do modelo
input_layer = Input(shape=(len(list_of_prev_t_instants),), dtype='float32')
dense1 = Dense(128, activation='relu')(input_layer)
dense2 = Dense(64, activation='relu')(dense1)
dropout_layer = Dropout(0.3)(dense2)
dense3 = Dense(32, activation='relu')(dropout_layer)
output_layer = Dense(1, activation='linear')(dense3)

# %% Estrutura da rede
model = Model(inputs=input_layer, outputs=output_layer)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)
# model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# %%
# Definindo proporções
test_size = 0.05
valid_size = 0.05

# Resetando o índice
# df_copy = df_new.reset_index(drop=True)
df_copy = df_new.copy()

# Calculando índices de corte
# O código divide o DataFrame em treino, validação e teste, preservando a ordem temporal (importante para séries temporais).
n = len(df_copy)
test_start = int(np.floor(n * (1 - test_size)))
valid_start = int(np.floor((n - (n * test_size)) * (1 - valid_size)))

# Separando os conjuntos
df_test = df_copy.iloc[test_start:]
df_train_plus_valid = df_copy.iloc[:test_start]
df_train = df_train_plus_valid.iloc[:valid_start]
df_valid = df_train_plus_valid.iloc[valid_start:]

# Separando features e alvo (assumindo que a primeira coluna é o alvo)
X_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, 0]
X_valid, y_valid = df_valid.iloc[:, 1:], df_valid.iloc[:, 0]
X_test, y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0]

# Exibindo os shapes
print('Train:', X_train.shape, y_train.shape)
print('Valid:', X_valid.shape, y_valid.shape)
print('Test:', X_test.shape, y_test.shape)
# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Cria os scalers
# feature_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
# target_scaler = MinMaxScaler(feature_range=(0.01, 0.99))


feature_scaler = StandardScaler()
target_scaler = StandardScaler()

# Ajusta o scaler só com os dados de treino e transforma todos os conjuntos
X_train_scaled = feature_scaler.fit_transform(X_train)
X_valid_scaled = feature_scaler.transform(X_valid)
X_test_scaled  = feature_scaler.transform(X_test)

y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1))
y_test_scaled  = target_scaler.transform(y_test.values.reshape(-1, 1))

# %%
# Treinando o modelo
history = model.fit(
    X_train_scaled,           # Dados de entrada de treino
    y_train_scaled,           # Alvo de treino
    batch_size=32,             # Tamanho do mini-batch
    epochs=100,                # Número de épocas
    validation_data=(X_valid_scaled, y_valid_scaled),  # Dados de validação
    shuffle=False,             
    verbose=1                 # Mostra o progresso
)

# %% Fazendo previsões com o modelo treinado
y_pred = model.predict(X_test_scaled)

# Revertendo a normalização para obter os valores reais
y_test_real = target_scaler.inverse_transform(y_test_scaled)
y_pred_real = target_scaler.inverse_transform(y_pred)

# Criando DataFrames para visualização
df_resultados = pd.DataFrame({
    'Actual': y_test_real.flatten(),
    'Predicted': y_pred_real.flatten()
})

# Exibindo as primeiras linhas
print(df_resultados.head())
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))  # Tamanho do gráfico

plt.plot(df_resultados['Actual'], color='red', label='Actual')
plt.plot(df_resultados['Predicted'], linestyle='dashed', color='navy', label='Predicted')

plt.legend(loc='best')
plt.ylabel('Adj Close')
plt.xlabel('Test Set Day no.')
plt.xticks(rotation=45)
plt.title('Comparação: Valor Real vs. Valor Previsto')
plt.tight_layout()
plt.show()
# %%
# Calculando as métricas principais
mae = metrics.mean_absolute_error(y_test_real, y_pred_real)
mse = metrics.mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test_real, y_pred_real)
ev = metrics.explained_variance_score(y_test_real, y_pred_real)
mgd = metrics.mean_gamma_deviance(y_test_real, y_pred_real)
mpd = metrics.mean_poisson_deviance(y_test_real, y_pred_real)

# Organizando em DataFrame
results = {
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2,
    "Explained variance": ev,
    "Mean gamma deviance": mgd,
    "Mean Poisson deviance": mpd
}

df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
print(df_results)

# MAE: Erro absoluto médio (quanto, em média, as previsões erram em relação ao valor real).
# MSE: Erro quadrático médio (penaliza mais erros grandes).
# RMSE: Raiz do erro quadrático médio (interpretação na mesma escala dos dados).
# R2: Coeficiente de determinação (quanto da variação dos dados é explicada pelo modelo).
# EV: Variância explicada (quanto da variância dos dados é explicada pelo modelo).
# MGD: Desvio gama médio (métrica para distribuições assimétricas, menos comum).
# MPD: Desvio de Poisson médio (métrica para contagens, menos comum).


# %%
n_steps = len(list_of_prev_t_instants)  # número de lags
n_forecast = 5  # quantos dias à frente você quer prever

# Pegue os últimos n_steps valores reais
input_seq = df['Close'][-n_steps:].values.tolist()
previsoes = []  # SEMPRE limpe a lista antes de começar

for _ in range(n_forecast):
    # Normaliza a sequência
    input_scaled = feature_scaler.transform(np.array(input_seq).reshape(1, -1))
    # Faz a previsão
    pred_scaled = model.predict(input_scaled)
    # Desfaz a normalização
    pred_real = target_scaler.inverse_transform(pred_scaled)[0][0]
    previsoes.append(pred_real)
    # Atualiza a sequência de entrada para a próxima previsão
    input_seq.append(pred_real)
    input_seq.pop(0)


# %% 
# ultima_data = df.index[-1]  # Última data disponível nos  dados
ultima_data = df_close.index[-1]

# Gera as próximas datas (dias úteis)
datas_futuras = pd.bdate_range(start=ultima_data + pd.Timedelta(days=1), periods=n_forecast)
print(f"Tamanho da lista datas_futuras: {len(datas_futuras)}")

# Cria o DataFrame de previsões
df_previsoes = pd.DataFrame({
    'Data': datas_futuras.strftime('%d/%m/%Y'),
    'Previsao': previsoes
})

print(df_previsoes)

# %%
# Configurações de estilo
sns.set(style="whitegrid", context="talk", palette="deep")
plt.figure(figsize=(10, 6))

# Gráfico de linha com pontos
sns.lineplot(
    x='Data',
    y='Previsao',
    data=df_previsoes,
    marker='o',
    linewidth=3,
    markersize=10,
    color='#1a73e8'
)

# --- RÓTULOS OTIMIZADOS ---
for i, row in df_previsoes.iterrows():
    # Ajuste dinâmico da posição vertical para evitar sobreposição
    offset_y = 0.02 if i != len(df_previsoes) - 1 else 0.05  # Último ponto mais alto
    
    plt.text(
        x=row['Data'], 
        y=row['Previsao'] + offset_y,
        s=f"{row['Previsao']:.2f}",
        ha='center',
        va='bottom',
        fontsize=12,
        color='#1a73e8',
        weight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)  # Fundo branco para contraste
    )

# Formatação do eixo X para abreviar datas (ex: "15/07" em vez de "15/07/2025")
plt.gca().set_xticklabels([d.split('/')[0] + '/' + d.split('/')[1] for d in df_previsoes['Data']])

# Configurações finais
plt.title('Previsão de Fechamento das Ações PETR4.SA', fontsize=20, weight='bold', color='#22223b')
plt.xlabel('Data', fontsize=14, weight='bold')
plt.ylabel('Preço Previsto (R$)', fontsize=14, weight='bold')
plt.xticks(rotation=30, ha='right', fontsize=14)
plt.yticks(fontsize=12)
plt.grid(visible=True, linestyle='--', alpha=0.3)
sns.despine()
plt.tight_layout()

plt.show()

# %%
