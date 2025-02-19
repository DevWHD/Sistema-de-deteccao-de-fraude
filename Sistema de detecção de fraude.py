import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from xgboost import XGBClassifier

np.random.seed(42)
num_samples = 10000
data = pd.DataFrame({
    'valor_transacao': np.random.uniform(10,10000,num_samples),
    'tempo_transacao': np.random.randint(0,86400, num_samples),
    'origem_pais': np.random.choice(['BR','US', 'FR','DE','JP'],num_samples),
    'destino_pais': np.random.choice(['BR','US','FR','DE','JP'],num_samples),
    'tipo_pagamento': np.random.choice(['Cartão de Crédito', 'Pix', 'Boleto', 'TED'],num_samples),
    'historico_fraude': np.random.randint(0,2,num_samples),
    'fraude': np.random.choice([0,1],num_samples, p=[0.95, 0.05])
})

print(data.head())

data = pd.get_dummies(data, columns=['origem_pais', 'destino_pais', 'tipo_pagamento'], drop_first=True)

X = data.drop(columns=['fraude'])
y= data['fraude']

scaler= StandardScaler()
X[['valor_transacao','tempo_transacao']]= scaler.fit_transform(X[['valor_transacao','tempo_transacao']])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Matriz de Confusão: ")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação: ")
print(classification_report(y_test, y_pred))

print("\nAcurácia: ", accuracy_score(y_test, y_pred))

nova_transacao = pd.DataFrame([{
    'valor_transacao': 5000,
    'tempo_transacao': 36000,
    'historico_fraude': 1,
    'origem_pais_US': 0, 'origem_pais_FR': 0,'origem_pais_DE': 0, 'origem_pais_JP':0,
    'destino_pais_US': 1, 'destino_pais_FR': 0, 'destino_pais_DE': 0, 'destino_pais_JP': 0,
    'tipo_pagamento_Cartão de Crédito':0, 'tipo_pagamento_Pix': 1, 'tipo_pagamento_Boleto': 0, 'tipo_pagamento_TED': 0
}])
colunas_faltantes = set(X.columns) - set(nova_transacao.columns)
for coluna in colunas_faltantes:
    nova_transacao[coluna] = 0

nova_transacao = nova_transacao[X.columns]

nova_transacao[['valor_transacao', 'tempo_transacao']]= scaler.transform(nova_transacao[['valor_transacao', 'tempo_transacao']])

previsao = modelo.predict(nova_transacao)

print("\nA transação é fraude?", "Sim" if previsao[0] == 1 else "Não")

modelo_xgb = XGBClassifier(n_estimators=100,random_state=42, eval_metric='logloss')
modelo_xgb.fit(X_train, y_train)
probabilidade = modelo_xgb.predict_proba(nova_transacao)[0][1]
print(f"\nProbabilidade de ser fraude: {probabilidade:.2%}")