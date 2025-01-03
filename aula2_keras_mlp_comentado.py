from tensorflow import keras
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

iris = datasets.load_iris(return_X_y=True)
x,y = iris[0], iris[1]

'''
x = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
y = 0-setosa, 1-versicolor, 2-virginica
'''

# print(f"Esse é o x: {x}\n")
# print(f"Esse é o y: {y}\n")

# To categorical (y) (Categoriza os valores em one-hot-encoding)
y = keras.utils.to_categorical(y) # (1,0,0, 0,1,0, 0,0,1)

# MinMaxScaler (Normaliza os valores para um valor em um intervalo de 0.0 à 1.0)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# train_test_split  (pq devo fazer isso e esses param: (test_size, stratify, random_state=42)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,stratify=y,random_state=42)
print(x_test.shape)
'''
-x = Dados de entrada

-y = Dados de saída

-test_size = Porcentagem de alocação para teste, nesse caso 20% será para as variáveis de teste.

-stratify = Garante que a divisão de amostras seja feita de forma proporcional. Se você por exemplo tiver 50 amostras de tal item. ele irá dividir
igualmente de forma que 20% fique para teste e 80% para treino. Como nesse exemplo.

-random_state = Uma seed para a divisão sempre ser a mesma, no caso sempre será o mesmo resultado independentemente quantas x rodar o código.
'''


# Explicar o motivo do model ser assim:

# Criação do modelo:
# print(f"shape: {x.shape}\n")
model = keras.Sequential([keras.layers.InputLayer([4,],name="input_data"),
                          # esse [4,] representa que para cada amostrar de flor terá 4 features(características). 4 elementos por amostra
                          keras.layers.Dense(512,"relu",name="hidden_layer",kernel_initializer=keras.initializers.RandomNormal(seed=142)),
                          # O valor de 512 neurônios, é um valor determinado pela instrutora. Para criar um eficaz,
                          # é apenas testando mesmo. Verificando as taxas de validação e teste.
                          keras.layers.Dense(3,activation="softmax",name="output")
                          # 3 neurônios pois são 3 classes de Iris.
                          ]) 

# Softmax é usado muito para questão de probabilidade, como estamos fazendo uma classificação não linear, e queremos que o nosso modelo
# "preveja" qual tipo de flor é. Softmax é ideal.

model.summary()

# Compilação do modelo:
model.compile(loss="categorical_crossentropy",metrics=["categorical_accuracy"])
'''

Esses parâmetros passados no compile, são tanto para as métricas de validação quanto para teste.

Loss = Função de perda que o modelo tenta minimizar durante o treinamento.

o modelo prevê com o softmax isso:

[0.2, 0.7, 0.1]

20% de chance de ser Setosa
70% de chance de ser Versicolor
10% de chance de ser Virginica

e na realidade realmente é Versicolor. Então ele irá fazer um cálculo para pegar a diferença
porque afinal não foi 100% de certeza que era versicolor então tem uma taxa pro erro.

metrics = Cálculo para a acurácia do modelo. Passado em uma lista pois pode receber mais de um se quiser.

--------------------------------------------------------------------------------------------------------------------------------------------------------
Underfitting =  Ocorre quando o modelo não aprende o suficiente com os dados de treinamento. 
Isso significa que o modelo é muito simples para capturar as relações complexas e os padrões nos dados. 
Como resultado, o modelo tem um desempenho ruim tanto nos dados de treinamento quanto nos dados de validação/teste.

Overfitting = Ocorre quando o modelo aprende excessivamente os dados de treinamento, incluindo ruídos ou peculiaridades que 
não representam padrões reais nos dados. Como resultado, o modelo se torna muito específico para os dados de treinamento e 
perde a capacidade de generalizar bem para novos dados. O modelo pode parecer excelente nos dados de treinamento, mas terá um 
desempenho muito ruim nos dados de validação/teste.

RESUMO:

Underfitting = Modelo muito simples → Não consegue aprender os padrões dos dados → Desempenho ruim.
Overfitting = Modelo muito complexo → Aprende os detalhes dos dados de treinamento, mas não generaliza bem → Desempenho ruim nos dados de validação.
--------------------------------------------------------------------------------------------------------------------------------------------------------

OBS:

Épocas são importantes porque permitem que o modelo ajuste seus parâmetros ao longo do tempo, melhorando suas previsões.

Poucas épocas podem resultar em underfitting, onde o modelo não aprendeu o suficiente.

Muitas épocas podem resultar em overfitting, onde o modelo aprendeu demais e perdeu a capacidade de fazer previsões precisas para novos dados.

O número de épocas deve ser ajustado de acordo com o comportamento da perda e das métricas de validação durante o treinamento.

'''
# Treinamento do modelo:
ep = 100 # Número de épocas

log = model.fit(x_train,y_train,epochs=ep,validation_split=0.3)

historico = log.history # Histórico das métricas de todos os treinamentos, o gráfico é feito em cima disso.

# Dados para testar o modelo
test_data = [
    [5.1, 3.5, 1.4, 0.2],  # Setosa
    [7.0, 3.2, 4.7, 1.4],  # Versicolor
    [6.3, 3.3, 6.0, 2.5],  # Virginica
    [5.5, 2.5, 4.0, 1.3],  # Versicolor
    [5.9, 3.0, 5.1, 1.8],  # Virginica
    [4.8, 3.4, 1.6, 0.2],  # Setosa
    [6.7, 3.1, 4.4, 1.4],  # Versicolor
    [6.2, 2.9, 4.3, 1.3],  # Versicolor
    [5.8, 2.7, 5.1, 1.9],  # Virginica
    [5.0, 3.5, 1.6, 0.6]   # Setosa
]

test_data_normalized = scaler.transform(test_data)
predicoes = model.predict(test_data_normalized)

# Obtendo as classes preditas com base na maior probabilidade
predicoes_classes = np.argmax(predicoes, axis=1)
nomes_classes = ['Setosa', 'Versicolor', 'Virginica']
predicoes_classes_nomes = [nomes_classes[i] for i in predicoes_classes]

print(predicoes_classes_nomes)


# Resolva o Overfitting.
