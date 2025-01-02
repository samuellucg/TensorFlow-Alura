from tensorflow import keras
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
print(x_test[1])
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
print(f"shape: {x.shape}\n")
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

# evaluate and predict

print()
# print(model.evaluate(x_test,y_test))
print(y_test)


# novas_amostras = [
#     [5.0, 3.4, 1.5, 0.2],  # Exemplo de uma flor (Setosa)
#     [6.7, 3.0, 5.2, 2.3],  # Exemplo de uma flor (Versicolor)
#     [5.8, 2.7, 5.1, 1.9],  # Exemplo de uma flor (Versicolor)
#     [6.4, 3.2, 4.5, 1.5],  # Exemplo de uma flor (Versicolor)
#     [7.1, 3.0, 5.9, 2.1],  # Exemplo de uma flor (Virginica)
#     [6.5, 3.0, 5.5, 1.8],  # Exemplo de uma flor (Virginica)
#     [4.6, 3.1, 1.5, 0.2],  # Exemplo de uma flor (Setosa)
#     [5.7, 3.0, 4.2, 1.2],  # Exemplo de uma flor (Versicolor)
#     [6.8, 3.0, 5.5, 2.1]   # Exemplo de uma flor (Virginica)
# ]
novas_amostras = [
    [0.47222222, 0.41666667, 0.6440678, 0.70833333],
    [0.25, 0.625, 0.08474576, 0.04166667],
    [0.16666667, 0.16666667, 0.38983051, 0.375],
    [0.19444444, 0.125, 0.38983051, 0.375],
    [0.41666667, 0.29166667, 0.69491525, 0.75],
    [0.55555556, 0.54166667, 0.62711864, 0.625],
    [0.94444444, 0.33333333, 0.96610169, 0.79166667],
    [0.72222222, 0.5, 0.79661017, 0.91666667],
    [0.08333333, 0.5, 0.06779661, 0.04166667],
    [0.30555556, 0.41666667, 0.59322034, 0.58333333],
    [0.19444444, 0.58333333, 0.08474576, 0.04166667],
    [0.05555556, 0.125, 0.05084746, 0.08333333],
    [0.19444444, 0.66666667, 0.06779661, 0.04166667],
    [0.58333333, 0.5, 0.59322034, 0.58333333],
    [0.83333333, 0.375, 0.89830508, 0.70833333],
    [0.44444444, 0.41666667, 0.69491525, 0.70833333],
    [0.41666667, 0.33333333, 0.69491525, 0.95833333],
    [0.36111111, 0.20833333, 0.49152542, 0.41666667],
    [0.5, 0.375, 0.62711864, 0.54166667],
    [0.02777778, 0.41666667, 0.05084746, 0.04166667],
    [0.61111111, 0.5, 0.69491525, 0.79166667],
    [0.19444444, 0.625, 0.05084746, 0.08333333],
    [0.63888889, 0.375, 0.61016949, 0.5],
    [0.16666667, 0.45833333, 0.08474576, 0.04166667],
    [0.25, 0.875, 0.08474576, 0],
    [0.66666667, 0.41666667, 0.6779661, 0.66666667],
    [0.63888889, 0.41666667, 0.57627119, 0.54166667],
    [0.94444444, 0.25, 1.0, 0.91666667],
    [0.30555556, 0.79166667, 0.05084746, 0.125],
    [0.58333333, 0.33333333, 0.77966102, 0.83333333]
]

predicoes = model.predict(novas_amostras)

# Obtendo as classes preditas com base na maior probabilidade
predicoes_classes = np.argmax(predicoes, axis=1)
nomes_classes = ['Setosa', 'Versicolor', 'Virginica']
predicoes_classes_nomes = [nomes_classes[i] for i in predicoes_classes]

# Exibindo as classes preditas
print(predicoes_classes_nomes)

# Resolva o Overfitting.
