from tensorflow import keras
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

iris = datasets.load_iris(return_X_y=True)
x,y = iris[0], iris[1]

'''
x = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
y = 0-setosa, 1-versicolor, 2-virginica
'''

# To categorical (y) (Categoriza os valores em one-hot-encoding)
y = keras.utils.to_categorical(y) # (1,0,0, 0,1,0, 0,0,1)

# MinMaxScaler (Normaliza os valores para um valor em um intervalo de 0.0 à 1.0)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Separa dados de treino de dados de teste
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,stratify=y,random_state=42)

model = keras.Sequential([keras.layers.InputLayer([4,],name="input_data"),
                          keras.layers.Dense(512,"relu",name="hidden_layer",kernel_initializer=keras.initializers.RandomNormal(seed=142)),
                          keras.layers.Dense(3,activation="softmax",name="output")
                          ]) 

# Compilação do modelo
model.compile(loss="categorical_crossentropy",metrics=["categorical_accuracy"])

# Treinamento do modelo
ep = 100 

log = model.fit(x_train,y_train,epochs=ep,validation_split=0.3)

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

# Normalizando os dados para teste
test_data_normalized = scaler.transform(test_data)

# Predição
predicoes = model.predict(test_data_normalized)

predicoes_classes = np.argmax(predicoes, axis=1)
nomes_classes = ['Setosa', 'Versicolor', 'Virginica']
predicoes_classes_nomes = [nomes_classes[i] for i in predicoes_classes]

# Resultado
print(predicoes_classes_nomes)


