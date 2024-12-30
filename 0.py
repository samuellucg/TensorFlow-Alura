from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 500)

activations = {
    "ReLU": np.maximum(0, x),
    "Sigmoid": 1 / (1 + np.exp(-x)),
    "Tanh": np.tanh(x),
    "Linear": x,
}

plt.figure(figsize=(10, 6))
for name, values in activations.items():
    plt.plot(x, values, label=name)
    
plt.title("Funções de Ativação")
plt.xlabel("Entrada")
plt.ylabel("Saída")
plt.legend()
plt.grid()
plt.show()
def show(models:list):
    for m in models:
        m.summary()
        print()

model_sigmoid = keras.Sequential([keras.layers.Dense(1,activation="sigmoid",input_shape=[3],name="SigmoidModel")])

keras.utils.plot_model(model_sigmoid,"sigmoid.png",True,show_layer_names=True,show_layer_activations=True)

model_reLu = keras.Sequential([keras.layers.Dense(1,activation="relu",input_shape=[3],name="ReLUModel",kernel_initializer=keras.initializers.he_normal)])

keras.utils.plot_model(model_reLu,"relu.png",True,show_layer_names=True,show_layer_activations=True)

model_tanh = keras.Sequential([keras.layers.Dense(1,activation="tanh",input_shape=[3],name="tanhModel",)])

keras.utils.plot_model(model_tanh,"tanh.png",True,show_layer_names=True,show_layer_activations=True)

model_linear = keras.Sequential([keras.layers.Dense(1,activation="linear",input_shape=[3],name="linearModel")])

keras.utils.plot_model(model_linear,"linear.png",True,show_layer_activations="True",show_layer_names=True)

models = [model_reLu,model_sigmoid,model_tanh,model_linear]

show(models)

'''
Então no caso o sigmoid é não linear e se mantém no gráfico entre 0 e 1

Tangh fica ali entre -1 que seria o 0 e 1 para positivo no caso recebendo sinal - Curva em S

Relu fica no 0 se a entrada ser menor ou igual a 0 agora se a entrada ser maior tipo 6 o gráfico iria pro 6 - Como não tá recebendo entrada se manterá no 0

Linear, apenas uma linha crescente retornando a entrada recebida
'''