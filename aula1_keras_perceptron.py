from tensorflow import keras
'''
Vamos modelar um Perceptron utilizando keras, mas apenas MODELAR pois para treinar não faria sentido pelo 
Keras ser focado em modelos não lineares.
'''

'''
Componentes principais do Keras:
- Keras.Sequential: Facilita a criação de uma rede neural ao conectar camadas sequencialmente.
- Keras.Layers: Oferece camadas pré-construídas com recursos específicos.
- Keras.Layers.Dense: Camada densa, com features built-in.
  Parâmetros principais:
    - units (int): Número de neurônios na camada.
    - input_shape (list[int]): Quantidade de entradas esperadas.
    - name (str): Nome do modelo para facilitar a identificação.
'''

model = keras.Sequential([keras.layers.Dense(1,input_shape=[5],name="FirstModel",kernel_initializer=keras.initializers.random_normal(),
                                             bias_initializer=keras.initializers.Ones())])

# model.summary() # Retorna a estrutura do modelo

'''
A quantidade de parâmetros que o summary mostra é referente a quantidade de entradas + bias.

Nesse caso, o summary retornou 3 parâmetros sendo eles:
    2 = Nossa entrada
    1 = Valor de BIAS.
'''

# keras.utils.plot_model(model,show_shapes=True) # Retorna a estrutura do modelo mas em uma imagem

# print(model.layers) # Retorna o tipo da nossa camada, nesse caso dense. Pode retornar mais de uma gpt se tiver mais de uma camada?

# print(model.layers[0].get_weights()) # Retorna dois arrays. O primeiro contendo o valor dos pesos e o outro contendo o valor de BIAS.

weight,bias = model.layers[0].get_weights() # Apenas separando o que expliquei anteriormente, em duas var diferentes para facilitar compreensão

print(weight.shape) # Retorna uma tupla com 2 valores sendo (2= Entradas, 1 = Saída)
print(f"PESOS: {weight} \n")

print(bias.shape) # Retorna uma tupla com 1 valor sendo (1=Valor de bias,)
print(f"BIAS: {bias}")


# Inicializando BIAS e PESOS diretamente no modelo.

model2 = keras.Sequential(
    [keras.layers.Dense(1,input_shape=[2],name="SecondModel",kernel_initializer=keras.initializers.random_normal(),
                        bias_initializer=keras.initializers.Ones())])

'''
Criamos da mesma forma anteriormente mas agora passando 2 parâmetros a mais que são:

  Kernel_initializer e Bias_initializer.

  Kernel_initializer = É para você inicializar os PESOS, aqui passamos o método
  keras.initializers.random_normal() que escolhe o valor dos pesos aleatoriamente.

  Bias_initializer = É para você inicializar o valor de BIAS, aqui passamos o método
  keras.initializers.Ones() que inicia o valor de BIAS sempre como 1.
'''


# model2.summary() 

# weight2,bias2 = model2.layers[0].get_weights()

# print(weight2.shape)
# print(f"PESOS INICIALIZADOS: {weight2}")

# print(bias2.shape)
# print(f"BIAS INICIALIZADOS: {bias2}")
