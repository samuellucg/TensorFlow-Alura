# sns.scatterplot(x=x[:,2],y=x[:,3],hue=y,palette="tab10")
# plt.xlabel("Comprimento (cm)",fontsize=16)
# plt.xlabel("largura (cm)",fontsize=16)
# plt.title("Distribuição de pétalas",fontsize=18)
# plt.show()



'''
Scatterplot: 

Eixo X: No scatterplot, o eixo X representa o comprimento das pétalas (coluna x[:, 2]).
Eixo Y: O eixo Y representa a largura das pétalas (coluna x[:, 3]).
Hue: O parâmetro hue determina como as diferentes categorias (no caso, os tipos de flores: 0 = setosa, 1 = versicolor, 2 = virginica) serão diferenciadas no gráfico, utilizando cores distintas.
Palette: A paleta de cores escolhida ("tab10") define as cores que serão usadas para representar essas categorias no gráfico.
'''


# Gráfico para avaliar métricas juntas ML
'''
# pd.DataFrame(log.history).plot()
# plt.grid()
# plt.show()

'''

# Gráfico para avaliar métricas separado ML
'''
fig,ax = plt.subplots(1,2, figsize=(14,5))
ax[0].plot(log.history["loss"],color="#111487",linewidth=3,label="Loss Train")
ax[0].plot(log.history["val_loss"],color="#EFA316",linewidth=3,label="Loss Validation",axes=ax[0])
legend = ax[0].legend(loc="best",shadow=True)

ax[1].plot(log.history["categorical_accuracy"],color="#111487", linewidth=3, label="Accuracy train")
ax[1].plot(log.history["val_categorical_accuracy"],color="#EFA316", linewidth=3, label="Accuracy validation")
legend = ax[1].legend(loc="best",shadow=True)

plt.suptitle("Desemepnho do treinamento",fontsize=18)
plt.show()
'''