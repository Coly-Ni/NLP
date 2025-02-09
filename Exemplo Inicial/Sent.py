import tensorflow_datasets as tfds

# Baixar o dataset IMDB
dataset, info = tfds.load(name="imdb_reviews", split=['train', 'test'], as_supervised=True)
train_data, test_data = dataset

# tfds.load baixa o dataset
# split=['train', 'test'] divide o dataset em treino e teste
# as_supervised=True Garante a formatação correta 
# train_data, test_data = dataset Armaneza os dados de treino e teste em variaveis separadas

for review, label in train_data.take(1): # Pega um elemento do dataset
  print("Texto:", review.numpy().decode('utf-8')) # Converte bytes para string
  print("Sentimento (0 = Negativo, 1 = Positivo):", label.numpy()) # Mostra o sentimento

# O dataset é composto por reviews de filmes e um sentimento associado
# 0 = Negativo
# 1 = Positivo
# .take(1) pega um elemento do dataset
# review.numpy().decode('utf-8') Converte bytes para string
# label.numpy() converte a resposta para um numero inteiro e Mostra o sentimento

# Observando o resultado

for review, label in train_data.take(1):
  print("Texto:", review.numpy().decode('utf-8'))
  print("Sentimento (0 = Negativo, 1 = Positivo):", label.numpy())