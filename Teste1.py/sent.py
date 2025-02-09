import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Baixar dados necessários
nltk.download('vader_lexicon')

# Criar o analisador de sentimentos
sia = SentimentIntensityAnalyzer()

# Teste com um texto
texto = "Eu amo Python"
resultado = sia.polarity_scores(texto)

# Exibir resultado
print("Análise de Sentimento:", resultado)
