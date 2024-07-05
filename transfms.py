from transformers import pipeline

# Инициализация пайплайна анализа настроений с использованием DistilBERT
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Анализ настроений текста
text = "According to Kevin Svenson, Bitcoin will reach a new peak in the next six months. The analyst justified his forecast "\
 "previous movement of the Bitcoin price after previous halving of the reward to miners. Historically, Bitcoin peaked after 40–80"\
 "weeks after the halving, says the analyst. The 2024 halving should correspond to this model, and already in January 2025"\
 "Bitcoin will reach higher levels this year, Svenson is sure. This period not only corresponds to historical ones" \
 "BTC indicators, but also coincides with the month of the inauguration of the US President after the November elections."
result = classifier(text)
print(f'result {result}')

text = "Bitcoin will grow, analysts predict! It will be higher and better! That means you need to buy."
result = classifier(text)
print(f'result {result}')