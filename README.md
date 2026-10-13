# Kalkulator półmaratonu

Aplikacja Streamlit szacująca czas ukończenia półmaratonu na podstawie płci, wieku i czasu na 5 km.

- dane użytkownika są wyciągane z tekstu przez OpenAI (JSON), a wywołania śledzi Langfuse
- model (scikit-learn) jest trenowany w notebooku [`pipeline_trenowania_modelu.ipynb`](pipeline_trenowania_modelu.ipynb):
  dane z DigitalOcean Spaces → czyszczenie → feature selection → porównanie modeli → zapis nowej wersji
  lokalnie i do Spaces; aplikacja pobiera z Spaces `models/latest.pkl`

## Uruchomienie lokalne

```bash
cp .env.example .env   # uzupełnij klucze
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

DigitalOcean App Platform: zmienne środowiskowe z `.env.example` ustaw jako zmienne aplikacji;
komenda startowa jest w `Procfile`.
