# Kalkulator półmaratonu

Aplikacja Streamlit szacująca czas ukończenia półmaratonu na podstawie płci, wieku i czasu na 5 km.

- dane użytkownika są wyciągane z tekstu przez OpenAI (JSON), a wywołania śledzi Langfuse
- model (scikit-learn) jest trenowany w notebooku `modul_9/41__zadanie_domowe.ipynb`
  i pobierany z DigitalOcean Spaces (`models/latest.pkl`)

## Uruchomienie lokalne

```bash
cp .env.example .env   # uzupełnij klucze
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

DigitalOcean App Platform: zmienne środowiskowe z `.env.example` ustaw jako zmienne aplikacji;
komenda startowa jest w `Procfile`.
