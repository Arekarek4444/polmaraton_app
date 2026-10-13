import io
import json
import os
from datetime import date

import boto3
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from langfuse.openai import OpenAI
from openai import AuthenticationError

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME", "arek-polmaraton")
MODEL_KEY = os.getenv("MODEL_KEY", "models/latest.pkl")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

REQUIRED_FIELDS = {
    "plec": "płeć",
    "wiek": "wiek (lub rok urodzenia)",
    "czas_5km": "czas na 5 km",
}

SYSTEM_PROMPT = f"""Wyciągasz z wypowiedzi użytkownika dane potrzebne do oszacowania czasu półmaratonu.
Dzisiejsza data: {date.today().isoformat()}.

Zwróć WYŁĄCZNIE obiekt JSON z kluczami:
- "plec": "M" (mężczyzna) albo "K" (kobieta). Możesz wywnioskować płeć z jednoznacznej formy gramatycznej
  (np. "biegałam", "jestem zmęczona") lub z imienia, jeśli jest jednoznaczne. W przeciwnym razie null.
- "wiek": liczba całkowita lat. Jeśli użytkownik podał rok urodzenia, policz wiek. Jeśli brak – null.
- "czas_5km": czas na 5 km w formacie "MM:SS" (np. "24:30"; "25 minut" -> "25:00",
  "pół godziny" -> "30:00"). Jeśli podano tempo na km, przelicz je na 5 km. Jeśli brak – null.

Nie zgaduj brakujących wartości – wtedy wpisz null."""


@st.cache_resource
def load_model():
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
    return joblib.load(io.BytesIO(obj["Body"].read()))


@st.cache_resource
def get_openai_client():
    return OpenAI()


def parse_time_to_seconds(value):
    if not value:
        return None
    try:
        parts = [int(p) for p in str(value).strip().split(":")]
    except ValueError:
        return None
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds if len(parts) in (2, 3) else None


def format_seconds(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def validate(data):
    """Zwraca (dane dla modelu, lista brakujących pól, lista problemów)."""
    missing, problems = [], []

    plec = data.get("plec")
    if plec not in ("M", "K"):
        missing.append(REQUIRED_FIELDS["plec"])

    wiek = data.get("wiek")
    if wiek is None:
        missing.append(REQUIRED_FIELDS["wiek"])
    elif not 10 <= int(wiek) <= 100:
        problems.append(f"Wiek {wiek} lat wygląda na nieprawidłowy.")

    czas_5km_s = parse_time_to_seconds(data.get("czas_5km"))
    if data.get("czas_5km") is None:
        missing.append(REQUIRED_FIELDS["czas_5km"])
    elif czas_5km_s is None:
        problems.append(f"Nie rozumiem czasu na 5 km: {data.get('czas_5km')}.")
    elif not 13 * 60 <= czas_5km_s <= 60 * 60:
        problems.append("Czas na 5 km powinien mieścić się między 13 a 60 minut.")

    features = None
    if not missing and not problems:
        features = {"czas_5km_s": czas_5km_s, "plec_m": int(plec == "M"), "wiek": int(wiek)}
    return features, missing, problems


@observe(name="ekstrakcja_danych_biegacza")
def extract_runner_data(text):
    with propagate_attributes(tags=["polmaraton"], metadata={"model": OPENAI_MODEL}):
        response = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            name="ekstrakcja_json",  # nazwa generacji w Langfuse
        )
    data = json.loads(response.choices[0].message.content)
    _, missing, problems = validate(data)

    # metryki skuteczności LLM w Langfuse
    langfuse = get_client()
    langfuse.score_current_trace(name="dane_kompletne", value=int(not missing and not problems), data_type="BOOLEAN")
    langfuse.score_current_trace(name="liczba_brakujacych_pol", value=len(missing))
    langfuse.score_current_trace(name="poprawny_format", value=int(not problems), data_type="BOOLEAN")

    return data, langfuse.get_current_trace_id()


# ---------------------------------------------------------------- UI

st.set_page_config(page_title="Kalkulator półmaratonu", page_icon="🏃")
st.title("🏃 Ile zajmie Ci półmaraton?")
st.write("Przedstaw się i napisz kilka słów o sobie: **płeć**, **wiek** i **czas na 5 km**. "
         "Model wytrenowany na wynikach Półmaratonu Wrocławskiego 2023–2024 oszacuje Twój czas.")

text = st.text_area(
    "Opowiedz o sobie",
    placeholder="Cześć, jestem Ania, mam 32 lata, a 5 km przebiegam w 27 minut.",
    height=120,
)

if st.button("Oszacuj czas", type="primary"):
    if not text.strip():
        st.warning("Najpierw napisz coś o sobie 🙂")
        st.stop()

    with st.spinner("Analizuję Twoją wypowiedź..."):
        try:
            data, trace_id = extract_runner_data(text)
        except AuthenticationError:
            st.error("Nieprawidłowy klucz OpenAI (OPENAI_API_KEY). Sprawdź konfigurację aplikacji.")
            st.stop()
        except Exception as e:
            st.error(f"Nie udało się przetworzyć tekstu przez LLM: {e}")
            st.stop()
        finally:
            get_client().flush()

    st.session_state["trace_id"] = trace_id
    st.session_state["data"] = data
    st.session_state.pop("prediction", None)

    features, missing, problems = validate(data)
    if missing or problems:
        if missing:
            st.warning("Brakuje mi danych: **" + ", ".join(missing) + "**. Uzupełnij je w opisie i spróbuj ponownie.")
        for p in problems:
            st.warning(p)
    else:
        st.session_state["prediction"] = load_model().predict(pd.DataFrame([features]))[0]

if "data" in st.session_state:
    with st.expander("Dane odczytane przez LLM"):
        st.json(st.session_state["data"])

if "prediction" in st.session_state:
    st.success(f"Szacowany czas półmaratonu: **{format_seconds(st.session_state['prediction'])}**")
    st.caption("Średni błąd modelu to ok. 5 minut.")

if st.session_state.get("trace_id"):
    st.write("Czy dane zostały odczytane poprawnie?")
    feedback = st.feedback("thumbs", key=f"feedback_{st.session_state['trace_id']}")
    if feedback is not None and not st.session_state.get(f"sent_{st.session_state['trace_id']}"):
        get_client().create_score(
            trace_id=st.session_state["trace_id"],
            name="ocena_uzytkownika",
            value=feedback,  # 1 = 👍, 0 = 👎
            data_type="BOOLEAN",
        )
        get_client().flush()
        st.session_state[f"sent_{st.session_state['trace_id']}"] = True
        st.toast("Dzięki za ocenę!")
