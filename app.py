"""
Streamlit web application for image classification.

Run with: streamlit run app.py
"""

from pathlib import Path
from urllib.request import urlopen
import math
import shutil
import tempfile
import os

import streamlit as st
from PIL import Image, UnidentifiedImageError


st.set_page_config(page_title="Classificador de Imagens", page_icon="🐱🐶", layout="wide")

MAX_UPLOAD_SIZE_MB = 10
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MODEL_FILENAME = "transfer_learning_final_20260216_162455.h5"
MODEL_PATH = Path("models") / MODEL_FILENAME


def get_remote_model_url() -> str | None:
    model_url = os.getenv("TRANSFER_MODEL_URL")
    if model_url:
        return model_url

    try:
        secret_url = st.secrets.get("TRANSFER_MODEL_URL")
    except Exception:
        secret_url = None

    return secret_url


def resolve_model_path() -> tuple[str | None, str | None]:
    if MODEL_PATH.exists():
        return str(MODEL_PATH), None

    remote_url = get_remote_model_url()
    if not remote_url:
        return None, "Modelo local não encontrado e TRANSFER_MODEL_URL não está configurado."

    MODEL_PATH.parent.mkdir(exist_ok=True)

    try:
        with urlopen(remote_url, timeout=8) as response:
            with MODEL_PATH.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
    except Exception as exc:
        return None, f"Falha ao baixar modelo remoto: {exc}"

    if not MODEL_PATH.exists():
        return None, "Download do modelo terminou sem criar arquivo local."

    return str(MODEL_PATH), None


@st.cache_resource
def load_model(model_path: str, cache_key: float):
    from src.predict import ImageClassifier

    return ImageClassifier(model_path=model_path, class_names=["Gato", "Cachorro"])


def save_validated_temp_image(uploaded_file) -> tuple[Path | None, str | None]:
    uploaded_file.seek(0, 2)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        return None, f"Arquivo excede {MAX_UPLOAD_SIZE_MB} MB."

    try:
        with Image.open(uploaded_file) as image_obj:
            width, height = image_obj.size
            if width * height > MAX_IMAGE_PIXELS:
                return None, f"Imagem muito grande ({width}x{height})."
            image_rgb = image_obj.convert("RGB")
    except UnidentifiedImageError:
        return None, "Arquivo inválido: não é uma imagem reconhecida."
    except Exception as exc:
        return None, f"Falha ao abrir imagem: {exc}"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    image_rgb.save(tmp_path, format="JPEG")

    return tmp_path, None


def predict_with_cleanup(classifier, temp_path: Path) -> tuple[str | None, float | None, str | None]:
    prediction_error = None
    pred_class = None
    confidence_value = None

    try:
        pred_class, confidence, _ = classifier.predict(str(temp_path), return_confidence=True)
        try:
            confidence_value = float(confidence)
        except Exception:
            confidence_value = 0.0

        if not math.isfinite(confidence_value):
            confidence_value = 0.0
    except Exception as exc:
        prediction_error = f"Falha na inferência: {exc}"
    finally:
        if temp_path.exists():
            temp_path.unlink()

    if prediction_error:
        return None, None, prediction_error

    return pred_class, confidence_value, None


st.title("🐱🐶 Classificador de Imagens")
st.write("Upload de múltiplas imagens com inferência de classe e confiança.")

uploaded_files = st.file_uploader(
    "Escolha uma ou mais imagens...",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Envie imagens para iniciar a análise.")
    st.stop()

with st.spinner("Preparando modelo..."):
    model_path, model_error = resolve_model_path()

if model_error:
    st.error(model_error)
    st.stop()

try:
    model_mtime = Path(model_path).stat().st_mtime
    classifier = load_model(model_path, cache_key=model_mtime)
except Exception as exc:
    st.error(f"Falha ao inicializar classificador: {exc}")
    st.stop()

for index, uploaded_file in enumerate(uploaded_files, start=1):
    st.subheader(f"Imagem {index}: {uploaded_file.name}")

    temp_path, validation_error = save_validated_temp_image(uploaded_file)
    if validation_error:
        st.error(validation_error)
        st.divider()
        continue

    pred_class, confidence_value, prediction_error = predict_with_cleanup(classifier, temp_path)
    if prediction_error:
        st.error(prediction_error)
        st.divider()
        continue

    st.success(f"Classe: {pred_class}")
    st.write(f"Confiança: {confidence_value:.2%}")
    st.divider()
