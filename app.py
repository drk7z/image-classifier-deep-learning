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
import numpy as np

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


def save_validated_temp_image(uploaded_file) -> tuple[Path | None, Image.Image | None, str | None]:
    uploaded_file.seek(0, 2)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        return None, None, f"Arquivo excede {MAX_UPLOAD_SIZE_MB} MB."

    try:
        with Image.open(uploaded_file) as image_obj:
            width, height = image_obj.size
            if width * height > MAX_IMAGE_PIXELS:
                return None, None, f"Imagem muito grande ({width}x{height})."
            image_rgb = image_obj.convert("RGB")
    except UnidentifiedImageError:
        return None, None, "Arquivo inválido: não é uma imagem reconhecida."
    except Exception as exc:
        return None, None, f"Falha ao abrir imagem: {exc}"

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    image_rgb.save(tmp_path, format="JPEG")

    return tmp_path, image_rgb, None


def predict_with_cleanup(classifier, temp_path: Path) -> tuple[str | None, float | None, np.ndarray | None, str | None]:
    prediction_error = None
    pred_class = None
    confidence_value = None
    scores_array = None

    try:
        pred_class, confidence, scores = classifier.predict(str(temp_path), return_confidence=True)
        try:
            confidence_value = float(confidence)
        except Exception:
            confidence_value = 0.0

        if not math.isfinite(confidence_value):
            confidence_value = 0.0

        scores_array = np.asarray(scores, dtype=float).flatten()
        scores_array = np.nan_to_num(scores_array, nan=0.0, posinf=1.0, neginf=0.0)
        if scores_array.size == 1:
            p1 = float(np.clip(scores_array[0], 0.0, 1.0))
            scores_array = np.array([1.0 - p1, p1], dtype=float)
    except Exception as exc:
        prediction_error = f"Falha na inferência: {exc}"
    finally:
        if temp_path.exists():
            temp_path.unlink()

    if prediction_error:
        return None, None, None, prediction_error

    return pred_class, confidence_value, scores_array, None


st.title("🐱🐶 Classificador de Imagens")
st.write(
    "Este app classifica imagens de pets com Transfer Learning (MobileNetV2) "
    "e mostra a classe prevista (Gato ou Cachorro) com nível de confiança."
)
st.info("Envie uma imagem para visualizar a predição e as probabilidades por classe.")

with st.sidebar.expander("⚙️ Como funciona", expanded=False):
    st.markdown("""
    1. Você envia uma imagem (JPG, JPEG, PNG ou BMP).
    2. A imagem é validada e normalizada para inferência.
    3. O modelo **Transfer Learning (MobileNetV2)** processa a entrada.
    4. O app retorna a classe predita (**Gato** ou **Cachorro**) e a confiança.
    5. Um gráfico mostra as probabilidades das duas classes.
    """)

with st.sidebar.expander("🧠 Arquitetura e treino", expanded=False):
    st.markdown("""
    - Dataset com 2 classes: **Gato** e **Cachorro**.
    - Estratégia de treino: **Data Augmentation** para robustez.
    - Backbone: **MobileNetV2** pré-treinada (Transfer Learning).
    - Camadas finais densas para classificação binária.
    - Monitoramento de treino com callbacks (early stopping e ajuste de learning rate).
    """)

with st.sidebar.expander("📊 Resultados esperados", expanded=False):
    st.markdown("""
    - Acurácia típica em Transfer Learning: **~96% a 98%**.
    - Métricas acompanhadas: **Accuracy, Precision, Recall e F1-score**.
    - A qualidade da imagem impacta diretamente a confiança da predição.

    **Dica prática:** use imagens nítidas, com boa iluminação e o pet em destaque.
    """)

with st.sidebar.expander("🔗 Links", expanded=False):
    st.markdown("Conecte-se comigo 👇")
    st.markdown("- [LinkedIn](https://www.linkedin.com/in/leandroandradeti/)")
    st.markdown("- [GitHub](https://github.com/drk7z)")

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

    temp_path, image_rgb, validation_error = save_validated_temp_image(uploaded_file)
    if validation_error:
        st.error(validation_error)
        st.divider()
        continue

    pred_class, confidence_value, scores_array, prediction_error = predict_with_cleanup(classifier, temp_path)
    if prediction_error:
        st.error(prediction_error)
        st.divider()
        continue

    preview_col, panel_col = st.columns([1, 1.4], gap="large")

    with preview_col:
        st.image(image_rgb, caption=f"Imagem {index}", use_container_width=True)

    with panel_col:
        st.success(f"Classe: {pred_class}")
        st.write(f"Confiança: {confidence_value:.2%}")

        if scores_array is not None and scores_array.size >= 2:
            cat_score = float(np.clip(scores_array[0], 0.0, 1.0))
            dog_score = float(np.clip(scores_array[1], 0.0, 1.0))
            st.write(f"Gato: {cat_score:.2%}")
            st.progress(cat_score)
            st.write(f"Cachorro: {dog_score:.2%}")
            st.progress(dog_score)

    st.divider()
