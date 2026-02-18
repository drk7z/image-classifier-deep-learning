
"""
Streamlit web application for image classification.

Run with: streamlit run app.py
"""

import streamlit as st
from PIL import Image, UnidentifiedImageError
import os
import tempfile
import shutil
from pathlib import Path
from urllib.request import urlopen





MAX_UPLOAD_SIZE_MB = 10
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

Image.MAX_IMAGE_PIXELS = 20_000_000

st.title("🐱🐶 Classificador de Imagens")
st.write(
    "Este app classifica imagens de pets com Transfer Learning (MobileNetV2) "
    "e mostra a classe prevista (Gato ou Cachorro) com nível de confiança."
)
st.info("Envie uma imagem para visualizar a predição e as probabilidades por classe.")
st.divider()

# Sidebar


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
    st.markdown("- [GitHub](https://github.com/leandroandradeti)")


def resolve_model_path(default_path, timestamped_pattern):
    """Resolve model path, preferring fixed filename and falling back to latest timestamped file."""
    default_model_path = Path(default_path)

    if default_model_path.exists():
        return str(default_model_path)

    model_candidates = sorted(
        default_model_path.parent.glob(timestamped_pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    if model_candidates:
        return str(model_candidates[0])

    return None


def ensure_model_from_url(target_path):
    """Try downloading model from env/secrets URL when local model is unavailable."""
    model_url = os.getenv("TRANSFER_MODEL_URL")
    if not model_url:
        try:
            model_url = st.secrets.get("TRANSFER_MODEL_URL")
        except Exception:
            model_url = None

    if not model_url:
        return None

    target = Path(target_path)
    target.parent.mkdir(exist_ok=True)

    try:
        with urlopen(model_url, timeout=20) as response:
            with target.open('wb') as output:
                shutil.copyfileobj(response, output)
        return str(target)
    except Exception:
        return None

# Load model
@st.cache_resource
def load_model(model_path, cache_key=None):
    """Load model from cache."""
    try:
        from src.predict import ImageClassifier
        return ImageClassifier(
            model_path=model_path,
            class_names=['Gato', 'Cachorro']
        )
    except FileNotFoundError:
        st.error("Arquivo de modelo não encontrado. Treine ou envie um modelo primeiro.")
        return None
    except Exception as e:
        st.error(f"Falha ao carregar o modelo no backend: {e}")
        return None


# Model paths

# Garante que a variável existe para evitar NameError
uploaded_model_file = None

# Sempre prioriza o modelo final mais recente e robusto
TRANSFER_MODEL_FILENAME = "transfer_learning_final_20260216_162455.h5"
TRANSFER_MODEL_PATH = f"models/{TRANSFER_MODEL_FILENAME}"

resolved_model_path = None

if uploaded_model_file is not None:
    uploaded_model_file.seek(0, 2)
    uploaded_model_size = uploaded_model_file.tell()
    uploaded_model_file.seek(0)

    if uploaded_model_size > 200 * 1024 * 1024:
        st.error("O modelo enviado é muito grande. Tamanho máximo permitido: 200 MB.")
        st.stop()

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    uploaded_model_path = models_dir / "uploaded_model.h5"
    uploaded_model_path.write_bytes(uploaded_model_file.getbuffer())
    resolved_model_path = str(uploaded_model_path)
    st.sidebar.info("Modelo enviado carregado para esta sessão.")
else:
    # No startup, nunca baixa modelo da internet para evitar travamento da página.
    if Path(TRANSFER_MODEL_PATH).exists():
        resolved_model_path = TRANSFER_MODEL_PATH
    else:
        resolved_model_path = TRANSFER_MODEL_PATH

model_exists = resolved_model_path is not None and Path(resolved_model_path).exists()
if model_exists:
    st.sidebar.success("Modelo encontrado: Transfer Learning (MobileNetV2)")
else:
    st.sidebar.warning("Modelo padrão não encontrado. Envie um .h5 para analisar imagens.")

try:
    uploaded_files = st.file_uploader(
        "Escolha uma ou mais imagens...",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )

    uploaded_files = uploaded_files or []

    if not uploaded_files:
        st.info("Envie uma imagem para iniciar a análise.")
    elif not model_exists:
        st.warning("Modelo local não encontrado. Tentando baixar modelo remoto...")
        downloaded_path = ensure_model_from_url(TRANSFER_MODEL_PATH)
        if downloaded_path is not None and Path(downloaded_path).exists():
            resolved_model_path = downloaded_path
            model_exists = True
            st.success("Modelo baixado com sucesso.")
        else:
            st.error("Não foi possível obter o modelo (local/remoto). Verifique TRANSFER_MODEL_URL.")

    if uploaded_files and model_exists:
        try:
            model_mtime = Path(resolved_model_path).stat().st_mtime
            classifier = load_model(resolved_model_path, cache_key=model_mtime)
        except Exception as e:
            classifier = None
            st.error(f"Erro ao preparar o modelo: {e}")

        if classifier is None:
            st.error("Não foi possível inicializar o modelo para inferência.")
            st.stop()
        for index, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                if uploaded_file is None:
                    continue

                uploaded_file.seek(0, 2)
                uploaded_file_size = uploaded_file.tell()
                uploaded_file.seek(0)

                if uploaded_file_size > MAX_UPLOAD_SIZE_BYTES:
                    st.error(
                        f"Imagem {index} excede {MAX_UPLOAD_SIZE_MB} MB. "
                        "Envie uma imagem menor para evitar travamentos no deploy."
                    )
                    continue

                try:
                    with Image.open(uploaded_file) as pil_image:
                        width, height = pil_image.size
                        if width * height > Image.MAX_IMAGE_PIXELS:
                            st.error(
                                f"Imagem {index} muito grande ({width}x{height}). "
                                "Use uma imagem menor."
                            )
                            continue
                        image = pil_image.convert("RGB")
                except UnidentifiedImageError:
                    st.error(f"Arquivo {index} não é uma imagem válida.")
                    continue
                except Exception:
                    st.error("Imagem muito grande ou inválida.")
                    continue

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                image.save(temp_path, format="JPEG")
                try:
                    pred_class, confidence, scores = classifier.predict(
                        str(temp_path),
                        return_confidence=True
                    )
                finally:
                    if temp_path.exists():
                        temp_path.unlink()

                st.subheader(f"Resultado da imagem {index}")
                preview_col, panel_col = st.columns([1, 1.4], gap="large")

                with preview_col:
                    st.image(image, caption=f"Imagem {index}", use_container_width=True)

                with panel_col:
                    with st.container(border=True):
                        st.markdown("### 📊 Painel de Indicadores")
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("Classe Identificada", pred_class)
                        with metric_col2:
                            st.metric("Nível de Confiança", f"{confidence:.2%}")

                        if confidence < 0.6:
                            st.warning("Imagem não reconhecida como gato ou cachorro.")
                        elif confidence >= 0.90:
                            st.success("Alta confiança")
                        elif confidence >= 0.70:
                            st.info("Média confiança")
                        else:
                            st.warning("Baixa confiança")

                        if len(scores) >= 2:
                            st.markdown("**Distribuição de Probabilidades**")
                            st.write(f"Gato: {scores[0]:.2%}")
                            st.progress(float(scores[0]))
                            st.write(f"Cachorro: {scores[1]:.2%}")
                            st.progress(float(scores[1]))

                if index < len(uploaded_files):
                    st.divider()
            except Exception as e:
                import traceback
                st.error(f"Erro inesperado ao processar a imagem: {e}")
                st.error(traceback.format_exc())
except Exception as e:
    st.error(f"Erro inesperado ao processar a imagem ou exibir resultados: {e}")
