"""
Streamlit web application for image classification.

Run with: streamlit run app.py
"""


import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os
from urllib.request import urlretrieve
import base64
from io import BytesIO

# Função utilitária global para converter imagem PIL para base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Import from src
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.predict import ImageClassifier


# Page configuration

st.set_page_config(
    page_title="Classificador de Imagens",
    page_icon="🐱",
    layout="wide"
)

st.markdown(

    """
    <style>
    .main .block-container {
        max-width: 1100px !important;
        min-width: 340px !important;
        margin: 0 auto !important;
    }
    .analysis-flex-outer {
        width: 100vw !important;
        display: flex !important;
        justify-content: center !important;
        align-items: flex-start !important;
        margin: 0 !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }
    .analysis-flex-inner {
        display: flex !important;
        flex-wrap: wrap !important;
        justify-content: center !important;
        align-items: flex-start !important;
        gap: 2.5rem !important;
        width: 100%;
        margin: 0 auto !important;
    }
    /* .kpi-card styles moved to main CSS block below */
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <script>
    window.addEventListener('DOMContentLoaded', function() {
        const dz = document.querySelector('[data-testid=\"stFileUploaderDropzone\"]');
        if (dz && !dz.querySelector('.custom-upload-msg')) {
            const msg = document.createElement('div');
            msg.className = 'custom-upload-msg';
            msg.innerText = 'Envie uma imagem para visualizar a predição e as probabilidades por classe.';
            dz.appendChild(msg);
        }
        // Força botão à esquerda
        const btn = dz.querySelector('button');
        if (btn) btn.style.marginLeft = '0';
    });
    </script>
    """,
    unsafe_allow_html=True
)




MAX_UPLOAD_SIZE_MB = 200
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

Image.MAX_IMAGE_PIXELS = 20_000_000

st.markdown(
    """
    <div class="hero-card">
        <h1 class="hero-title">Classificador de Imagens</h1>
        <p class="hero-subtitle">Este app serve para classificar imagens de pets de forma rápida e visual. Ele usa Transfer Learning com MobileNetV2 para reaproveitar conhecimento de visão computacional, processa sua imagem e exibe a classe prevista (Gato ou Cachorro) com nível de confiança.</p>
    </div>
""",
unsafe_allow_html=True
)

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
    st.markdown(
        """
        <div class="social-links">
            <div class="social-links-title">Conecte-se comigo 👇</div>
            <a class="social-link" href="https://www.linkedin.com/in/leandroandradeti/" target="_blank" rel="noopener noreferrer">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                    <path d="M4.98 3.5C4.98 4.88 3.86 6 2.48 6S0 4.88 0 3.5 1.12 1 2.5 1s2.48 1.12 2.48 2.5zM.5 8h4V24h-4V8zm7 0h3.8v2.2h.1c.53-1 1.83-2.2 3.77-2.2C19.2 8 21 10.2 21 14v10h-4v-8.5c0-2-.03-4.5-2.75-4.5-2.75 0-3.17 2.15-3.17 4.36V24h-4V8z"/>
                </svg>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


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
        urlretrieve(model_url, str(target))
        return str(target)
    except Exception:
        return None

# Load model
@st.cache_resource
def load_model(model_path, cache_key=None):
    """Load model from cache."""
    try:
        return ImageClassifier(
            model_path=model_path,
            class_names=['Gato', 'Cachorro']
        )
    except FileNotFoundError:
        st.error("Arquivo de modelo não encontrado. Treine ou envie um modelo primeiro.")
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
    # Sempre tenta usar o modelo final mais recente
    if Path(TRANSFER_MODEL_PATH).exists():
        resolved_model_path = TRANSFER_MODEL_PATH
    else:
        # Se não existir localmente, tenta baixar do URL
        resolved_model_path = ensure_model_from_url(TRANSFER_MODEL_PATH)

classifier = None
if resolved_model_path is not None and Path(resolved_model_path).exists():
    model_mtime = Path(resolved_model_path).stat().st_mtime
    classifier = load_model(resolved_model_path, cache_key=model_mtime)

if classifier is not None:
    st.sidebar.success("Modelo ativo: Transfer Learning (MobileNetV2)")
else:
    st.sidebar.warning("Modelo padrão não encontrado. Envie um .h5 para analisar imagens.")

if "image_uploader_key" not in st.session_state:
    st.session_state.image_uploader_key = 0

# File uploader


try:
    show_uploader = not st.session_state.get("hide_uploader", False)
    uploaded_files = None

    if show_uploader:
        uploaded_files = st.file_uploader(
            "Escolha uma ou mais imagens...",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key=f"image_uploader_{st.session_state.image_uploader_key}"
        )
        st.markdown(
            """
            <script>
            window.addEventListener('DOMContentLoaded', function() {
                const dz = document.querySelector('[data-testid=\"stFileUploaderDropzone\"]');
                if (dz && !dz.querySelector('.custom-upload-msg')) {
                    const msg = document.createElement('div');
                    msg.className = 'custom-upload-msg';
                    msg.innerText = 'Envie uma imagem para visualizar a predição e as probabilidades por classe.';
                    dz.appendChild(msg);
                }
            });
            </script>
            """,
            unsafe_allow_html=True
        )
        if uploaded_files:
            st.session_state.hide_uploader = True
            st.session_state.last_uploaded_files = uploaded_files
            st.rerun()

    if not show_uploader:
        if st.button("Nova análise"):
            st.session_state.hide_uploader = False
            st.session_state.image_uploader_key += 1
            st.rerun()

    if not show_uploader and classifier is None:
        st.error("Nenhum modelo carregado. Envie um arquivo .h5 na barra lateral para continuar.")

    if not show_uploader and classifier is not None:
        uploaded_files = st.session_state.get("last_uploaded_files", [])
        for index, uploaded_file in enumerate(uploaded_files, start=1):
            uploaded_file.seek(0, 2)
            uploaded_file_size = uploaded_file.tell()
            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert("RGB")
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

            if confidence >= 0.90:
                badge_text = "Alta confiança"
                badge_class = "confidence-high"
            elif confidence >= 0.70:
                badge_text = "Média confiança"
                badge_class = "confidence-medium"
            else:
                badge_text = "Baixa confiança"
                badge_class = "confidence-low"

            st.markdown(f"""
<div style='display: flex; flex-wrap: wrap; gap: 2.2rem; align-items: flex-start; margin: 1.5rem 0 2.2rem 0; background: #f8fafc; border-radius: 18px; box-shadow: 0 2px 12px #0001; padding: 1.3rem 1.3rem 1.3rem 1.3rem;'>
    <div style='flex:1; min-width:220px; max-width:340px; display:flex; flex-direction:column; align-items:center;'>
        <img src='data:image/png;base64,{image_to_base64(image)}' width='290' style='border-radius:12px; box-shadow:0 1px 8px #0001; margin-bottom:0.7rem;' />
    </div>
    <div style='flex:2; min-width:260px; max-width:480px;'>
        <div class="kpi-card" style="padding-top:0.1rem; margin-top:0; box-shadow:none; background:transparent;">
            <div class="kpi-card-title" style="margin-top:0; font-size:1.15rem;">&#128200; Painel de Indicadores</div>
            <div style='display:flex; gap:1.2rem; margin-bottom:0.5rem; margin-top:0.7rem;'>
                <div style='flex:1;'>
                    <div style='font-size:0.97rem; color:#64748b; font-weight:600;'>Classe Identificada</div>
                    <div style='font-size:1.18rem; color:#0f172a; font-weight:700; margin-bottom:0.2rem;'>{pred_class}</div>
                </div>
                <div style='flex:1;'>
                    <div style='font-size:0.97rem; color:#64748b; font-weight:600;'>Nível de Confiança</div>
                    <div style='font-size:1.18rem; color:#0f172a; font-weight:700; margin-bottom:0.2rem;'>{confidence:.2%}</div>
                </div>
            </div>
            <span class="confidence-badge {badge_class}" style="margin-bottom:0.7rem;">{badge_text}</span>
            <div class="kpi-section-title" style='margin-top:1.1rem;'>Distribuição de Probabilidades</div>
            <div style='margin-bottom:0.7rem;'>
                <div class="kpi-score-row"><strong>{classifier.class_names[0]}</strong>: {scores[0]:.2%}</div>
                <div class="kpi-score-row"><strong>{classifier.class_names[1]}</strong>: {scores[1]:.2%}</div>
            </div>
        </div>
    </div>
</div>
""",
unsafe_allow_html=True
)

            if index < len(uploaded_files):
                st.markdown('---')
except Exception as e:
    st.error(f"Erro inesperado ao processar a imagem ou exibir resultados: {e}")
