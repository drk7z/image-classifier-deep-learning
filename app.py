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

# Import from src
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.predict import ImageClassifier


# Page configuration

st.set_page_config(
    page_title="Classificador de Imagens",
    page_icon="🐱🐶",
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
    .stColumns {
        display: flex !important;
        justify-content: center !important;
        align-items: flex-start !important;
        gap: 2.5rem !important;
    }
    .stColumn {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: flex-start !important;
        min-width: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    .kpi-card {
        max-width: 440px;
        margin-left: auto !important;
        margin-right: auto !important;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .kpi-card-title, .kpi-section-title, .kpi-score-row, .confidence-badge {
        text-align: center !important;
        width: 100%;
        margin-left: auto !important;
        margin-right: auto !important;
        display: block;
    }
    .stImage > img, .stImage img {
        max-width: 350px !important;
        width: 100% !important;
        height: auto !important;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .stMetric, .stMetricLabel, .stMetricValue {
        text-align: center !important;
        margin-left: auto !important;
        margin-right: auto !important;
        display: block !important;
    }
    @media (max-width: 900px) {
        .stColumns {
            flex-direction: column !important;
            align-items: center !important;
        }
        .stColumn {
            min-width: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }
        .kpi-card {
            max-width: 100% !important;
        }
        .kpi-card-title, .kpi-section-title, .kpi-score-row, .confidence-badge {
            max-width: 100% !important;
        }
        .stImage > img, .stImage img {
            max-width: 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        color: #0f172a;
    }
    .stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #0f172a;
    }
    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        background: white;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        margin-bottom: 1rem;
    }
    .hero-title {
        margin: 0;
        color: #0f172a !important;
        font-size: 1.85rem;
        font-weight: 700;
    }
    .hero-subtitle {
        margin-top: 0.5rem;
        color: #334155 !important;
        font-size: 1rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid #e2e8f0;
        background: #ffffff;
    }
    [data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #f8fafc;
        border: 1px dashed #94a3b8;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: #0f172a !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #f8fafc;
        border: 1px dashed #94a3b8;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.7rem;
        min-height: 72px;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #0f172a !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: #e2e8f0 !important;
        color: #0f172a !important;
        border: 1px solid #94a3b8 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #cbd5e1 !important;
        color: #0f172a !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        font-size: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "Procurar arquivos";
        font-size: 0.95rem;
        font-weight: 600;
    }
    .main [data-testid="stFileUploaderDropzone"]::after {
        content: "Arraste e solte o arquivo aqui\A Limite de 200 MB por arquivo • JPG, JPEG, PNG, BMP";
        white-space: pre-line;
        line-height: 1.25;
        font-size: 0.82rem;
        color: #334155 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]::after {
        content: "Arraste e solte o arquivo aqui\A Limite de 200 MB por arquivo • H5";
        white-space: pre-line;
        line-height: 1.25;
        font-size: 0.82rem;
        color: #334155 !important;
    }
    .stButton > button,
    [data-baseweb="button"] {
        background: #e2e8f0 !important;
        color: #0f172a !important;
        border: 1px solid #94a3b8 !important;
    }
    .stButton > button:hover,
    [data-baseweb="button"]:hover {
        background: #cbd5e1 !important;
        color: #0f172a !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }
    .social-links {
        margin-top: 0.5rem;
        display: flex;
        flex-direction: column;
        gap: 0.55rem;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding-bottom: 0.9rem;
        width: 100%;
    }
    .social-links-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #334155 !important;
        text-align: center;
        margin-bottom: 0.15rem;
    }
    .social-link {
        text-decoration: none !important;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        width: 92%;
        max-width: 240px;
        padding: 0.55rem 0.7rem;
        border-radius: 10px;
        border: 1px solid #94a3b8;
        background: #eef2ff;
        color: #0f172a !important;
        font-weight: 600;
        transition: all 0.18s ease;
        box-sizing: border-box;
        margin: 0 auto;
    }
    .social-link:hover {
        background: #dbeafe;
        border-color: #64748b;
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.12);
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 14px;
        padding: 0.9rem 1rem 1rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    }
    .kpi-card-title {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a !important;
        margin-bottom: 0.45rem;
    }
    .kpi-section-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #334155 !important;
        margin-top: 0.35rem;
        margin-bottom: 0.25rem;
    }
    .kpi-score-row {
        font-size: 0.9rem;
        color: #334155 !important;
        margin-bottom: 0.2rem;
    }
    .confidence-badge {
        display: inline-block;
        margin: 0.2rem 0 0.65rem;
        padding: 0.3rem 0.55rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid transparent;
    }
    .confidence-high {
        background: #dcfce7;
        color: #166534 !important;
        border-color: #86efac;
    }
    .confidence-medium {
        background: #fef3c7;
        color: #92400e !important;
        border-color: #fcd34d;
    }
    .confidence-low {
        background: #fee2e2;
        color: #991b1b !important;
        border-color: #fca5a5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

MAX_UPLOAD_SIZE_MB = 200
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

Image.MAX_IMAGE_PIXELS = 20_000_000

st.markdown(
    """
    <div class="hero-card">
        <h1 class="hero-title">🐱 vs 🐶 Classificador de Imagens</h1>
        <p class="hero-subtitle">🚀 Este app serve para classificar imagens de pets de forma rápida e visual. Ele usa Transfer Learning com MobileNetV2 para reaproveitar conhecimento de visão computacional, processa sua imagem e exibe a classe prevista (Gato ou Cachorro) com nível de confiança 📊.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Configurações")

# Optional uploaded model
uploaded_model_file = st.sidebar.file_uploader(
    "Enviar modelo (.h5)",
    type=["h5"],
    help="Use essa opção caso você queira testar um modelo .h5 diferente do padrão."
)

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
                <span>LinkedIn</span>
            </a>
            <a class="social-link" href="https://github.com/drk7z" target="_blank" rel="noopener noreferrer">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                    <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.1 3.3 9.43 7.88 10.96.58.1.79-.25.79-.56v-2.16c-3.2.7-3.87-1.35-3.87-1.35-.52-1.33-1.28-1.68-1.28-1.68-1.05-.72.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.03 1.75 2.7 1.25 3.36.95.1-.75.4-1.26.73-1.55-2.55-.29-5.23-1.27-5.23-5.68 0-1.26.45-2.3 1.18-3.12-.12-.29-.51-1.45.11-3.03 0 0 .97-.31 3.19 1.19a11.1 11.1 0 0 1 5.8 0c2.22-1.5 3.18-1.19 3.18-1.19.63 1.58.24 2.74.12 3.03.74.82 1.18 1.86 1.18 3.12 0 4.42-2.69 5.39-5.26 5.67.41.35.78 1.05.78 2.12v3.14c0 .31.21.67.8.56A11.52 11.52 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5z"/>
                </svg>
                <span>GitHub</span>
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
    uploaded_files = st.file_uploader(
        "Escolha uma ou mais imagens...",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key=f"image_uploader_{st.session_state.image_uploader_key}"
    )

    if uploaded_files:
        if st.button("Nova análise"):
            st.session_state.image_uploader_key += 1
            st.rerun()

    if uploaded_files and classifier is None:
        st.error("Nenhum modelo carregado. Envie um arquivo .h5 na barra lateral para continuar.")

    if uploaded_files and classifier is not None:
        for index, uploaded_file in enumerate(uploaded_files, start=1):
            uploaded_file.seek(0, 2)
            uploaded_file_size = uploaded_file.tell()
            uploaded_file.seek(0)

            if uploaded_file_size > MAX_UPLOAD_SIZE_BYTES:
                st.error(f"A imagem #{index} é muito grande. Tamanho máximo permitido: {MAX_UPLOAD_SIZE_MB} MB.")
                continue

            if not uploaded_file.type or not uploaded_file.type.startswith("image/"):
                st.error(f"A imagem #{index} tem tipo inválido. Envie uma imagem válida.")
                continue

            try:
                image = Image.open(uploaded_file)
                image.verify()
                uploaded_file.seek(0)
                image = Image.open(uploaded_file)
            except Exception:
                st.error(f"A imagem #{index} está inválida ou corrompida.")
                continue

            image = image.convert("RGB") if image.mode != "RGB" else image

            st.markdown(f"### Foto {index}")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Imagem Original")
                st.image(image, width="stretch")

            with col2:
                st.subheader("Análise")
                st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
                st.markdown('<div class="kpi-card-title">📈 Painel de Indicadores</div>', unsafe_allow_html=True)

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

                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Classe Identificada", pred_class)
                with metric_col2:
                    st.metric("Nível de Confiança", f"{confidence:.2%}")

                if confidence >= 0.90:
                    badge_text = "Alta confiança"
                    badge_class = "confidence-high"
                elif confidence >= 0.70:
                    badge_text = "Média confiança"
                    badge_class = "confidence-medium"
                else:
                    badge_text = "Baixa confiança"
                    badge_class = "confidence-low"

                st.markdown(
                    f'<span class="confidence-badge {badge_class}">{badge_text}</span>',
                    unsafe_allow_html=True
                )

                st.markdown('<div class="kpi-section-title">Distribuição de Probabilidades</div>', unsafe_allow_html=True)
                for i, class_name in enumerate(classifier.class_names):
                    st.markdown(
                        f'<div class="kpi-score-row"><strong>{class_name}</strong>: {scores[i]:.2%}</div>',
                        unsafe_allow_html=True
                    )

                fig, ax = plt.subplots(figsize=(6.2, 3.2))
                colors = ['#22c55e' if i == np.argmax(scores) else '#cbd5e1'
                         for i in range(len(scores))]
                bars = ax.barh(classifier.class_names, scores, color=colors)
                ax.set_xlim([0, 1])
                ax.set_xlabel('Probabilidade')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.grid(axis='x', linestyle='--', alpha=0.25)

                for bar, score in zip(bars, scores):
                    ax.text(
                        min(score + 0.02, 0.98),
                        bar.get_y() + bar.get_height() / 2,
                        f"{score:.1%}",
                        va='center',
                        ha='left',
                        fontsize=9,
                        color='#0f172a'
                    )

                fig.tight_layout()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

            if index < len(uploaded_files):
                st.markdown('---')
    else:
        st.info("Envie uma imagem para visualizar a predição e as probabilidades por classe.")
except Exception as e:
    st.error(f"Erro inesperado ao processar a imagem ou exibir resultados: {e}")
