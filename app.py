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

st.sidebar.success("Modelo ativo: Transfer Learning (MobileNetV2)")

# Optional uploaded model
uploaded_model_file = st.sidebar.file_uploader(
    "Enviar modelo (.h5)",
    type=["h5"],
    help="Use essa opção caso você queira testar um modelo .h5 diferente do padrão."
)
st.sidebar.caption("Arraste e solte o arquivo aqui")
st.sidebar.caption("Limite de 200 MB por arquivo • H5")

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
transfer_learning_config = {
    "default_path": "models/transfer_learning_final.h5",
    "timestamped_pattern": "transfer_learning_final_*.h5"
}

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
    resolved_model_path = resolve_model_path(
        transfer_learning_config["default_path"],
        transfer_learning_config["timestamped_pattern"]
    )

if resolved_model_path is None:
    st.error("Modelo Transfer Learning não encontrado.")
    st.info("Você pode enviar um arquivo .h5 pela barra lateral para executar as previsões.")
    st.stop()

model_mtime = Path(resolved_model_path).stat().st_mtime if Path(resolved_model_path).exists() else None
classifier = load_model(resolved_model_path, cache_key=model_mtime)

if classifier is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Escolha uma imagem...",
    type=["jpg", "jpeg", "png", "bmp"]
)
st.caption("Arraste e solte o arquivo aqui")
st.caption("Limite de 200 MB por arquivo • JPG, JPEG, PNG, BMP")

if uploaded_file is not None:
    uploaded_file.seek(0, 2)
    uploaded_file_size = uploaded_file.tell()
    uploaded_file.seek(0)

    if uploaded_file_size > MAX_UPLOAD_SIZE_BYTES:
        st.error(f"A imagem é muito grande. Tamanho máximo permitido: {MAX_UPLOAD_SIZE_MB} MB.")
        st.stop()

    if not uploaded_file.type or not uploaded_file.type.startswith("image/"):
        st.error("Tipo de arquivo inválido. Envie uma imagem válida.")
        st.stop()

    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
    except Exception:
        st.error("Imagem inválida ou corrompida.")
        st.stop()

    # Display image
    image = image.convert("RGB") if image.mode != "RGB" else image
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(image, width="stretch")
    
    # Make prediction
    with col2:
        st.subheader("Predição")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        image.save(temp_path, format="JPEG")

        try:
            # Get prediction
            pred_class, confidence, scores = classifier.predict(
                str(temp_path),
                return_confidence=True
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()
        
        # Display results
        st.metric("Classe Predita", pred_class)
        st.metric("Confiança", f"{confidence:.2%}")
        
        # Display confidence for all classes
        st.subheader("Pontuação de Confiança")
        for i, class_name in enumerate(classifier.class_names):
            st.write(f"{class_name}: {scores[i]:.2%}")
        
        # Visualize confidence
        fig, ax = plt.subplots()
        colors = ['green' if i == np.argmax(scores) else 'lightgray' 
                 for i in range(len(scores))]
        ax.bar(classifier.class_names, scores, color=colors)
        ax.set_ylabel('Confiança')
        ax.set_ylim([0, 1])
        st.pyplot(fig)
else:
    st.info("Envie uma imagem para visualizar a predição e as probabilidades por classe.")
