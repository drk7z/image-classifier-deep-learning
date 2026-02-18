# 🚀 Guia Rápido - Quick Start

## ⚡ 5 Minutos para Começar

### Demo imediata (sem setup local)

- Streamlit: https://image-classifier-dl.streamlit.app/

### 1. Preparar Ambiente
```bash
# Clone o repositório
git clone https://github.com/drk7z/image-classifier-deep-learning.git
cd image-classifier-deep-learning

# Crie ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instale dependências
pip install -r requirements.txt
```

### 2. Prepare seus Dados
```
Coloque imagens em:
data/train/     → para treino
data/validation/ → para validação
data/test/      → para teste

Estrutura esperada:
data/train/
  ├── cats/
  └── dogs/
```

### 3. Treino Rápido
```bash
# Via Jupyter (recomendado)
jupyter notebook notebooks/01_cats_vs_dogs_classifier.ipynb

# Via script Python
python -c "from src.train import ImageClassifierTrainer; ImageClassifierTrainer('data').train(epochs=50)"
```

### 4. Teste seu Modelo
```bash
# Interface Web
streamlit run app.py

# Ou via Python
python -c "
from src.predict import ImageClassifier
classifier = ImageClassifier('models/cnn_classifier_final.h5')
classifier.visualize_prediction('path/seu/imagem.jpg')
"
```

---

## 📚 Próximos Passos

### Explorar o Código
- **model.py** - Arquiteturas CNN e Transfer Learning
- **train.py** - Treinamento com callbacks
- **evaluate.py** - Métricas e visualizações
- **predict.py** - Inferência em novas imagens

### Aprender Mais
1. Abra o Jupyter notebook para tutorial completo
2. Leia o README.md para documentação detalhada
3. Explore as métricas de avaliação
4. Compare CNN vs Transfer Learning

### Customizações Comuns
```python
# Mudar tamanho da imagem
IMG_SIZE = 256  # padrão: 224

# Ajustar batch size
BATCH_SIZE = 64  # padrão: 32

# Mais épocas de treino
epochs = 100  # padrão: 50

# Learning rate diferente
learning_rate = 0.0001  # padrão: 0.001
```

---

## ⚙️ Verificação do Setup

```bash
# Verificar instalação
python -c "
import tensorflow as tf
import segpy as np
from pathlib import Path

print(f'✅ TensorFlow: {tf.__version__}')
print(f'✅ GPU disponível: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')
print(f'✅ Data dir existe: {Path(\"data\").exists()}')
print(f'✅ Src dir existe: {Path(\"src\").exists()}')
"
```

---

## 🎯 O que você consegue fazer

- ✅ Treinar um classificador CNN do zero
- ✅ Usar Transfer Learning (MobileNetV2)
- ✅ Comparar performance entre modelos
- ✅ Fazer predições em novas imagens
- ✅ Visualizar metrics (confusion matrix, ROC)
- ✅ Deploy com interface web

---

## 💡 Dicas

- **GPU mais rápido**: Instale CUDA/cuDNN para acelerar 10-100x
- **Memória limitada**: Reduza BATCH_SIZE de 32 para 16 ou 8
- **Treino mais rápido**: Use modelos pré-treinados (Transfer Learning)
- **Melhores resultados**: Aumentar dataset com data augmentation

---

## 🆘 Troubleshooting

| Problema | Solução |
|----------|---------|
| Erro de memória | Reduzir BATCH_SIZE |
| Treino muito lento | Ativar GPU ou usar Transfer Learning |
| Data não encontrada | Verificar estrutura em data/ |
| Modelo não carrega | Usar caminho absoluto para arquivo .h5 |

---

**Pronto para começar? Execute `jupyter notebook`! 🎉**
