# 🖼️ Image Classifier com Deep Learning

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

Um projeto completo de classificação de imagens com TensorFlow/Keras, focado em **Transfer Learning (MobileNetV2)** para alta performance e inferência prática.

## 🌐 Demo Online

- Streamlit: https://image-classifier-dl.streamlit.app/

---

## 📋 Indice

- [Características](#características)
- [Configuração](#configuração)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Uso](#uso)
- [Resultados](#resultados)
- [Tecnologias](#tecnologias)
- [Autor](#autor)

---

## ✨ Características

- ✅ **Transfer Learning** - Utiliza MobileNetV2 pré-treinado no ImageNet
- ✅ **Data Augmentation** - Técnicas de aumento de dados para robustez
- ✅ **Métricas Completas** - Accuracy, Precision, Recall, Confusion Matrix, ROC-AUC
- ✅ **Monitoramento** - Early Stopping, Learning Rate Reduction
- ✅ **Interface Web** - Aplicação Streamlit para inferência
- ✅ **Notebook Jupyter** - Tutorial completo passo a passo
- ✅ **Modular** - Código reutilizável e bem organizado

---

## 🚀 Configuração

### Pré-requisitos

- Python 3.9+
- pip ou conda
- Git

### Instalação

1. **Clone o repositório**
```bash
git clone https://github.com/drk7z/image-classifier-deep-learning.git
cd image-classifier-deep-learning
```

2. **Crie um ambiente virtual**
```bash
# Com venv
python -m venv venv

# Ative o ambiente
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

### Download do Dataset

O dataset Cats vs Dogs pode ser obtido em:
- [Microsoft Cats vs Dogs](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip)
- [Kaggle Dataset](https://www.kaggle.com/datasets/shaunacy/catsanddogs)

**Organize os arquivos na seguinte estrutura:**
```
data/
├── train/
│   ├── cats/
│   │   ├── cat_1.jpg
│   │   ├── cat_2.jpg
│   │   └── ...
│   └── dogs/
│       ├── dog_1.jpg
│       ├── dog_2.jpg
│       └── ...
├── validation/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

---

## 📁 Estrutura do Projeto

```
image-classifier-deep-learning/
│
├── 📂 data/                 # Dataset (treino, validação, teste)
│   ├── train/
│   ├── validation/
│   └── test/
│
├── 📂 src/                  # Código-fonte principal
│   ├── model.py            # Arquitetura de Transfer Learning
│   ├── train.py            # Script de treinamento
│   ├── evaluate.py         # Avaliação e métricas
│   └── predict.py          # Predições em novas imagens
│
├── 📂 notebooks/            # Jupyter Notebooks
│   └── 01_cats_vs_dogs_classifier.ipynb
│
├── 📂 models/              # Modelos treinados (.h5)
│
├── 📂 logs/                # TensorBoard logs
│
├── app.py                   # Aplicação Streamlit
├── requirements.txt         # Dependências Python
└── README.md               # Este arquivo
```

---

## 💻 Uso

### 1. Treinar o Modelo (via Script)

```bash
python -m src.train
```

**Customizações:**
```python
from src.train import ImageClassifierTrainer

trainer = ImageClassifierTrainer(data_dir='data')
history = trainer.train(
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    use_augmentation=True
)
```

### 2. Usar o Notebook Jupyter

```bash
jupyter notebook notebooks/01_cats_vs_dogs_classifier.ipynb
```

Siga o notebook passo a passo para:
- Explorar o dataset
- Executar data augmentation
- Treinar o modelo de Transfer Learning
- Comparar desempenho
- Fazer predições

### 3. Executar Avaliação

```bash
python -c "
from src.evaluate import ModelEvaluator

evaluator = ModelEvaluator('models/transfer_learning_final.h5')
metrics = evaluator.evaluate_on_test_set()
evaluator.plot_confusion_matrix()
"
```

### 4. Fazer Predições em Novas Imagens

```python
from src.predict import ImageClassifier

classifier = ImageClassifier(
    model_path='models/transfer_learning_final.h5',
    class_names=['cat', 'dog']
)

# Predição simples
pred_class = classifier.predict('path/to/image.jpg')
print(f"Predição: {pred_class}")

# Com confiança
pred_class, confidence, scores = classifier.predict(
    'path/to/image.jpg',
    return_confidence=True
)
print(f"Predição: {pred_class} ({confidence:.2%})")

# Visualizar predição
classifier.visualize_prediction('path/to/image.jpg', save_path='prediction.png')
```

### 5. Interface Web (Streamlit)

```bash
streamlit run app.py
```

Acesse em: `http://localhost:8501`

Funcionalidades:
- Upload de imagens
- Predição em tempo real
- Gráfico de confiança
- Classificação com Transfer Learning (MobileNetV2)

---

## 📊 Resultados

### Arquitetura do Transfer Learning (MobileNetV2)

```
Input: 224x224x3
    ↓
MobileNetV2 (pré-treinada no ImageNet)
    ↓
GlobalAveragePooling
    ↓
Dense(relu) + Dropout
    ↓
Dense(2, Softmax) → [cat, dog]
```

### Métricas Esperadas

| Métrica | Transfer Learning |
|---------|------------------|
| Accuracy | ~96-98% |
| Precision | ~95-97% |
| Recall | ~95-97% |
| F1-Score | ~96-97% |

### Exemplos de Visualização

- **Curvas de Treino vs Validação**: Monitorar overfitting
- **Matriz de Confusão**: Analisar classificações erradas
- **Curvas ROC**: Avaliar trade-off entre True Positive e False Positive
- **Augmentação de Dados**: Visualizar transformações

---

## 🛠️ Tecnologias

### Deep Learning
- **TensorFlow 2.13+** - Framework de deep learning
- **Keras** - API de alto nível
- **TensorFlow Keras Applications** - Modelos pré-treinados

### Processamento de Imagens
- **OpenCV (cv2)** - Processamento de imagens
- **Pillow (PIL)** - Manipulação de imagens
- **NumPy** - Operações numéricas

### Análise e Visualização
- **Pandas** - Manipulação de dados
- **Matplotlib** - Visualização
- **Seaborn** - Gráficos estatísticos
- **Scikit-learn** - Métricas e avaliação

### Interface Web
- **Streamlit** - App web interativa

---

## 📈 Melhorias Futuras

- [ ] Fine-tuning com camadas desbloqueadas
- [ ] Testar arquiteturas: ResNet50, EfficientNet, InceptionV3
- [ ] Implementar Grad-CAM para explicabilidade
- [ ] Deploy com FastAPI + Docker
- [ ] Otimizar para mobile (TensorFlow Lite)
- [ ] Adicionar suporte a outras classificações (frutas, plantas, etc)
- [ ] Implementar ensemble de modelos
- [ ] Adicionar autenticação à aplicação web

---

## 🎓 Conceitos Abordados

### Deep Learning
- Camadas Convolucionais e Pooling
- Batch Normalization
- Dropout para regularização
- Transfer Learning
- ImageNet pre-training

### Machine Learning
- Validação cruzada
- Data Augmentation
- Early Stopping
- Learning Rate Scheduling
- Métricas de classificação

### Engenharia de Software
- Modularização de código
- Callbacks e logging
- Versionamento de modelos
- Documentação de código

---

## 📖 Referências

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Fast.ai - Practical Deep Learning](https://www.fast.ai/)
- [Stanford CS231n - Computer Vision](http://cs231n.stanford.edu/)

---

## 💡 Tips & Tricks

1. **GPU Training**: Ative GPU para acelerar treinamento
```bash
# Verificar disponibilidade
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

2. **Memory Management**: Para GPUs com pouca memória
```python
# Reduzir batch size
BATCH_SIZE = 16

# Ou usar mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

3. **Visualizar TensorBoard**
```bash
tensorboard --logdir=logs
```

4. **Converter modelo para formato otimizado**
```python
# Para TensorFlow Lite
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('models/transfer_learning_final')
tflite_model = converter.convert()
```

---

## ⚖️ Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👤 Autor

**Leandro Vieira**
- GitHub: [@drk7z](https://github.com/drk7z)
- LinkedIn: [leandroandradeti](https://www.linkedin.com/in/leandroandradeti/)

Desenvolvido com ❤️ para a comunidade de Deep Learning

---

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

## 📞 Suporte

Se tiver dúvidas ou encontrar problemas:

- Abra uma [Issue](https://github.com/drk7z/image-classifier-deep-learning/issues)
- Verifique as [Discussões](https://github.com/drk7z/image-classifier-deep-learning/discussions)
- Entre em contato via email
