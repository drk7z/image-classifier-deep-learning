# 👋 Projeto de Portfólio — Classificador de Imagens de Pets (DL + MLOps + Segurança)

## 🎯 Resumo do Projeto
Este repositório apresenta um produto de IA ponta a ponta focado em **classificação de imagens** (Gato vs Cachorro) utilizando **Deep Learning** com uma abordagem orientada à produção.

Não é apenas um notebook de treinamento de modelo — ele inclui:
- Uma aplicação deployável em Streamlit
- Manipulação segura de upload de arquivos
- Pipeline de CI com auditoria de vulnerabilidades em dependências
- Execução containerizada com Docker para ambientes consistentes

## 💼 Por que este projeto é relevante
Este projeto demonstra habilidades práticas valorizadas no mercado:
- **Engenharia de Machine Learning**: treinamento, avaliação e inferência de modelos
- **Engenharia de Software**: arquitetura modular em Python e componentes reutilizáveis
- **Product Thinking**: aplicação voltada ao usuário com UX intuitiva
- **Segurança & DevOps**: verificações em CI, análise de vulnerabilidades e hardening de container

## 🧠 O que a aplicação faz
A aplicação recebe uma imagem enviada pelo usuário e prevê se ela é:
- 🐱 Gato
- 🐶 Cachorro

Utiliza **Transfer Learning (MobileNetV2)** para aproveitar features visuais pré-treinadas e retorna:
- Classe prevista
- Score de confiança
- Gráfico de confiança para ambas as classes

## 🏗️ Stack Técnica
- Python
- TensorFlow / Keras
- Streamlit
- NumPy / Pillow / Matplotlib
- GitHub Actions (CI)
- Docker

## 🔐 Destaques de Segurança & Confiabilidade
- Validação de tipo de arquivo e MIME no upload
- Validação de integridade da imagem antes da predição
- Limite de tamanho de upload
- Manipulação segura de arquivos temporários
- CI com `pip-audit` para detectar CVEs conhecidas nas dependências
- Execução do container como usuário não-root

## 📁 Visão Geral da Arquitetura
- `src/model.py`: definições do modelo
- `src/train.py`: pipeline de treinamento
- `src/evaluate.py`: avaliação do modelo
- `src/predict.py`: lógica de inferência
- `app.py`: frontend em Streamlit
- `.github/workflows/ci.yml`: verificações de segurança e qualidade no CI
- `Dockerfile`: containerização pronta para produção

## 🚀 Execução Rápida
1. Instale as dependências:
   - `pip install -r requirements.txt`
2. Inicie a aplicação:
   - `streamlit run app.py`
3. Abra no navegador:
   - `http://localhost:8501`

## 📌 Observações
Se você estiver avaliando este repositório:
- Este projeto foi desenvolvido para demonstrar tanto **capacidade técnica em ML** quanto **maturidade na entrega de software**.
- O código reflete preocupação com **legibilidade, manutenibilidade e padrões seguros por padrão**.
- Pode ser facilmente estendido para classificação multi-classe e deploy em cloud.

## 📫 Contato
Se for útil, posso oferecer um walkthrough guiado sobre:
- decisões de treinamento do modelo
- trade-offs de performance
- escolhas de hardening para produção

## 🌐 Links Profissionais

<a href="https://www.linkedin.com/in/leandroandradeti/" target="_blank" rel="noopener noreferrer" style="text-decoration:none; display:inline-flex; align-items:center; gap:8px; margin-right:16px;">
   <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M4.98 3.5C4.98 4.88 3.86 6 2.48 6S0 4.88 0 3.5 1.12 1 2.5 1s2.48 1.12 2.48 2.5zM.5 8h4V24h-4V8zm7 0h3.8v2.2h.1c.53-1 1.83-2.2 3.77-2.2C19.2 8 21 10.2 21 14v10h-4v-8.5c0-2-.03-4.5-2.75-4.5-2.75 0-3.17 2.15-3.17 4.36V24h-4V8z"/>
   </svg>
   <span>LinkedIn</span>
</a>

<a href="https://github.com/drk7z" target="_blank" rel="noopener noreferrer" style="text-decoration:none; display:inline-flex; align-items:center; gap:8px;">
   <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.1 3.3 9.43 7.88 10.96.58.1.79-.25.79-.56v-2.16c-3.2.7-3.87-1.35-3.87-1.35-.52-1.33-1.28-1.68-1.28-1.68-1.05-.72.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.03 1.75 2.7 1.25 3.36.95.1-.75.4-1.26.73-1.55-2.55-.29-5.23-1.27-5.23-5.68 0-1.26.45-2.3 1.18-3.12-.12-.29-.51-1.45.11-3.03 0 0 .97-.31 3.19 1.19a11.1 11.1 0 0 1 5.8 0c2.22-1.5 3.18-1.19 3.18-1.19.63 1.58.24 2.74.12 3.03.74.82 1.18 1.86 1.18 3.12 0 4.42-2.69 5.39-5.26 5.67.41.35.78 1.05.78 2.12v3.14c0 .31.21.67.8.56A11.52 11.52 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5z"/>
   </svg>
   <span>GitHub</span>
</a>
