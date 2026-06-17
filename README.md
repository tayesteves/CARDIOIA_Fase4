<p align="center">
  <img src="assets/logo-fiap.png" alt="Logo FIAP" width="320"/>
</p>

<h1 align="center">🫀 CardioIA — Fase 4</h1>

<h3 align="center">Assistente Cardiológico Virtual com Visão Computacional</h3>

<p align="center">
  Classificação de imagens médicas com CNN · FIAP · Inteligência Artificial · 2026
</p>

---

## 📌 Identificação do Projeto

| Item | Descrição |
| --- | --- |
| **Instituição** | FIAP |
| **Curso** | Tecnólogo em Inteligência Artificial |
| **Fase** | Fase 4 |
| **Projeto** | CardioIA — Assistente Cardiológico Virtual |
| **Tema** | Visão Computacional aplicada à saúde |
| **Ano** | 2026 |
| **Grupo** | 88 |

---

## 🧠 Sobre o Projeto

O **CardioIA** é um assistente virtual voltado à análise de imagens médicas com técnicas de **Visão Computacional** e **Redes Neurais Convolucionais (CNNs)**.

Nesta fase, o projeto evolui do monitoramento contínuo de saúde para a análise de imagens, explorando um pipeline completo:

```text
Dataset médico → Pré-processamento → Treinamento CNN → Avaliação → Protótipo de uso
```

Como base pública de imagens médicas, foi utilizado o dataset **Brain Tumor MRI Dataset**, disponível no Kaggle. Embora o nome do projeto seja CardioIA, o dataset de ressonância magnética cerebral foi adotado como base pública estruturada para validação do pipeline de visão computacional. A arquitetura proposta pode ser adaptada futuramente para exames cardiológicos, como ECGs digitalizados, raios-X de tórax ou outros exames de imagem relacionados à saúde cardiovascular.

---

## 🎯 Objetivo

Desenvolver um protótipo funcional capaz de:

| Objetivo | Descrição |
| --- | --- |
| **Pré-processar imagens médicas** | Aplicar redimensionamento, conversão de formato, normalização e organização em treino, validação e teste |
| **Treinar modelos CNN** | Implementar uma CNN do zero e uma abordagem de Transfer Learning com VGG16 |
| **Avaliar desempenho** | Medir acurácia, precisão, recall, F1-score e matriz de confusão |
| **Apresentar resultados** | Disponibilizar resultados em notebook e protótipo com app Expo + backend Flask |
| **Documentar decisões** | Explicar escolhas técnicas, limitações e cuidados éticos no uso de IA em saúde |

---

## 🗂 Dataset

**Dataset utilizado:** Brain Tumor MRI Dataset — Kaggle

O dataset contém imagens de ressonância magnética cerebral organizadas em quatro classes:

| Classe | Descrição |
| --- | --- |
| `glioma` | Imagens com tumor do tipo glioma |
| `meningioma` | Imagens com tumor do tipo meningioma |
| `notumor` | Imagens sem tumor |
| `pituitary` | Imagens com tumor na região da hipófise |

### Distribuição após o Pipeline

| Conjunto | Total de imagens | Imagens por classe |
| --- | :---: | :---: |
| **Treino** | 4.760 | 1.190 |
| **Validação** | 840 | 210 |
| **Teste** | 1.600 | 400 |
| **Total** | 7.200 | — |

### Visualizações do Dataset

| Distribuição do Dataset | Exemplos por Classe |
| :---: | :---: |
| ![Distribuição](data/Figura%201%20—%20distribuicao_dataset.png) | ![Exemplos](data/Figura%202%20-%20exemplos_classes.png) |

| Antes e Depois do Pré-processamento | Dashboard Final |
| :---: | :---: |
| ![Pré-processamento](data/Figura%203%20-%20antes_depois_preprocessamento.png) | ![Dashboard](data/Figura%204%20-%20dashboard_dataset_final.png) |

---

## 🏗 Arquitetura da Solução

<p align="center">
  <img src="assets/Arquitetura de solução.png" alt="Arquitetura da Solução CardioIA" width="900"/>
</p>

---

## ⚙️ Pipeline de Pré-processamento

**Notebook:** `notebooks/01_preprocessamento_pipeline.ipynb`

**Colab:** [Abrir notebook de pré-processamento](https://colab.research.google.com/drive/1S-5SZZlKrsEn6lZ6APxJXYsTi5yX8mSX)

### Etapas executadas

| Etapa | Descrição |
| --- | --- |
| **1. Carregamento do dataset** | Extração do arquivo `.zip` original e leitura das pastas de treino e teste |
| **2. Inspeção inicial** | Contagem de imagens por classe e visualização de amostras |
| **3. Conversão de imagem** | Conversão para RGB para compatibilidade com redes pré-treinadas |
| **4. Redimensionamento** | Padronização para 224×224 pixels |
| **5. Normalização** | Escalonamento dos pixels para o intervalo adequado ao treinamento |
| **6. Separação dos dados** | Criação dos conjuntos `train`, `val` e `test` |
| **7. Geração de metadados** | Criação do arquivo `metadata_dataset.csv` |
| **8. Geração de visualizações** | Gráficos e imagens utilizadas na documentação |

### Saída do Pipeline

```text
data/
├── metadata_dataset.csv
├── Figura 1 — distribuicao_dataset.png
├── Figura 2 - exemplos_classes.png
├── Figura 3 - antes_depois_preprocessamento.png
├── Figura 4 - dashboard_dataset_final.png
└── processed/
    ├── train/
    │   ├── train_glioma.zip
    │   ├── train_meningioma.zip
    │   ├── train_notumor.zip
    │   └── train_pituitary.zip
    ├── val/
    │   ├── val_glioma.zip
    │   ├── val_meningioma.zip
    │   ├── val_notumor.zip
    │   └── val_pituitary.zip
    └── test/
        ├── test_glioma.zip
        ├── test_meningioma.zip
        ├── test_notumor.zip
        └── test_pituitary.zip
```

---

## 🤖 Modelos de IA

**Notebook:** `notebooks/02_modelos_cnn.ipynb`

O projeto contempla duas abordagens de classificação de imagens:

### 1. CNN treinada do zero

Rede convolucional simples criada em Keras/TensorFlow, com blocos de convolução, pooling, camada densa e saída `softmax` para quatro classes.

```python
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(4, activation="softmax")
])
```

### 2. Transfer Learning com VGG16

Modelo baseado na arquitetura VGG16 pré-treinada, utilizando pesos do ImageNet e adaptação do topo para classificação das quatro classes médicas do projeto.

```python
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(4, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
```

### Métricas previstas

| Métrica | Finalidade |
| --- | --- |
| **Acurácia** | Mede o percentual geral de classificações corretas |
| **Precisão** | Mede a confiabilidade das predições positivas |
| **Recall** | Mede a capacidade de recuperar corretamente os casos de cada classe |
| **F1-score** | Equilibra precisão e recall |
| **Matriz de confusão** | Mostra onde o modelo acerta e onde confunde classes |

> Observação: o notebook `02_modelos_cnn.ipynb` contém a estrutura de treinamento, avaliação e comparação entre CNN do zero e VGG16. A execução completa depende da disponibilidade do dataset processado no ambiente Colab/Drive.

---

## 🖥 Backend Flask

O backend foi desenvolvido com Flask para receber imagens via requisição HTTP, aplicar o pré-processamento necessário e retornar uma resposta JSON para a interface.

### Endpoints

| Método | Rota | Descrição |
| --- | --- | --- |
| `GET` | `/` | Verifica se a API está ativa |
| `GET` | `/health` | Retorna status do servidor, classes e estado do modelo |
| `GET` | `/classes` | Retorna as classes disponíveis |
| `POST` | `/predict` | Recebe uma imagem e retorna a predição |

### Exemplo de resposta do `/health`

```json
{
  "status": "ok",
  "model_loaded": false,
  "classes": ["glioma", "meningioma", "notumor", "pituitary"]
}
```

### Exemplo de resposta do `/predict`

```json
{
  "predicted_class": "glioma",
  "confidence": 34.7,
  "probabilities": {
    "glioma": 34.7,
    "meningioma": 22.1,
    "notumor": 18.5,
    "pituitary": 24.7
  },
  "is_placeholder": true
}
```

> Enquanto o modelo real `.h5` não estiver integrado ao backend, a API retorna uma predição simulada com `"is_placeholder": true`. Esse modo permite validar o fluxo completo entre app, backend e tela de resultado.

---

## 📱 Protótipo

**Responsável:** Carlos Eduardo de Souza — RM566487

O protótipo foi desenvolvido com **React Native**, **Expo SDK 51** e **Expo Router**.

### Funcionalidades

| Funcionalidade | Descrição |
| --- | --- |
| **Selecionar imagem** | Permite escolher uma imagem médica pela galeria |
| **Capturar imagem** | Estrutura preparada para uso da câmera |
| **Enviar para análise** | Envia a imagem ao backend Flask |
| **Visualizar resultado** | Exibe classe prevista, confiança e probabilidades |
| **Aviso clínico** | Reforça que o sistema é apenas um protótipo acadêmico |

### Fluxo de uso

```text
1. Abrir o app
2. Selecionar imagem pela galeria ou câmera
3. Enviar imagem para análise
4. Backend processa a imagem
5. Interface exibe classe, confiança e probabilidades
```

### Código de cores dos resultados

| Cor | Classe | Interpretação visual |
| --- | --- | --- |
| 🟢 Verde | `notumor` | Sem tumor identificado pelo modelo |
| 🟡 Amarelo | `meningioma` / `pituitary` | Classe tumoral de menor criticidade relativa |
| 🔴 Vermelho | `glioma` | Classe tumoral de maior criticidade relativa |

---

## 🛠 Tecnologias Utilizadas

| Categoria | Tecnologias |
| --- | --- |
| **Linguagem principal** | Python |
| **IA / Deep Learning** | TensorFlow, Keras, VGG16 |
| **Análise e métricas** | NumPy, Pandas, Scikit-learn |
| **Visualização** | Matplotlib, Seaborn |
| **Backend** | Flask, Flask-CORS, Pillow |
| **Mobile/Web** | React Native, Expo SDK 51, Expo Router |
| **Ambiente de execução** | Google Colab, VS Code |
| **Versionamento** | Git e GitHub |

---

## 🚀 Como Executar

### Pré-requisitos

| Ferramenta | Versão recomendada | Observação |
| --- | --- | --- |
| Python | 3.10+ | Para backend Flask e notebooks |
| Node.js | 18+ | Para rodar o projeto Expo |
| Expo | SDK 51 | O projeto foi mantido em SDK 51 |
| Google Colab | GPU opcional | Recomendado para treinamento dos modelos |
| Git | Atual | Para clonar e versionar o projeto |

> Caso o Expo Go do celular esteja em versão mais recente e acuse incompatibilidade com SDK 51, utilize o modo web com `npx expo start` e pressione `w`.

---

### 1. Clonar o repositório

```powershell
git clone https://github.com/tayesteves/CARDIOIA_Fase4.git
cd CARDIOIA_Fase4
```

---

### 2. Rodar o Backend Flask

```powershell
cd cardioia\backend

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

python app.py
```

A API ficará disponível em:

```text
http://localhost:5000
```

Teste:

```powershell
curl http://localhost:5000/health
```

Ou abra no navegador:

```text
http://localhost:5000/health
```

---

### 3. Rodar o App Expo

Em outro terminal:

```powershell
cd cardioia\mobile

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

npm install
npx expo start --clear
```

Para abrir no navegador, pressione:

```text
w
```

Caso o modo web peça dependências extras:

```powershell
npx expo install react-native-web react-dom
```

Depois rode novamente:

```powershell
npx expo start --clear
```

---

### 4. Configurar o IP do Backend no App

O arquivo de configuração fica em:

```text
cardioia/mobile/app/config/api.js
```

Para uso no navegador, `localhost` funciona.

Para uso no celular, substitua o IP pelo IPv4 da sua máquina:

```js
const DEV_IP = "10.0.0.59";
```

Para descobrir o IP no Windows:

```powershell
ipconfig | findstr "IPv4"
```

> Celular e computador precisam estar na mesma rede Wi-Fi.

---

### 5. Rodar os Notebooks

#### Notebook de pré-processamento

```text
notebooks/01_preprocessamento_pipeline.ipynb
```

Também disponível no Colab:

```text
https://colab.research.google.com/drive/1S-5SZZlKrsEn6lZ6APxJXYsTi5yX8mSX
```

#### Notebook de modelos CNN

```text
notebooks/02_modelos_cnn.ipynb
```

Para executar corretamente, o dataset processado precisa estar acessível no ambiente Colab/Drive ou dentro da estrutura `data/processed`.

---

## 📁 Estrutura do Repositório

```text
CARDIOIA_Fase4/
│
├── assets/
│   ├── logo-fiap.png                    # Logo da FIAP usado no início do README
│   └── arquitetura-solucao.png           # Imagem da arquitetura da solução
│
├── cardioia/
│   ├── README.md                         # Documentação específica do módulo app/backend
│   │
│   ├── backend/
│   │   ├── app.py                        # API Flask com rotas /, /health, /classes e /predict
│   │   └── requirements.txt              # Dependências específicas do backend Flask
│   │
│   └── mobile/
│       ├── app.json                      # Configuração do projeto Expo
│       ├── package.json                  # Dependências do app React Native / Expo
│       ├── babel.config.js               # Configuração Babel
│       ├── metro.config.js               # Configuração Metro Bundler
│       │
│       └── app/
│           ├── _layout.jsx               # Configuração de navegação com Expo Router
│           ├── index.jsx                 # Tela inicial para seleção/upload da imagem
│           ├── result.jsx                # Tela de exibição do resultado da análise
│           │
│           └── config/
│               └── api.js                # Configuração da URL do backend Flask
│
├── data/
│   ├── README.md                         # Documentação do dataset e do pipeline de dados
│   ├── metadata_dataset.csv              # Metadados com caminho, classe e split das imagens
│   ├── Figura 1 — distribuicao_dataset.png
│   ├── Figura 2 - exemplos_classes.png
│   ├── Figura 3 - antes_depois_preprocessamento.png
│   ├── Figura 4 - dashboard_dataset_final.png
│   │
│   ├── raw/
│   │   └── README.md                     # Orientações sobre o dataset bruto
│   │
│   └── processed/
│       ├── train/                        # Arquivos compactados do conjunto de treino
│       ├── val/                          # Arquivos compactados do conjunto de validação
│       └── test/                         # Arquivos compactados do conjunto de teste
│
├── notebooks/
│   ├── 01_preprocessamento_pipeline.ipynb # Notebook do pipeline de pré-processamento
│   └── 02_modelos_cnn.ipynb               # Notebook com CNN do zero e Transfer Learning com VGG16
│
├── outputs/
│   ├── figures/                          # Figuras e gráficos gerados pelos notebooks
│   ├── metrics/                          # Métricas, relatórios e matrizes de confusão
│   └── samples/                          # Amostras de predições ou imagens usadas nos testes
│
├── .gitignore                            # Arquivos e pastas ignorados pelo Git
├── README.md                             # Documentação principal do projeto
└── requirements.txt                      # Dependências Python gerais do projeto
```

---

## 📊 Critérios de Avaliação Atendidos

| Critério | Status | Evidência |
| --- | :---: | --- |
| Pipeline de pré-processamento | ✅ | `notebooks/01_preprocessamento_pipeline.ipynb` |
| Organização treino/validação/teste | ✅ | `data/processed/` e `metadata_dataset.csv` |
| CNN do zero | ✅ | `notebooks/02_modelos_cnn.ipynb` |
| Transfer Learning | ✅ | Implementação com VGG16 |
| Métricas de avaliação | ✅ | Accuracy, Precision, Recall, F1-score e matriz de confusão no notebook |
| Protótipo simples | ✅ | App Expo + Backend Flask |
| Documentação clara | ✅ | README, notebooks e estrutura do projeto |
| Trabalho em equipe | ✅ | 4 integrantes |


---

## 👥 Integrantes

| Nome | RM | Função | E-mail |
| --- | --- | --- | --- |
| Tayná Esteves | RM562491 | Engenharia de Dados e Pipeline | Preencher e-mail |
| João Vittor Fontes | RM565999 | Cientista de IA e Modelos CNN | Preencher e-mail |
| Carlos Eduardo de Souza | RM566487 | Desenvolvimento de Interface Mobile | Preencher e-mail |
| Endrew Alves dos Santos | RM563646 | Documentação e Gestão da Entrega | endrewalves42@gmail.com |

---

## 👨‍🏫 Professores

### Tutor

- [Caique Nonato da Silva Bezerra](https://linkedin.com/in/caique-nonato)

### Coordenador

- [André Godoi Chiovato](https://www.linkedin.com/in/andregodoichiovato/)

---

## 📌 Observações Finais

- O backend Flask funciona em modo placeholder enquanto o modelo `.h5` real não estiver integrado.
- O modo placeholder é utilizado apenas para validar o fluxo entre frontend, backend e tela de resultado.
- O app foi validado no navegador por meio do Expo Web.
- O projeto utiliza um dataset público de imagens médicas como base para demonstrar o pipeline de visão computacional.
- A solução é acadêmica e não substitui profissionais de saúde.

---

<p align="center">
  <strong>FIAP · Inteligência Artificial · Fase 4 · 2026</strong>
</p>

<p align="center">
  Desenvolvido para fins acadêmicos por estudantes da FIAP.
</p>
