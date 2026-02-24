# 🧠 Alzheimer MRI Classification using Deep Learning
Approcci di Deep Learning per la diagnosi dell’Alzheimer a partire da immagini MRI cerebrali.

## 📌 Progetto DLCMM – NeuroVision DL

Repository : `unict-dlcmm-Maccarrone-Brancaforte-Cassia`
per il progetto di **Deep Learning Modulo Core Models and Methods** del laboratorio.  

### Progetto
- Gruppo: Team **Alzheimer MRI**
- Nome del progetto: **NeuroVision** DL
- Descrizione breve del progetto: Il progetto si concentra sull’applicazione di tecniche di Deep Learning all’analisi di **immagini di Risonanza Magnetica (MRI) cerebrale** con l’obiettivo di supportare il riconoscimento automatico dell’**Alzheimer** e dei suoi diversi **stadi di progressione**.


### Membri del gruppo
- Maccarrone Alessia
- Martina Brancaforte
- Massimiliano Cassia

---

## 🎯 Obiettivo del progetto

L'obiettivo del progetto è sviluppare e confrontare diversi modelli di **Deep Learning** per il **riconoscimento dell'Alzheimer a partire da immagini MRI cerebrali**.

Il lavoro è articolato in tre fasi principali:

1. **Classificazione binaria**: Alzheimer vs Non-Alzheimer
2. **Classificazione multiclasse**: 4 stadi di deterioramento cognitivo
3. **Demo web interattiva** per la predizione su nuove immagini

---

## 📂 Struttura del progetto

```
unict-dlcmm-Maccarrone-Brancaforte-Cassia/
│
├── data/                   # Dataset e file di dati
│   ├── raw/                # Dataset originale Kaggle
│   ├── processed/          # Dataset preprocessato
│   └── splits/             # Train / Validation / Test
│
│
├── docs/                   # Documentazione e report
│
├── media/                  # Video demo, immagini, screenshot
│
├── notebooks/              # Jupyter notebooks per analisi ed esperimenti
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_binary_classification.ipynb
│   └── 04_multiclass_classification.ipynb
│
│
├── results/                # Risultati, log, metriche, modelli
│
├── src/                    # Codice sorgente
│   ├── data/               # Preprocessing e DataLoader
│   ├── models/             # Modelli Deep Learning
│   ├── training/           # Script di training
│   ├── evaluation/         # Metriche e valutazione
│   └── utils/              # Funzioni di supporto
│
│
└── README.md   # Questo file
```

---

## 📂 Dataset

Il dataset utilizzato è l'**Augmented Alzheimer MRI Dataset**, disponibile su Kaggle con Immagini MRI cerebrali al link: https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset

### 🔹 Classificazione binaria

| Classe binaria | Classi originali                                 |
| -------------- | ------------------------------------------------ |
| Non-Alzheimer  | NonDemented                                      |
| Alzheimer      | VeryMildDemented, MildDemented, ModerateDemented |

### 🔹 Classificazione multiclasse

| Label | Classe           |
| ----- | ---------------- |
| 0     | NonDemented      |
| 1     | VeryMildDemented |
| 2     | MildDemented     |
| 3     | ModerateDemented |

---

## 🧠 Modelli utilizzati

### 🔹 Classificazione binaria 

*
*

### 🔹 Classificazione multiclasse

*
*
*
*

---

## 📊 Metriche di valutazione

* Accuracy
* Precision
* Recall
* F1-score (macro)
* Confusion Matrix
* ROC Curve (multiclasse one-vs-rest)

---

## 🌐 Demo Web

È stata realizzata una demo web interattiva che consente di:

* Caricare un'immagine MRI
* Selezionare il modello
* Visualizzare la classe predetta
* Mostrare le probabilità di classificazione

La demo è sviluppata utilizzando **Streamlit**.

Avvio della demo:

```bash
cd demo
streamlit run app.py
```

---

## ⚙️ Installazione e riproducibilità

1. Clonare il repository:

```bash
git clone https://github.com/MaccarroneAlessia/unict-dlcmm-Maccarrone-Brancaforte-Cassia
cd unict-dlcmm-Maccarrone-Brancaforte-Cassia
```

2. Installare le dipendenze:

```bash
pip install -r requirements.txt
```

3. Scaricare il dataset da Kaggle e inserirlo in:

```text
data/raw/
```

4. Eseguire i notebook in ordine numerico.

---

## 📌 Note finali

Il progetto dimostra come tecniche di **Deep Learning applicate a immagini mediche** possano supportare la diagnosi precoce dell'Alzheimer, confrontando modelli differenti e fornendo una demo utilizzabile anche da utenti non tecnici.

---

## 🎓 Corso

**Deep Learning: Core Models and Methods**
Università degli Studi di Catania
