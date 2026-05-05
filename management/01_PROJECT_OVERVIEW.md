# 📄 01: Project Overview & Task Definition

## 1. General Description

L'obiettivo principale di questo progetto è affrontare un problema di **Computational Imaging** combinando diverse famiglie metodologiche. Lo scopo è comprendere e confrontare in modo critico gli approcci classici e moderni sotto le **medesime condizioni sperimentali**.
L'attenzione non deve essere focalizzata esclusivamente sull'ottenimento di buoni risultati di ricostruzione, ma soprattutto sul ragionamento critico riguardo i **punti di forza e le limitazioni** di ciascun metodo.

## 2. Dataset Specifics

- **Dataset Selezionato:** LSUN Church
- **Link per il Download:** [https://github.com/fyu/lsun](https://github.com/fyu/lsun)
- **Selezione del Subset:** Il progetto richiede l'utilizzo di un sottoinsieme di circa **4000 immagini** dal dataset originale.
- **Preprocessing:**
  - Tutte le immagini devono essere ridimensionate (resized) a **256x256 pixel**.
  - È richiesta un'attenta ispezione della struttura del dataset.
  - È obbligatorio definire e implementare una *preprocessing pipeline* coerente.
  - Le operazioni includono la **normalizzazione** (valori dei pixel) ed eventuali adattamenti.
  - I dati devono essere suddivisi rigorosamente in **Training, Validation e Testing sets** (quando applicabile, ad es. per i modelli end-to-end).
  - **Requisito Critico:** Qualsiasi scelta effettuata durante la fase di preprocessing deve essere documentata in modo meticoloso e logicamente giustificata.

## 3. Task Definition: Superresolution

Il task in oggetto è la **Super-Risoluzione** (Superresolution). Deve essere formulato come un *problema inverso* in cui un'osservazione degradata viene mappata nuovamente in una ricostruzione ad alta qualità (high-quality reconstruction).

### Parametri del Processo di Degradazione

L'esperimento richiede di operare sotto specifiche condizioni di degradazione che combinano il fattore di super-risoluzione (SR factor) con l'aggiunta di rumore (noise level).

| SR Factor | Noise Level |
| :---: | :---: |
| 2 | 0.005 |
| 2 | 0.01 |
| 4 | 0.005 |
| 4 | 0.01 |

**Requisito Critico Indispensabile:** Per garantire l'equità e la correttezza metodologica del confronto, gli studenti devono assicurarsi, a livello di codice, che **tutti i metodi implementati ricevano e utilizzino esattamente gli stessi input degradati**.
