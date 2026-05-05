# 📋 04: Master Task Tracker e Workdown Structure

*Nota: Assegnare il nominativo ai vari task e aggiornare lo stato inserendo una 'x' tra le parentesi: [x]*

## Fase 1: Data Engineering e Task Initialization

| Status | Task ID | Descrizione Dettagliata | Assegnato A | Note / Risultati |
|:---:|:---|:---|:---|:---|
| [x] | 1.1 | Download repository LSUN da GitHub. | | Fatto tramite HuggingFace (Ottimizzato, evita 40GB) |
| [x] | 1.2 | Comprensione della struttura raw del dataset LSUN. | | |
| [x] | 1.3 | Script per selezione subset di circa 4.000 immagini della categoria 'Church'. | | |
| [x] | 1.4 | Script di Resizing: applicazione rigorosa ridimensionamento a 256x256. | | |
| [x] | 1.5 | Definizione e applicazione Normalizzazione Dati (giustificare matematicamente). | | Fatto in normalization.py. Giust. in 06_NORMALIZATION_JUSTIFICATION.md |
| [x] | 1.6 | Script di Splitting: divisione strutturata in Train, Validation e Testing sets. | | |
| [x] | 1.7 | Implementazione della funzione di Degradazione (SR Factor: 2, 4). | | |
| [x] | 1.8 | Aggiunta rumore (Noise Level: 0.005, 0.01) coerente post-SR. | | |
| [x] | 1.9 | Creazione pipeline assicurativa: garantire un unico DataLoader con gli *stessi input degradati* per tutti i metodi. | | `create_degraded_dataset` in degradations.py |

## Fase 2: Implementazione Metodi di Ricostruzione

*Da decidere in fase di kick-off: Metodo scartato tra End-to-End, Generativo e Ibrido.*

| Status | Task ID | Descrizione Dettagliata | Assegnato A | Note / Risultati |
|:---:|:---|:---|:---|:---|
| [ ] | 2.1.a | **Variazionale:** Stesura codice ottimizzazione matematica. | | |
| [ ] | 2.1.b | **Variazionale:** Implementazione prior Total Variation (TV). | | |
| [ ] | 2.1.c | **Variazionale:** Ricerca euristica parametri di regolarizzazione e tracciamento storico esperimenti. | | |
| [ ] | 2.2.a | **End-to-End (Opz. 1):** Scelta logica e studio architettura (UNet, ViT o NAF-Net). | | |
| [ ] | 2.2.b | **End-to-End (Opz. 1):** Implementazione architettura selezionata. | | |
| [ ] | 2.2.c | **End-to-End (Opz. 1):** Training run e tuning parametri. | | |
| [ ] | 2.3.a | **Generativo (Opz. 2):** Studio dell'algoritmo DiffPIR. | | |
| [ ] | 2.3.b | **Generativo (Opz. 2):** Implementazione prior di diffusione (DiffPIR). | | |
| [ ] | 2.3.c | **Generativo (Opz. 2):** Ottimizzazione parametri e step di diffusione. | | |
| [ ] | 2.4.a | **Ibrido (Opz. 3):** Implementazione framework iterativo Half Quadratic Splitting (HQS). | | Probabile Metodo Scartato |
| [ ] | 2.4.b | **Ibrido (Opz. 3):** Integrazione logica Plug and Play (PnP). | | Probabile Metodo Scartato |
| [ ] | 2.4.c | **Ibrido (Opz. 3):** Tuning euristico su pesi e num iterazioni. | | Probabile Metodo Scartato |

## Fase 3: Testing, Metriche e Plot

| Status | Task ID | Descrizione Dettagliata | Assegnato A | Note / Risultati |
|:---:|:---|:---|:---|:---|
| [ ] | 3.1 | Sviluppo script di test massivo su Test Set. | | |
| [ ] | 3.2 | Funzione per calcolo Peak Signal-to-Noise Ratio (PSNR). | | |
| [ ] | 3.3 | Funzione per calcolo Structural Similarity Index (SSIM). | | |
| [ ] | 3.4 | Script per salvataggio figure composite (Ground Truth vs Degraded vs Reconstructions). | | |
| [ ] | 3.5 | Sviluppo script Python (matplotlib/seaborn) per generare il "Comparison Plot" riassuntivo (su 4 varianti degradazione). | | |

## Fase 4: Produzione Deliverables

| Status | Task ID | Descrizione Dettagliata | Assegnato A | Note / Risultati |
|:---:|:---|:---|:---|:---|
| [ ] | 4.1.a | Codebase: refactoring e ottimizzazione architettura directory. | | |
| [ ] | 4.1.b | Codebase: redazione estensiva di Docstrings e Commenti. | | |
| [ ] | 4.1.c | GitHub: setup repo, commit finale e generazione Link. | | |
| [ ] | 4.2.a | Report: stesura astrazione, dataset info, e metodi. | | |
| [ ] | 4.2.b | Report: stesura giustificazioni setup/preprocessing. | | |
| [ ] | 4.2.c | Report: stesura analisi tuning euristico. | | |
| [ ] | 4.2.d | Report: redazione discussione finale (forza e limiti). | | |
| [ ] | 4.3.a | Slide: strutturazione template Beamer o PowerPoint. | | |
| [ ] | 4.3.b | Slide: importazione di grafici e confronti visivi e stesura punti discussione. | | |
