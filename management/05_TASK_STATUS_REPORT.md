# 📊 05: Task Status Report

Di seguito l'analisi dello stato di avanzamento per ciascun task definito in `04_TASK_TRACKER.md`. L'analisi è basata sul codice attualmente presente nel progetto.

## Fase 1: Data Engineering e Task Initialization

| Task ID | Status | Valutazione e Commenti |
|:---|:---:|:---|
| **1.1** | ✅ | **Svolto correttamente:** Abbiamo deciso di mantenere *HuggingFace* in streaming (`src/dataset.py`) al posto del repo GitHub originale per motivi di ottimizzazione (evitando il download di 40GB non necessari). Da giustificare nel report. |
| **1.2** | ✅ | **Svolto:** Comprensione effettuata (esiste il notebook `01_data_exploration.ipynb` e la logica in `dataset.py` è corretta per immagini raw). |
| **1.3** | ✅ | **Svolto correttamente:** Lo script `src/dataset.py` estrae correttamente il subset di 4000 immagini. |
| **1.4** | ✅ | **Svolto correttamente:** Il resizing a 256x256 con interpolazione bicubica è implementato correttamente in `src/dataset.py`. |
| **1.5** | ✅ | **Svolto correttamente:** Normalizzazione `[0, 1]` codificata ufficialmente in `src/normalization.py`. Giustificazione matematica redatta nel documento dedicato `management/06_NORMALIZATION_JUSTIFICATION.md`. |
| **1.6** | ✅ | **Svolto correttamente:** La divisione in Train (80%), Val (10%) e Test (10%) è logicamente corretta e presente in `src/dataset.py`. |
| **1.7** | ✅ | **Svolto correttamente:** La classe `DegradationOperator` in `src/degradations.py` implementa correttamente il downsampling con fattore 2 e 4. |
| **1.8** | ✅ | **Svolto correttamente:** L'aggiunta di rumore Gaussiano (livelli 0.005 e 0.01) è implementata in `src/degradations.py`. |
| **1.9** | ✅ | **Svolto correttamente:** La funzione `create_degraded_dataset` permette di generare e salvare offline il test set degradato, garantendo un input unico e deterministico per tutti i modelli. |

## Fase 2: Implementazione Metodi di Ricostruzione

| Task ID | Status | Valutazione e Commenti |
|:---|:---:|:---|
| **2.1.a-c** | ❌ | **Non svolto:** Il file `src/methods/tv.py` è stato creato, ma è **completamente vuoto (0 bytes)**. Tutta la logica matematica Variazionale (Total Variation) deve essere scritta. |
| **2.2.a-c** | ❌ | **Non svolto:** I file `src/methods/unet.py` e `src/train_unet.py` sono **completamente vuoti (0 bytes)**. L'architettura End-to-End e il loop di training sono da sviluppare. |
| **2.3.a-c** | ❌ | **Non svolto:** Il file `src/methods/diffpir.py` è **completamente vuoto (0 bytes)**. Manca l'implementazione del prior Generativo (DiffPIR). |
| **2.4.a-c** | ❌ | **Non svolto / Scartato:** Non c'è alcuna traccia dell'implementazione del metodo Ibrido (HQS/PnP). Se è il metodo scartato come indicato dai documenti di management, va esplicitato che non verrà svolto. |

## Fase 3: Testing, Metriche e Plot

| Task ID | Status | Valutazione e Commenti |
|:---|:---:|:---|
| **3.1 - 3.5** | ❌ | **Non svolti:** Il file `src/metrics.py` è **completamente vuoto (0 bytes)**. Inoltre, il notebook `02_results_comparison.ipynb` è in uno stato preliminare poiché nessun metodo produce ancora output. Mancano tutte le implementazioni matematiche di PSNR e SSIM e le logiche di visualizzazione. |

## Fase 4: Produzione Deliverables

| Task ID | Status | Valutazione e Commenti |
|:---|:---:|:---|
| **4.1 - 4.3** | ❌ | **Iniziati / Da fare:** La codebase (docstrings) ha un buon livello su quanto fatto finora (in `degradations.py` in particolare), ma andrà fatta per il resto. Report e slide sono completamente da redigere in futuro, sulla base dei risultati ottenuti nella Fase 2 e 3. |
