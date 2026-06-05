# Computational Imaging - Kaggle Setup Guide

Questo file spiega come importare il progetto su Kaggle, configurare l'ambiente e caricare correttamente il dataset personalizzato per replicare i risultati dell'addestramento e dell'inferenza.

## 1. Importare il Notebook su Kaggle
1. Accedi a [Kaggle](https://www.kaggle.com/) e vai nella sezione **Code**.
2. Clicca su **New Notebook**.
3. In alto a sinistra, clicca su **File** -> **Import Notebook** e seleziona il nostro file `kaggle_project_notebook.ipynb`.

## 2. Configurare l'Ambiente Kaggle
Per far funzionare l'ambiente (e per velocizzare enormemente i calcoli) è necessario attivare le GPU e abilitare Internet per scaricare le dipendenze:
1. Apri il pannello laterale destro (cliccando sulla freccetta in basso a destra se è chiuso).
2. Sotto la sezione **Session options**:
   - **Accelerator**: Seleziona **GPU T4 x2**. (Il nostro codice usa `nn.DataParallel` per sfruttare entrambe le schede simultaneamente).
   - **Internet**: Attiva l'interruttore (assicurati che sia **On**). Questo è fondamentale per permettere al notebook di fare i `pip install` e per scaricare i pesi del modello pre-addestrato di DiffPIR.

## 3. Importare il Dataset
Abbiamo preparato un dataset pre-splittato di 4000 immagini perfette a 256x256 pixel (3200 Train, 400 Val, 400 Test).
1. Sempre nel pannello di destra, in alto, clicca su **+ Add Input**.
2. Nella finestra che si apre, seleziona la voce **Upload** (icona a forma di nuvola/freccia in alto a destra).
3. Inserisci un nome per il dataset (es. `lsun-church-256x256`).
4. Trascina e rilascia il file `lsun_dataset_256.zip` e clicca su **Create**.
5. Attendi che Kaggle completi l'upload e scompatti automaticamente l'archivio.

## 4. Collegare il Dataset al Codice
A volte Kaggle altera i nomi delle cartelle durante l'upload. Per assicurarti che il DataLoader trovi i file, usa questo trucco per rintracciare il percorso assoluto esatto:

Crea una nuova cella temporanea all'inizio del notebook ed esegui questo codice:
```python
import glob
import os

trovati = glob.glob('/kaggle/input/**/*.png', recursive=True)
if trovati:
    cartella_corretta = os.path.dirname(os.path.dirname(trovati[0]))
    print(f'Sostituisci in KAGGLE_DATASET_PATH questa stringa:\\n"{cartella_corretta}"')
```

Vai alla Cella 3 del Notebook (`3. Degradation Operator and Dataset`) e aggiorna la variabile alla fine del blocco:
```python
KAGGLE_DATASET_PATH = "/kaggle/input/datasets/tuo-username/lsun-church-256x256"
```

## 5. Avviare l'Esecuzione
1. Seleziona dal menu in alto: **Run** -> **Run All**.
2. Il notebook eseguirà le installazioni, imposterà le classi e preparerà il dataset.
3. Se l'addestramento della UNet è commentato (per evitare avvii accidentali), vai all'ultima cella, rimuovi i tripli apici `'''` e avviala manualmente per far partire le epoche!

## 6. Salvare e Ricaricare i Pesi della UNet (Evitare il Retrain)
Durante l'addestramento, il notebook salva in automatico il miglior modello nel file `unet_best.pth`.
Per evitare che gli altri membri del gruppo debbano ri-addestrare la rete da zero:
1. Al termine dell'addestramento su Kaggle, apri il pannello laterale destro.
2. Vai nella sezione **Output** (`/kaggle/working`).
3. Troverai il file `unet_best.pth`. Clicca sui tre puntini di fianco al file e fai **Download**.
4. Inserisci il file nella cartella del progetto su GitHub (il file pesa pochi MB).
5. Quando il tuo collega aprirà il Notebook, gli basterà caricare il file `unet_best.pth` su Kaggle (nella cartella di working) e il codice `unet_model.load_state_dict(torch.load('unet_best.pth'))` ricaricherà all'istante la rete già allenata!

