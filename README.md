# Computational Imaging - Group I Project

Questo repository contiene l'implementazione del progetto per il corso di Computational Imaging.
Il task è la **Super-Risoluzione (SR)** partendo da immagini degradate (downsampling 2x/4x e rumore Gaussiano 0.005/0.01).

## Metodi Implementati
1. **Metodo Variazionale**: Total Variation (TV)
2. **Metodo End-to-End**: UNet
3. **Metodo Generativo**: DiffPIR

## Struttura del Progetto
- `data/`: Contiene il dataset LSUN Church e gli split (cartella ignorata da Git).
- `src/`: Contiene il codice sorgente.
  - `dataset.py`: Gestione dati e split.
  - `degradations.py`: Funzioni (Forward model) per downsampling e rumore.
  - `metrics.py`: Calcolo PSNR e SSIM.
  - `methods/`: Implementazioni dei 3 algoritmi.
  - `train_unet.py`: Script di addestramento per la UNet.
- `notebooks/`: Notebook Jupyter per l'esplorazione dei dati e la visualizzazione dei risultati.
- `weights/`: Modelli pre-addestrati e pesi salvati.

## Setup
```bash
pip install -r requirements.txt
```
