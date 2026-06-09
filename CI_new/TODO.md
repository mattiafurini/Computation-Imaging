# Progetto Computational Imaging - Group I (Super-Resolution)

## 🎯 Stato del Progetto

### ✅ Completato
- Analisi teorica dei requisiti (rispetto delle slide del corso e delle richieste del bando).
- Creazione della pipeline di degradazione uniforme per tutti i metodi (`DegradationOperator` con blur Gaussiano e rumore additivo).
- Implementazione del **Metodo End-to-End**: Architettura UNet completa di loop di addestramento su PyTorch.
- Implementazione del **Metodo Variazionale**: Ottimizzazione con Total Variation (TV) tramite FISTA. (Risolto bug delle matrici e mantenuto l'algoritmo indipendente per canale per supportare l'RGB nativo!).
- Implementazione del **Metodo Generativo**: DiffPIR con modello di diffusione pre-addestrato (`google/ddpm-ema-church-256`).
- Setup di **Kaggle** con supporto multi-GPU (`DataParallel` su doppia T4).
- Creazione e formattazione del dataset perfetto: subset di 4000 immagini da LSUN Church, già suddiviso in `train` (3200), `val` (400) e `test` (400), ridimensionato nativamente a 256x256.
- **Addestramento della UNet completato con successo:** Raggiunta l'Epoca 48 con PSNR 27.68 dB e SSIM 0.8802. Modello salvato (`unet_best.pth`).
- **Valutazione Visiva (Qualitativa) completata:** Eseguito il plot comparativo finale (`[GT] | [Degradata] | [UNet] | [TV] | [DiffPIR]`) per la configurazione base (SR=2, noise=0.005) su immagini di test.

### ⏳ Da Fare (Next Steps)
1. **Valutazione Quantitativa sul Test Set:**
   - [x] Eseguire un loop sulle immagini del Test Set per calcolare la media esatta di PSNR e SSIM per la UNet, TV e DiffPIR.

2. **Test sulle altre degradazioni richieste:**
   - [x] Valutare le prestazioni dei metodi cambiando i parametri di degradazione (Fattore SR=4, Livello di rumore=0.01) come richiesto dal bando per vedere quale metodo "crolla" prima.

3. **Discussione e Presentazione (Google Slides):**
   - [x] Inserire i Plot visivi appena creati.
   - [x] Inserire le tabelle numeriche con PSNR e SSIM medi.
   - [x] Discutere la scelta dell'adattamento RGB per i metodi variazionali.
   - [x] Discutere il trade-off teorico tra UNet (*over-smoothing*, regressione alla media, PSNR alto) e DiffPIR (*allucinazioni realistiche*, rumore da data-fidelity, PSNR potenzialmente più basso).
   - [x] Aggiungere la giustificazione dei parametri euristici scelti (es. $\lambda=0.05$ per il TV).

4. **Consegna Github:**
   - [x] Pulire il notebook.
   - [x] Pushare il `TODO.md` e `README.md` aggiornati assieme al codice per il proprio collega e per il prof.
