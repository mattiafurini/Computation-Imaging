# 🛠️ 07: Guida all'Implementazione — Metodi e Metriche

*Ultimo aggiornamento: 5 Maggio 2026*

Questo documento descrive i file implementati nella Fase 2 e 3 del progetto, il loro funzionamento interno e come utilizzarli. È pensato per chiunque lavori al progetto e debba capire rapidamente cosa fa ogni componente.

---

## Panoramica dei File

```
src/
├── metrics.py                 # PSNR e SSIM (valutazione quantitativa)
├── methods/
│   ├── tv.py                  # Metodo 1: Total Variation (variazionale)
│   ├── unet.py                # Metodo 2: UNet (end-to-end, architettura)
│   └── diffpir.py             # Metodo 3: DiffPIR (generativo)
└── train_unet.py              # Script di training per la UNet
```

Tutti i metodi condividono la stessa interfaccia di base:
- **Input:** immagine degradata (LR, con rumore) come tensore PyTorch, PIL Image, o numpy array
- **Output:** immagine ricostruita `(3, 256, 256)` in range `[0, 1]`

---

## 1. `src/metrics.py` — Metriche di Valutazione

### Cosa fa
Calcola le due metriche richieste dal progetto per confrontare i metodi:
- **PSNR** (Peak Signal-to-Noise Ratio): misura la qualità in dB, più alto = meglio
- **SSIM** (Structural Similarity Index): misura la similarità strutturale, va da 0 a 1 (1 = identiche)

### Funzioni principali

| Funzione | Descrizione |
|---|---|
| `compute_psnr(pred, target)` | Calcola PSNR tra due immagini. Restituisce un `float` in dB. |
| `compute_ssim(pred, target)` | Calcola SSIM tra due immagini. Restituisce un `float` in [0, 1]. |
| `evaluate_batch(pred_dir, target_dir)` | Calcola PSNR/SSIM su un'intera directory di immagini. Restituisce un dizionario con medie, deviazioni standard e valori per-immagine. |

### Esempio d'uso

```python
from src.metrics import compute_psnr, compute_ssim, evaluate_batch

# Singola immagine
psnr = compute_psnr(immagine_ricostruita, ground_truth)
ssim = compute_ssim(immagine_ricostruita, ground_truth)

# Batch su directory
results = evaluate_batch("data/output_tv/", "data/test/", max_images=50)
print(f"PSNR medio: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
print(f"SSIM medio: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
```

### Dettagli tecnici
- Internamente usa `skimage.metrics` (libreria `scikit-image` già nei requirements)
- Accetta qualsiasi formato di input (tensore, numpy, PIL) e lo converte automaticamente
- Le immagini devono essere in range [0, 1]; se sono in [0, 255] vengono normalizzate automaticamente

---

## 2. `src/methods/tv.py` — Total Variation (Metodo Variazionale)

### Cosa fa
Risolve il problema inverso di super-resolution con un approccio **classico/matematico** (nessuna rete neurale). Formula il problema come un'ottimizzazione:

```
x* = argmin_x  ½ · ||H·x - y||²  +  λ · TV(x)
```

Dove:
- `½ · ||H·x - y||²` è il **data-fidelity term** → l'immagine ricostruita, una volta degradata, deve assomigliare all'osservazione `y`
- `λ · TV(x)` è il **prior di Total Variation** → penalizza le variazioni brusche (rumore) preservando i bordi netti
- `H` è l'operatore di degradazione (blur gaussiano + downsampling), lo stesso usato in `degradations.py`

### Come funziona internamente

1. **Inizializzazione:** parte dall'upsampling bicubico dell'immagine degradata (come baseline)
2. **Iterazione FISTA** (Fast Iterative Shrinkage-Thresholding Algorithm):
   - Calcola il gradiente del data-fidelity: `H^T(H·x - y)` (dove `H^T` è l'operatore aggiunto)
   - Calcola il gradiente della TV: `-div(∇x / |∇x|)` (divergenza del campo gradiente normalizzato)
   - Aggiorna `x` con un passo di gradient descent
   - Applica il momentum FISTA per accelerare la convergenza
3. **Proiezione:** dopo ogni step, forza l'output in `[0, 1]`

### Parametri euristici da tuning

| Parametro | Default | Significato |
|---|---|---|
| `lambda_tv` | 0.01 | Peso della regolarizzazione TV. **Più alto = più smooth, meno rumore ma meno dettagli.** |
| `n_iter` | 500 | Numero di iterazioni. Più iterazioni = risultato migliore ma più lento. |
| `step_size` | 0.005 | Passo del gradient descent. Troppo grande → diverge, troppo piccolo → convergenza lenta. |

### Uso

```python
from src.methods.tv import TVSuperResolution, solve_tv
from PIL import Image

# Modo semplice (funzione convenience)
result = solve_tv("data/test_sr2x_noise0005/church_0001.png", sr_factor=2, noise_std=0.005)

# Modo completo (classe)
solver = TVSuperResolution(sr_factor=2, noise_std=0.005, lambda_tv=0.01, n_iter=500)
degraded = Image.open("data/test_sr2x_noise0005/church_0001.png")
result = solver.solve(degraded)  # → Tensor (3, 256, 256)
```

### Test rapido

```bash
cd /home/mattia/Documents/Computational\ Imaging
source venv/bin/activate
python src/methods/tv.py
```

**Risultato ottenuto:** PSNR 31.52 dB, SSIM 0.9282 su `church_0001.png` (SR 2×, noise 0.005). Funziona su CPU senza problemi (~30 secondi per immagine).

---

## 3. `src/methods/unet.py` — UNet (Metodo End-to-End)

### Cosa fa
È una **rete neurale convoluzionale** che impara direttamente il mapping:

```
bicubic_upsample(degraded) → ground_truth
```

A differenza della TV che usa la matematica dell'ottimizzazione, la UNet impara dai dati. Richiede una fase di **training** (vedere sezione successiva).

### Architettura

```
Input (3, 256, 256)
  ↓ Encoder
  [DoubleConv 64] → [Pool] → [DoubleConv 128] → [Pool] → [DoubleConv 256] → [Pool] → [DoubleConv 512] → [Pool]
  ↓ Bottleneck
  [DoubleConv 1024]
  ↓ Decoder (con skip connections dall'encoder)
  [UpConv+Cat 512] → [UpConv+Cat 256] → [UpConv+Cat 128] → [UpConv+Cat 64]
  ↓
Output (3, 256, 256)
```

- **Skip connections:** ogni livello dell'encoder è collegato al livello corrispondente del decoder (concatenazione). Questo preserva i dettagli a diverse scale.
- **DoubleConv:** ogni blocco fa Conv3×3 → BatchNorm → LeakyReLU (×2)
- **Output:** sigmoid finale per forzare range [0, 1]
- **Parametri totali:** 31,037,763 (~31M)

### Uso (dopo il training)

```python
import torch
from src.methods.unet import UNet

model = UNet()
checkpoint = torch.load("weights/unet_sr2x_noise0005_best.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# L'input deve essere l'immagine degradata GIÀ upsampled a 256×256 con bicubic
input_img = ...  # Tensor (1, 3, 256, 256)
with torch.no_grad():
    output = model(input_img)  # Tensor (1, 3, 256, 256)
```

---

## 4. `src/train_unet.py` — Training della UNet

### Cosa fa
Script completo per addestrare la UNet. Gestisce tutto: caricamento dati, degradazione on-the-fly, training loop, validazione, salvataggio modello.

### Come funziona il training

1. **Dataset (`SRDataset`):** per ogni immagine ground truth 256×256:
   - Applica la degradazione (blur + downsampling + noise) → immagine LR
   - Fa upsampling bicubico della LR → torna a 256×256 (questa è l'**input** della rete)
   - La ground truth originale è il **target**
   - La rete impara: "dato questo input sfocato, restituisci l'originale nitido"

2. **Loss:** L1 (Mean Absolute Error) — preferita a MSE perché preserva meglio i dettagli fini
3. **Optimizer:** Adam con learning rate scheduling (ReduceLROnPlateau)
4. **Validazione:** ogni epoca calcola PSNR e SSIM sul validation set
5. **Salvataggio:** salva il modello migliore (best PSNR) in `weights/`

### Come lanciare il training

```bash
# Attivare il virtual environment
source venv/bin/activate

# Training per una configurazione specifica
python src/train_unet.py --sr_factor 2 --noise_std 0.005 --epochs 50 --batch_size 8

# Tutte le 4 configurazioni da trainare:
python src/train_unet.py --sr_factor 2 --noise_std 0.005 --epochs 50
python src/train_unet.py --sr_factor 2 --noise_std 0.01  --epochs 50
python src/train_unet.py --sr_factor 4 --noise_std 0.005 --epochs 50
python src/train_unet.py --sr_factor 4 --noise_std 0.01  --epochs 50
```

### Output del training

Per ogni configurazione, vengono salvati in `weights/`:
- `unet_sr2x_noise0005_best.pth` — checkpoint del modello migliore
- `unet_sr2x_noise0005_final.pth` — checkpoint dell'ultima epoca
- `unet_sr2x_noise0005_history.json` — storico loss/PSNR/SSIM per ogni epoca

### ⚠️ Requisiti GPU
Il training su CPU è troppo lento (ore per una singola epoca). **Serve una GPU** (Google Colab, server universitario, ecc.). Lo script rileva automaticamente CUDA e usa la GPU se disponibile.

---

## 5. `src/methods/diffpir.py` — DiffPIR (Metodo Generativo)

### Cosa fa
Usa un **modello di diffusione pre-addestrato** come prior generativo per la super-resolution. È l'approccio più moderno dei tre.

### Come funziona (algoritmo DiffPIR)

L'idea è partire da rumore puro e iterativamente "denoizzare" l'immagine, guidando il processo verso una ricostruzione coerente con l'osservazione degradata `y`:

1. **Inizializzazione:** `x_T ← rumore gaussiano puro` (256×256)
2. **Per ogni timestep t = T, T-1, ..., 1:**
   - **Denoising step:** il modello di diffusione rimuove un po' di rumore → stima `x_0`
   - **Data-fidelity step:** corregge `x_0` perché, una volta degradata, assomigli a `y`:
     ```
     x_0_corretto = x_0 - ρ_t · H^T(H·x_0 - y)
     ```
   - **Re-noise:** aggiunge rumore calibrato al livello `t-1` e continua
3. **Output finale:** `x_0` pulito e coerente con `y`

### Struttura del codice

| Classe/Funzione | Descrizione |
|---|---|
| `DiffPIR` | Classe principale. Gestisce la diffusion schedule, il sampling, e i data-fidelity step. |
| `SimpleDenoiser` | Rete denoiser integrata (fallback). Funziona per testare la pipeline ma non dà buoni risultati — serve un modello **pre-addestrato**. |
| `restore_diffpir()` | Funzione convenience per uso rapido. |

### Parametri euristici da tuning

| Parametro | Default | Significato |
|---|---|---|
| `sampling_steps` | 100 | Numero di step di diffusione. Più step = qualità migliore ma più lento. |
| `rho` | 1.0 | Peso complessivo del data-fidelity step. |
| `lambda_reg` | 0.5 | Forza della regolarizzazione. Bilancia fedeltà all'osservazione vs. qualità generativa. |

### Uso

```python
from src.methods.diffpir import DiffPIR, restore_diffpir

# Modo semplice
result = restore_diffpir(
    "data/test_sr2x_noise0005/church_0001.png",
    sr_factor=2, noise_std=0.005,
    model_path="weights/diffusion_model.pt",  # modello pre-addestrato
    sampling_steps=100,
    device='cuda'
)

# Modo completo
solver = DiffPIR(sr_factor=2, noise_std=0.005, model_path="weights/diffusion_model.pt")
result = solver.restore(degraded_image)  # → Tensor (3, 256, 256)
```

### ⚠️ Requisiti

1. **GPU obbligatoria** — l'inferenza su CPU è estremamente lenta (~30-60 sec per immagine con GPU, ore su CPU)
2. **Modello pre-addestrato** — il `SimpleDenoiser` integrato è solo un placeholder per testare la pipeline. Per risultati reali serve scaricare un modello di diffusione pre-addestrato (es. guided-diffusion di OpenAI, ~2GB) e passare il path via `model_path`

---

## Come Integrare il Tutto — Pipeline di Valutazione

Una volta che tutti i metodi producono risultati, la valutazione si fa così:

```python
from src.metrics import evaluate_batch

# Ogni metodo salva i suoi output in una directory separata
methods = {
    'TV':      'data/output_tv/',
    'UNet':    'data/output_unet/',
    'DiffPIR': 'data/output_diffpir/',
}

for name, pred_dir in methods.items():
    results = evaluate_batch(pred_dir, 'data/test/')
    print(f"{name}: PSNR={results['psnr_mean']:.2f} dB, SSIM={results['ssim_mean']:.4f}")
```

Il notebook `notebooks/02_results_comparison.ipynb` è già predisposto per questa valutazione comparativa.

---

## Stato Attuale e Prossimi Passi

| Componente | Stato | Note |
|---|---|---|
| `metrics.py` | ✅ Pronto | Testato e funzionante |
| `tv.py` | ✅ Pronto | Funziona su CPU, risultati buoni (31.52 dB) |
| `unet.py` | ✅ Architettura pronta | Serve training su GPU |
| `train_unet.py` | ✅ Pronto | Da lanciare su macchina con GPU |
| `diffpir.py` | ✅ Pipeline pronta | Serve modello pre-addestrato + GPU |

### Da fare
1. **Trainare la UNet** sulle 4 configurazioni (serve GPU)
2. **Scaricare il modello di diffusione** pre-addestrato per DiffPIR
3. **Eseguire tutti i metodi** sul test set completo (400 immagini × 4 configurazioni)
4. **Popolare il notebook** `02_results_comparison.ipynb` con i risultati
5. **Tuning euristico** dei parametri (da documentare per la presentazione)
