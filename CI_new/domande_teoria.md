---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
color: #334155
style: |
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');
  
  section {
    font-family: 'Plus Jakarta Sans', sans-serif;
    padding: 60px;
    background: #ffffff;
  }
  
  section.lead {
    background: linear-gradient(135deg, #f0fdf4 0%, #eff6ff 100%);
    color: #1e293b;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.lead h1 { 
    color: #1e3a8a; 
    font-size: 2.6em;
    font-weight: 800; 
    letter-spacing: -1px;
    border-bottom: none;
    margin-bottom: 10px;
  }
  section.lead h2 { 
    color: #475569; 
    font-size: 1.3em;
    font-weight: 500;
    border: none;
  }
  
  h2 { 
    color: #1e3a8a; 
    font-size: 1.8em;
    font-weight: 700;
    letter-spacing: -0.5px;
    border-bottom: 3px solid #3b82f6; 
    padding-bottom: 10px;
    margin-bottom: 25px;
  }
  
  ul { margin-top: 10px; }
  li { margin-bottom: 12px; line-height: 1.5; color: #334155; font-size: 1em; }
  li strong { color: #0f172a; font-weight: 700; }
  
  code {
    background-color: #f1f5f9;
    color: #0f172a;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Consolas', monospace;
    font-size: 0.85em;
  }
---

<!-- _class: lead -->

# Domande Teoriche d'Esame
## Q&A per la preparazione all'orale
**Gruppo I - Super-Resolution**

---

## 1. Perché la Super-Resolution è un problema "mal posto"?

**Domanda:** *Secondo Hadamard, cosa definisce un problema ben posto e perché il vostro non lo è?*

* **Definizione di Hadamard:** Un problema è ben posto se la soluzione *esiste*, è *unica* e dipende con *continuità* dai dati.
* **Il nostro problema ($y = Hx + e$):** È un sistema **sottodeterminato** ($M < N$). I pixel dell'immagine degradata $y$ sono inferiori ai pixel dell'immagine ad alta risoluzione $x$ originale che vogliamo trovare.
* **Risultato:** La soluzione **non è unica** (infinite combinazioni di pixel HR possono produrre la stessa immagine LR). Serve per forza introdurre informazioni "a priori" (Regolarizzazione o Prior neurali).

---

## 2. Total Variation vs Tichonov

**Domanda:** *Perché nel metodo variazionale avete usato la Total Variation (TV) e non la Norma 2 (Tichonov)?*

* **Tichonov (Norma 2):** Impone una omogeneità assoluta all'immagine. Questo cancella efficacemente il rumore, ma ha il difetto inesorabile di **sfocare pesantemente i contorni** degli oggetti.
* **Total Variation (TV):** Applica la Norma 1 al gradiente dell'immagine. Matematicamente assume che l'immagine sia "costante a tratti". 
* **Vantaggio della TV:** Permette di **preservare i bordi netti** e i salti drastici di colore. 
* **Svantaggio della TV:** Può indurre un effetto di artificialità a blocchi chiamato *staircasing* (effetto "acquerello").

---

## 3. Il meccanismo Plug-and-Play (PnP)

**Domanda:** *Come funziona l'algoritmo Plug-and-Play all'interno di DiffPIR?*

* **Concept:** I metodi PnP scompongono un'ottimizzazione difficile in due sotto-problemi più facili e iterativi.
* In DiffPIR, ogni step temporale della diffusione viene diviso esplicitamente in due fasi:
  1. **Prior Step (Denoising):** La rete di diffusione pre-addestrata agisce come un denoiser puro per generare dettagli realistici partendo dal rumore corrente.
  2. **Data Fidelity Step:** Un passo di ottimizzazione classico in cui si sfrutta il nostro operatore di degradazione noto $H$ per "forzare" la coerenza tra l'immagine che si sta allucinando e l'osservazione a bassa risoluzione $y$.

---

## 4. Rischi dei modelli Generativi (Diffusione)

**Domanda:** *Qual è il rischio più grande nell'usare modelli Generativi (come DiffPIR) rispetto a una normale UNet?*

* **UNet:** Tende a minimizzare l'errore calcolando la "media statistica" (Regression to the mean), risultando sfocata e conservativa ma sempre sicura.
* **Modelli Generativi:** Campionano la distribuzione in modo da ricreare una realtà estremamente "plausibile", introducendo dettagli ad alta frequenza che a bassa risoluzione non esistevano.
* **Il Rischio (Allucinazioni):** Se il modello generativo è troppo libero, può produrre artefatti o dettagli visivi molto realistici e convincenti, ma fisicamente **completamente falsi o inventati** rispetto al vero Ground Truth (un rischio inaccettabile, ad esempio, in campo medico).

---

## 5. Il paradosso di PSNR vs SSIM

**Domanda:** *La UNet ha il PSNR più alto di tutti. Perché allora il risultato di DiffPIR vi sembra visivamente migliore?*

* **Il PSNR (Peak Signal-to-Noise Ratio):** Si basa sull'errore quadrantico medio (MSE) calcolato brutalmente **pixel-per-pixel**. Tende a premiare immagini un po' sfocate (che stanno in media e non sbagliano troppo) rispetto a immagini con dettagli nitidi posizionati in un pixel leggermente sbagliato.
* **La percezione visiva:** Il PSNR ha una **bassa correlazione con l'occhio umano**. La UNet massimizza il PSNR ma il risultato è liscio e artificiale.
* **DiffPIR e lo SSIM:** DiffPIR ha metriche PSNR inferiori a causa di piccoli artefatti "allucinati", ma genera micro-texture percepite come molto più realistiche e definite, confermando il limite concettuale dell'affidarsi solo all'MSE.

---

## 6. L'Operatore di sfocatura nel dominio di Fourier

**Domanda:** *Se passiamo nel dominio delle frequenze (Fourier), cosa fa esattamente l'operatore di sfocatura Gaussiana?*

* **Azione del Blur:** Un filtro Gaussiano spaziale agisce esattamente come un **Filtro Passa-Basso (Low-pass filter)** nel dominio di Fourier.
* **L'effetto fisico:** Mantiene le basse frequenze (i colori generali e le masse dell'immagine) ma **taglia o attenua le alte frequenze**.
* **L'effetto visivo:** Siccome, secondo Fourier, le alte frequenze corrispondono ai bordi netti e ai micro-dettagli degli oggetti, tagliarle equivale a ottenere un'immagine sfocata in cui i contorni sono persi. L'obiettivo della Super-Resolution è proprio "inventarsi" le frequenze mancanti.

---

## 7. Velocizzare i calcoli con Fourier (FFT)

**Domanda:** *Perché nel Data Consistency Step (o nei metodi classici) invertire l'operatore $H$ spesso si fa sfruttando il dominio di Fourier?*

* **Il Problema:** Computare una convoluzione globale nel dominio spaziale su immagini molto grandi ha una complessità temporale elevatissima.
* **La Soluzione (Teorema di Convoluzione):** Il teorema matematico afferma che una convoluzione tra due segnali nello spazio equivale a un **semplice prodotto algebrico punto-a-punto** delle loro trasformate nel dominio di Fourier.
* **Vantaggio pratico:** Sfruttando la FFT (Fast Fourier Transform), operazioni pesantissime diventano quasi istantanee (complessità abbattuta a $O(N \log_2 N)$), rendendo la risoluzione iterativa dei problemi inversi enormemente più veloce.

---

## 8. Gestione del colore (RGB) nella Total Variation

**Domanda:** *La Total Variation è formulata originariamente per segnali o immagini a singolo canale (grayscale). Come avete gestito le immagini a 3 canali (RGB) del vostro dataset?*

* **La problematica:** Applicare la TV classica in vettoriale accoppiando i canali è matematicamente complesso. Se si usano operatori "standard" pensati per il medicale (come il Gradient di IPPy), i canali aggiuntivi vengono addirittura ignorati, corrompendo l'immagine.
* **La nostra soluzione:** Abbiamo implementato una **Channel-wise Anisotropic TV** sfruttando lo slicing tensoriale di PyTorch. Calcoliamo la divergenza in parallelo per ogni canale colore (R, G, B) in modo del tutto **indipendente**.
* **Pro e Contro:** È un approccio estremamente rapido da calcolare che ci ha permesso di evitare bug e bypassare le limitazioni degli operatori base, ma siccome penalizza i canali in modo slegato, può occasionalmente generare microscopici sfasamenti (color bleeding) in zone di forte gradiente netto.
