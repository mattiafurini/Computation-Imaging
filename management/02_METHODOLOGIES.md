# 🔬 02: Methodological Requirements

Gli studenti sono tenuti a testare e confrontare metodologie appartenenti a diverse famiglie sullo stesso set di dati. 

## 1. Regola per Gruppi da 2 Studenti
Se il gruppo è costituito da soli 2 studenti, è consentito scegliere di implementare **tre dei quattro metodi totali**, escludendone uno a scelta tra gli ultimi tre (End-to-end, Generative, o Hybrid). L'implementazione del metodo Variazionale è **obbligatoria**. (Nota: è comunque possibile implementare tutti e quattro i metodi in via facoltativa).

## 2. Dettaglio dei Metodi

### 2.1 Metodo Variazionale (Obbligatorio)
- **Approccio:** Regolarizzazione con Total Variation (TV).
- **Descrizione:** Un approccio classico ai problemi inversi che sfrutta la priorità di variazione totale per preservare i contorni netti minimizzando al contempo il rumore.

### 2.2 Metodo End-to-end (A scelta)
- **Approccio:** Deep Learning supervisionato diretto.
- **Opzioni Architetturali:** Selezionare un metodo appropriato all'interno della seguente lista:
  - **UNet:** Una classica architettura CNN encoder-decoder con skip connections.
  - **ViT (Vision Transformer):** Architettura basata sull'attenzione per catturare dipendenze a lungo raggio.
  - **NAF-Net (Nonlinear Activation Free Network):** Un'architettura SOTA ottimizzata specificatamente per il restauro di immagini.
- **Richiesta Specifica:** Scegliere l'architettura specifica e definire la pipeline di training in rigorosa relazione alle proprietà del task di Super-Risoluzione.

### 2.3 Metodo Generativo (A scelta)
- **Approccio:** Modelli di Diffusione (Diffusion Models).
- **Architettura Specificata:** **DiffPIR** (Diffusion models for Plug-and-Play Image Restoration).
- **Descrizione:** Utilizzare modelli di diffusione pre-addestrati come prior per la risoluzione del problema inverso.

### 2.4 Metodo Ibrido (A scelta)
- **Approccio:** Plug and Play (PnP).
- **Framework Specificato:** Half Quadratic Splitting (HQS).
- **Descrizione:** Inserimento di denoiser off-the-shelf all'interno dell'algoritmo iterativo HQS per risolvere l'ottimizzazione legata al data-fidelity term e al prior term in modo alternato.

## 3. Gestione degli Iperparametri e Ricerca Euristica
Ciascuno dei metodi precedenti possiede un set specifico di parametri intrinseci che deve essere attentamente configurato (ad esempio: il numero di step di diffusione, come in DPS, o il parametro di regolarizzazione per la TV). 
- **Compito:** Gli studenti devono identificare i valori ottimali di tali parametri in modo **euristico**.
- **Deliverable Associato:** La scelta finale, assieme a una descrizione formale del processo metodologico seguito per fissarli, **deve essere discussa** in modo approfondito nella presentazione orale e nel report.
