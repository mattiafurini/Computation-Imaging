# 📈 03: Evaluation Metrics and Deliverables

## 1. Metriche di Valutazione e Confronto

Durante la presentazione orale e nel report tecnico, gli studenti devono fornire un confronto diretto, rigoroso e analitico di tutti i metodi testati in base alle seguenti metriche.

### 1.1 Metriche Quantitative

- **PSNR (Peak Signal-to-Noise Ratio):** Metrica per quantificare la fedeltà a livello di pixel della ricostruzione.
- **SSIM (Structural Similarity Index Measure):** Metrica per valutare la percezione della degradazione della struttura e dei dettagli visivi.
- Entrambe le metriche devono essere calcolate per ogni metodo rispetto a tutte le configurazioni sperimentali richieste (SR factor 2 e 4, Noise 0.005 e 0.01).

### 1.2 Metriche Visive (Qualitative)

- Estrazione ed esposizione di **immagini ricostruite**.
- **Confronti diretti (Comparisons):** Devono essere mostrati affiancamenti visivi tra ground truth (immagine originale ad alta qualità), l'osservazione degradata, e l'output ricostruito da ciascun metodo.

### 1.3 Plot di Confronto (Comparison Plot)

- È obbligatorio includere un **comparison plot** aggregato. Questo grafico (o serie di grafici) deve visualizzare le performance relative (PSNR e SSIM) *tra tutti i metodi* contemporaneamente, evidenziando il comportamento al variare della severità della degradazione.

## 2. Deliverable da Consegnare

L'output finale del progetto si articola su tre componenti fondamentali, da produrre in modo professionale e completo.

### 2.1 Repository del Codice (GitHub)

- **Codice:** Completamente funzionante, strutturato in modo logico e mantenibile.
- **Documentazione:** Il codice deve essere ampiamente commentato (docstrings e commenti in-line).
- **Distribuzione:** Preferibilmente effettuato il push su una repository **GitHub**. Il link della repository dovrà essere fornito ai docenti durante la presentazione orale.

### 2.2 Report Tecnico

- Un documento professionale in formato testuale (es. PDF) che includa:
  - Descrizione delle Metodologie implementate e delle loro peculiarità teoriche.
  - Definizione del Setup sperimentale (preprocessing rigorosamente giustificato).
  - Discussione (inclusa la ricerca euristica dei parametri).
  - Risultati quantitativi e qualitativi.

### 2.3 Presentazione Orale

- **Formato:** Presentazione strutturata utilizzando PowerPoint o Beamer (LaTeX).
- **Contenuto:** Deve riportare tutti i deliverable sopra descritti e fungere da supporto visivo alla discussione critica dei metodi come indicato nella project description originaria.
