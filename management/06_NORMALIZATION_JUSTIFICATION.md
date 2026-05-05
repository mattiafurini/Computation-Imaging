# 📐 06: Giustificazione Matematica Normalizzazione Dati (Task 1.5)

Nel contesto dei problemi inversi e della **Super-Risoluzione (SR)** per il Computational Imaging, la scelta della normalizzazione dei dati in ingresso non è arbitraria ma risponde a precise necessità matematiche e numeriche.

Per questo progetto, abbiamo implementato una normalizzazione canonica nel range continuo **$[0, 1]$** (implementata in `src/normalization.py`). Di seguito le giustificazioni matematiche della scelta rispetto ad altri approcci (come la *Z-score normalization* standard).

## 1. Mappatura Fisica e Dominio del Segnale
Le immagini digitali grezze sono acquisite con intensità discrete $x_{ij} \in \{0, 1, \dots, 255\}$. Nei modelli continui (in particolare per l'approccio **Variazionale**), l'intensità luminosa è modellata come una funzione continua $u: \Omega \rightarrow \mathbb{R}$.
Scalare il segnale dividendo per 255 applica una trasformazione lineare $x' = \frac{x}{255}$ che mappa il dominio discreto in $[0, 1]$. Questo preserva la **positività fisica** della luce: un pixel nero (assenza di segnale) rimane $0$. Una normalizzazione $N(0, 1)$ introdurrebbe radianze negative, prive di significato fisico e deleterie per i prior come la *Total Variation* (TV).

## 2. Calibrazione del Rumore Additivo Gaussiano
Il nostro modello di forward degradation (Task 1.7-1.8) è definito come:
$$ y = Hx + n $$
dove $n$ è il rumore gaussiano additivo, $n \sim \mathcal{N}(0, \sigma^2)$. 
I requisiti del progetto impongono l'uso di $\sigma \in \{0.005, 0.01\}$. Questi livelli di deviazione standard sono intrinsecamente calibrati per segnali di ampiezza unitaria (range $[0,1]$). 
Se mantenessimo i pixel nel range $[0, 255]$, applicare un $\sigma=0.01$ equivarrebbe a un rumore impercettibile, rendendo l'esperimento di degradazione inefficace. Mantenere l'immagine in $[0,1]$ permette al termine di data-fidelity $\|y - Hx\|_2^2$ di avere un peso proporzionato.

## 3. Stabilità dell'Ottimizzazione (Gradienti)
Sia nei metodi end-to-end (es. UNet) tramite backpropagation, sia in metodi iterativi HQS/Variazionali, avere il dato in range $[0, 1]$ assicura che il modulo dei gradienti rimanga limitato, prevenendo fenomeni di *gradient explosion*. L'inizializzazione dei pesi standard nelle reti neurali e lo step-size $\gamma$ nei gradient descent assumono implicitamente input a varianza ridotta e limitata.

## 4. Perché NON utilizzare la Z-Score Normalization (Mean/Std)?
La normalizzazione per media e varianza (es. normalizzazione ImageNet $x_{norm} = \frac{x - \mu}{\sigma}$) è ideale per task di *classificazione*, dove interessa rimuovere lo shift di illuminazione globale per estrarre feature invarianti. 
Nei task di *restoration/Super-Risoluzione*, l'obiettivo è ripristinare **l'esatto valore di radianza originale**. Applicare una Z-score altererebbe la media globale dell'immagine: due patch con la stessa struttura ma luminosità diverse verrebbero mappate sugli stessi valori, perdendo un'informazione fondamentale che la rete (o il solver) dovrebbe invece imparare a ricostruire fedelmente.
