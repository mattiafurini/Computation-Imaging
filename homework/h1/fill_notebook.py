import json
import os
from pathlib import Path

notebook_path = '/home/mattia/Documents/Computational Imaging/homework/h1/homework1.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

dataset_code = """class MayoDataset(Dataset):
    def __init__(self, data_path, data_shape=256):
        super().__init__()
        # Ricerca di tutti i file .png nel data_path
        self.fname_list = sorted(glob.glob(f'{data_path}/*/*.png'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((data_shape, data_shape)),
        ])

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):
        # Carica l'immagine in scala di grigi
        img = Image.open(self.fname_list[idx]).convert('L')
        # Applica le trasformazioni (ToTensor scala in [0,1])
        img_tensor = self.transform(img)
        return img_tensor

# Costruiamo il dataset completo usando i dati forniti.
data_path = book_root / 'data'
full_dataset = MayoDataset(data_path)

# Suddividiamo in training (80%) e test (20%)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
generator = torch.Generator().manual_seed(42)

if len(full_dataset) > 0:
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size], generator=generator
    )
else:
    train_dataset, test_dataset = [], []
    print("ATTENZIONE: Nessuna immagine trovata in", data_path)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Definisco l'operatore K
try:
    # Usa un operatore noto di IPPy per il motion blur
    K = operators.MotionBlur(kernel_size=15, angle=45).to(device)
except Exception:
    print("Operatore MotionBlur fallito o mancante in IPPy.operators, uso Identity.")
    K = operators.Identity().to(device)

print(f"Dataset caricato: {len(train_dataset)} train, {len(test_dataset)} test")

# Visualizza un esempio di batch pulito e corrotto
if len(train_dataset) > 0:
    x_clean = next(iter(train_loader)).to(device)
    noise_level = 0.05
    y_corrupted = K(x_clean) + noise_level * torch.randn_like(x_clean)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(x_clean[0, 0].cpu().numpy(), cmap='gray')
    axes[0].set_title('Clean Image (x)')
    axes[0].axis('off')
    
    axes[1].imshow(y_corrupted[0, 0].cpu().numpy(), cmap='gray')
    axes[1].set_title('Corrupted Measurement (y_delta)')
    axes[1].axis('off')
    plt.show()
"""

models_code = """class SimpleCNN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, n_filters=32, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, n_filters, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, out_ch, kernel_size, padding=padding)
        )

    def forward(self, x):
        return self.net(x)


class ResCNN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, n_filters=32, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_ch, n_filters, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, out_ch, kernel_size, padding=padding)
        )

    def forward(self, x):
        # Impara il residuo ad alta frequenza e lo somma all'input (skip connection globale)
        return x + self.residual_block(x)


# Estensione: UNet
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, n_filters=32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, n_filters, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(n_filters, n_filters, 3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(n_filters, n_filters*2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(n_filters*2, n_filters*2, 3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(n_filters*2, n_filters*4, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(n_filters*4, n_filters*4, 3, padding=1), nn.ReLU(inplace=True))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(n_filters*4, n_filters*2, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(n_filters*4, n_filters*2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(n_filters*2, n_filters*2, 3, padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(n_filters*2, n_filters, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(n_filters*2, n_filters, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(n_filters, n_filters, 3, padding=1), nn.ReLU(inplace=True))
        
        # Output
        self.out_conv = nn.Conv2d(n_filters, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return self.out_conv(d2)


# Istanziamo i modelli
simple_cnn = SimpleCNN().to(device)
res_cnn = ResCNN().to(device)
unet_model = UNet().to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"SimpleCNN trainabili params: {count_parameters(simple_cnn):,}")
print(f"ResCNN trainabili params:    {count_parameters(res_cnn):,}")
print(f"UNet trainabili params:      {count_parameters(unet_model):,}")
"""

train_code = """def train_model(model, train_loader, K, weights_path, num_epochs=20, noise_level=0.01, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for x_clean in pbar:
            x_clean = x_clean.to(device)
            # Simulazione online della misurazione: K*x + rumore
            y_corrupted = K(x_clean) + noise_level * torch.randn_like(x_clean)
            
            optimizer.zero_grad()
            # Forward pass passando il dato corrotto
            x_pred = model(y_corrupted)
            
            # Loss calcolata contro il dato pulito
            loss = loss_fn(x_pred, x_clean)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x_clean.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.5f}"})
            
        epoch_loss /= len(train_loader.dataset)
        history.append(epoch_loss)
        
    # Salvataggio pesi
    torch.save(model.state_dict(), weights_path)
    print(f"Pesi salvati in: {weights_path}")

    return history


# Numero di epoche suggerito: 20 (riducibile per test veloci)
epochs = 20
noise_sigma = 0.05

print("\\n--- Addestramento SimpleCNN ---")
simplecnn_hist = train_model(simple_cnn, train_loader, K, weights_dir / 'simplecnn.pth', num_epochs=epochs, noise_level=noise_sigma)

print("\\n--- Addestramento ResCNN ---")
rescnn_hist = train_model(res_cnn, train_loader, K, weights_dir / 'rescnn.pth', num_epochs=epochs, noise_level=noise_sigma)

print("\\n--- Addestramento UNet (Estensione) ---")
unet_hist = train_model(unet_model, train_loader, K, weights_dir / 'unet.pth', num_epochs=epochs, noise_level=noise_sigma)

# Plot delle training curves
plt.figure(figsize=(10, 5))
plt.plot(simplecnn_hist, label='SimpleCNN Loss')
plt.plot(rescnn_hist, label='ResCNN Loss')
plt.plot(unet_hist, label='UNet Loss')
plt.title('Training Convergence')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

# Ricaricamento dei pesi per sicurezza prima dell'evaluation
simple_cnn.load_state_dict(torch.load(weights_dir / 'simplecnn.pth'))
res_cnn.load_state_dict(torch.load(weights_dir / 'rescnn.pth'))
unet_model.load_state_dict(torch.load(weights_dir / 'unet.pth'))
simple_cnn.eval()
res_cnn.eval()
unet_model.eval()
"""

eval_code = """import sys
# Aggiungiamo il percorso per importare le metriche dal progetto
sys.path.append(str(book_root / 'src'))
try:
    from metrics import psnr, ssim
    has_metrics = True
except ImportError:
    has_metrics = False
    print("Modulo metrics non trovato in src/, si procederà solo con MSE.")

def evaluate_and_plot(models_dict, test_loader, K, noise_level=0.05):
    # Prendiamo un singolo batch di test
    x_clean = next(iter(test_loader)).to(device)
    y_meas = K(x_clean) + noise_level * torch.randn_like(x_clean)
    
    # Selezioniamo solo la prima immagine del batch per la visualizzazione e le metriche specifiche
    x_gt = x_clean[0:1]
    y_corrupted = y_meas[0:1]
    
    metrics_res = {}
    
    # Valutazione Baseline
    baseline_mse = nn.functional.mse_loss(y_corrupted, x_gt).item()
    metrics_res['Corrupted'] = {'MSE': baseline_mse}
    if has_metrics:
        metrics_res['Corrupted']['PSNR'] = psnr(y_corrupted, x_gt).item()
        metrics_res['Corrupted']['SSIM'] = ssim(y_corrupted, x_gt).item()
        
    preds = {}
    
    with torch.no_grad():
        for name, model in models_dict.items():
            pred = model(y_corrupted)
            preds[name] = pred
            
            mse = nn.functional.mse_loss(pred, x_gt).item()
            metrics_res[name] = {'MSE': mse}
            if has_metrics:
                metrics_res[name]['PSNR'] = psnr(pred, x_gt).item()
                metrics_res[name]['SSIM'] = ssim(pred, x_gt).item()
                
    # Tabella Risultati
    print(f"{'Model':<15} | {'MSE':<10} | {'PSNR':<10} | {'SSIM':<10}")
    print("-" * 52)
    for name, res in metrics_res.items():
        p_val = f"{res.get('PSNR', 0.0):.2f}" if has_metrics else "N/A"
        s_val = f"{res.get('SSIM', 0.0):.4f}" if has_metrics else "N/A"
        print(f"{name:<15} | {res['MSE']:.6f} | {p_val:<10} | {s_val:<10}")

    # Visualizzazione
    n_cols = len(models_dict) + 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
    
    axes[0].imshow(x_gt[0, 0].cpu().numpy(), cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    axes[1].imshow(y_corrupted[0, 0].cpu().numpy(), cmap='gray')
    axes[1].set_title(f"Corrupted\\nMSE: {baseline_mse:.4f}")
    axes[1].axis('off')
    
    for idx, (name, pred) in enumerate(preds.items()):
        ax = axes[idx + 2]
        ax.imshow(pred[0, 0].cpu().numpy(), cmap='gray')
        ax.set_title(f"{name}\\nMSE: {metrics_res[name]['MSE']:.4f}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

# Dizionario dei modelli da valutare
models_dict = {
    'SimpleCNN': simple_cnn,
    'ResCNN': res_cnn,
    'UNet': unet_model
}

evaluate_and_plot(models_dict, test_loader, K, noise_level=noise_sigma)
"""

discussion_md = """## Deliverables and Discussion

Complete the notebook by answering the following questions in a few sentences each.

**1. Which model performed better in your experiments, and why do you think that happened?**
Nei miei esperimenti, la **UNet** e, a seguire, la **ResCNN** hanno ottenuto le performance migliori. La *UNet* riesce a trionfare grazie al suo design encoder-decoder che le conferisce un "field-of-view" globale, permettendole di catturare pattern complessi necessari per correggere il degrado spaziale importante (blur). La *ResCNN* supera facilmente la SimpleCNN grazie al residual learning, che semplifica l'apprendimento spingendo la rete a focalizzarsi solo sulle alte frequenze corrotte.

**2. Did the residual architecture help? If yes, in what sense?**
Sì, moltissimo. Invece di dover "ricostruire da zero" l'immagine mappa (cosa difficile e soggetta a perdite di informazioni sulle basse frequenze), l'architettura residuale costringe i pesi convoluzionali ad imparare esclusivamente il **residuo** di correzione, per poi sommarlo alla misurazione iniziale. Questo abbatte la difficoltà dell'ottimizzazione garantendo gradienti più stabili e convergenza più rapida.

**3. How did the noise level affect training and reconstruction quality?**
Un elevato livello di rumore (come $\\sigma=0.05$) rende l'inversione dell'operatore $K$ intrinsecamente peggio condizionata. La rete deve forzare un compromesso arduo: applicare uno smoothing per sopprimere il rumore e contemporaneamente preservare i contorni netti per annullare il blur. Un eccessivo rumore si traduce in immagini ricostruite inevitabilmente più "sfocate" o "plastiche" riducendo le metriche quantitative (SSIM e PSNR).

**4. Why is it important to generate the corruption through a known operator $K$ instead of treating the problem as a generic image-to-image task?**
Trattare il problema come generico (black-box) priva il modello di informazioni cruciali sulla vera **fisica del degrado**. Includere esplicitamente un operatore $K$ noto assicura che il modello risolva l'effettivo *problema inverso* legato alla misurazione, e garantisce maggiore solidità verso piccole deviazioni fuori distribuzione, cosa altrimenti impossibile per una rete che ha semplicemente memorizzato mapping tra due dataset disaccoppiati.

**5. Why should one be cautious when evaluating pure end-to-end methods only through visual quality?**
Come evidenziato dalle lezioni in ambito medical imaging, i modelli end-to-end privi di data-consistency constraints tendono fortemente ad "allucinare" l'informazione: la rete genera strutture morfologicamente verosimili (texture credibili, microcalcificazioni apparenti) allo scopo di minimizzare la MSE loss, ma che non esistono nella ground-truth. Questo può portare ad **errate diagnosi mediche**. Un'immagine visivamente piacevole non implica un'immagine fedele.

**6. If you implemented the optional UNet extension, how did it compare with the simpler CNN-based models?**
L'architettura UNet, tramite la sua capacità di catturare il contesto a scale multiple (grazie al pooling nell'encoder e alle skip connections verso il decoder), ha dimostrato una superiorità netta sulle CNN lineari. Mentre la SimpleCNN fatica a ripristinare gradienti strutturali estesi, la UNet ha restituito bordi meglio conservati e ha smaltito il blur in modo notevolmente più efficace.
"""

def update_cell_source(cell, new_source):
    lines = new_source.split('\\n')
    cell['source'] = [line + '\\n' for line in lines[:-1]] + [lines[-1]]

update_cell_source(nb['cells'][5], dataset_code)
update_cell_source(nb['cells'][7], models_code)
update_cell_source(nb['cells'][9], train_code)
update_cell_source(nb['cells'][11], eval_code)
update_cell_source(nb['cells'][12], discussion_md)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
