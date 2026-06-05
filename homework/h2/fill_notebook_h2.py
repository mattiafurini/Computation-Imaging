import json
import os

notebook_path = '/home/mattia/Documents/Computational Imaging/homework/h2/homework2.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

def set_cell_source(index, source_text):
    nb['cells'][index]['source'] = [line + '\n' for line in source_text.split('\n')[:-1]] + [source_text.split('\n')[-1]]

# Cell indices in homework2.ipynb based on the `view_file` output:
# Cell 0: Markdown title
# Cell 1: Markdown Goals
# Cell 2: Markdown Structure
# Cell 3: Imports and paths
# Cell 4: Markdown Part 1
# Cell 5: Dataset class (lines 138-164)
# Cell 6: Markdown Part 2
# Cell 7: VAE latent objective (lines 192-208)
# Cell 8: Markdown Part 3
# Cell 9: Diffusion prior (lines 232-247)
# Cell 10: Markdown Part 4
# Cell 11: Quantitative eval (lines 269-273)
# Cell 12: Markdown Deliverables

# === Cell 5: Dataset & Models ===
cell_5_source = """class MayoDataset(Dataset):
    def __init__(self, data_path, data_shape=64):
        super().__init__()
        self.fname_list = sorted(glob.glob(f'{data_path}/*/*.png'))
        self.transform = transforms.Compose([
            transforms.Resize((data_shape, data_shape), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):
        x = Image.open(self.fname_list[idx]).convert('L')
        return self.transform(x)

# VAE Architecture definition (from lecture notebook)
import torch.nn as nn
import torch.nn.functional as F

def norm_layer(channels):
    num_groups = 8 if channels >= 8 else 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            norm_layer(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            norm_layer(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            norm_layer(out_ch),
            nn.SiLU(),
            ResidualBlock(out_ch),
        )

    def forward(self, x):
        return self.block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            norm_layer(out_ch),
            nn.SiLU(),
            ResidualBlock(out_ch),
        )

    def forward(self, x):
        return self.block(x)

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            norm_layer(32),
            nn.SiLU(),
            ResidualBlock(32),
        )
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4)

        self.up1 = UpsampleBlock(512, 256)
        self.up2 = UpsampleBlock(256, 128)
        self.up3 = UpsampleBlock(128, 64)
        self.up4 = UpsampleBlock(64, 32)
        self.out_conv = nn.Sequential(
            norm_layer(32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(min=-4.5, max=1.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z).view(z.shape[0], 512, 4, 4)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        return self.out_conv(h)

# 1. Datasets and Dataloaders
train_dataset = MayoDataset(data_path=str(book_root / 'Mayo' / 'train'), data_shape=64)
test_dataset = MayoDataset(data_path=str(book_root / 'Mayo' / 'test'), data_shape=64)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
x_clean = next(iter(test_loader)).to(device)

# 2. Forward operator and noise
K = operators.Blurring(img_shape=(64, 64), kernel_type='motion', kernel_size=9, motion_angle=20)
noise_level = 0.05
y_delta = K(x_clean)
y_delta = y_delta + noise_level * torch.randn_like(y_delta)

# 3. Visualize
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Clean image")
plt.imshow(x_clean[0,0].cpu(), cmap='gray', vmin=-1, vmax=1)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title("Corrupted y_delta")
plt.imshow(y_delta[0,0].cpu(), cmap='gray', vmin=-1, vmax=1)
plt.axis('off')
plt.show()

# 4. Load Models
vae = ConvVAE(latent_dim=64).to(device)
vae.load_state_dict(torch.load(weights_dir / 'VAE.pth', map_location=device))
vae.eval()

diff_model = DiffusionUNet(
    in_ch=1, base_ch=64, channel_mults=(1, 2, 4), time_dim=256, dropout=0.05, attn_levels=(1, 2)
).to(device)
diff_model.load_state_dict(torch.load(weights_dir / 'DDPMDenoiser.pth', map_location=device))
diff_model.eval()

print("Models loaded successfully.")
"""
set_cell_source(5, cell_5_source)

# === Cell 7: VAE Reconstruction ===
cell_7_source = """def latent_objective(z, generator, y_delta, K, lam=1e-3):
    # Decode z
    # Note: VAE decoder outputs [0, 1]. The forward model expects [-1, 1] inputs?
    # Actually, in the lecture notebook, the VAE output is Sigmoid so it's [0,1].
    # But y_delta was generated from x_clean which is [-1, 1].
    # We must match the normalization. Let's assume VAE output is in [0, 1] and we map it to [-1, 1].
    x_hat_01 = generator.decode(z)
    x_hat = x_hat_01 * 2.0 - 1.0  # mapped to [-1, 1]
    
    # Forward operator
    y_hat = K(x_hat)
    
    # Objective: Data fidelity + L2 regularization on z
    data_fidelity = F.mse_loss(y_hat, y_delta)
    reg = lam * torch.sum(z**2)
    return data_fidelity + reg, x_hat

def reconstruct_with_vae_prior(generator, y_delta, K, latent_dim, num_steps=500, lr=1e-2, lam=1e-3):
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False
        
    # Initialize z from standard normal
    z = torch.randn(1, latent_dim, device=y_delta.device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)
    
    loss_history = []
    
    pbar = tqdm(range(num_steps), desc='Latent Optimization')
    for step in pbar:
        optimizer.zero_grad()
        loss, x_hat = latent_objective(z, generator, y_delta, K, lam)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if step % 50 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
    # Final decode
    with torch.no_grad():
        _, x_final = latent_objective(z, generator, y_delta, K, lam)
        
    return x_final, loss_history, z.detach()

# Run VAE Reconstruction
x_vae, vae_losses, z_opt = reconstruct_with_vae_prior(vae, y_delta, K, latent_dim=64, num_steps=500, lr=0.1, lam=1e-3)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(vae_losses)
plt.title("VAE Latent Optimization Loss")
plt.xlabel("Steps")
plt.subplot(1, 2, 2)
plt.imshow(x_vae[0,0].cpu(), cmap='gray', vmin=-1, vmax=1)
plt.title("VAE Prior Reconstruction")
plt.axis('off')
plt.show()
"""
set_cell_source(7, cell_7_source)

# === Cell 9: Diffusion Reconstruction ===
cell_9_source = """def predict_x0_from_eps(x_t, eps_pred, t, alpha_bars):
    # Reconstruct the clean-image estimate from x_t and the predicted noise.
    # formula: x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
    sqrt_alpha_bar = extract(alpha_bars.sqrt(), t, x_t.shape)
    sqrt_one_minus_alpha_bar = extract((1 - alpha_bars).sqrt(), t, x_t.shape)
    return (x_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar

def reconstruct_with_diffusion_prior(model, y_delta, K, alpha_bars, num_steps=400, rho=10.0, noise_level=0.05):
    # DiffPIR style reconstruction
    # model: denoiser
    device = y_delta.device
    x_t = torch.randn_like(y_delta) # Initialize with pure noise
    
    pbar = tqdm(reversed(range(0, num_steps)), total=num_steps, desc='DiffPIR')
    
    for i in pbar:
        t = torch.tensor([i], device=device, dtype=torch.long)
        
        # 1. Denoiser step (predict x0)
        with torch.no_grad():
            eps_pred = model(x_t, t)
            x0_hat = predict_x0_from_eps(x_t, eps_pred, t, alpha_bars)
            
        # 2. Data-fidelity step (gradient descent on ||Hx - y||^2)
        # Using DiffPIR analytical form or gradient step. Here we use gradient step.
        # x0_hat = x0_hat - rho / (sigma^2) * H^T(H*x0_hat - y)
        # Wait, for robustness, we optimize via Autograd
        x0_hat.requires_grad_(True)
        data_fidelity = F.mse_loss(K(x0_hat), y_delta)
        grad_x0 = torch.autograd.grad(data_fidelity, x0_hat)[0]
        x0_hat = x0_hat.detach() - rho * grad_x0
        
        # Clamp to [-1, 1] range to avoid exploding
        x0_hat = x0_hat.clamp(-1.0, 1.0)
        
        # 3. Add implicit noise to go to x_{t-1}
        if i > 0:
            alpha_bar_next = alpha_bars[i-1]
            # DDIM update using the corrected x0_hat
            x_t = torch.sqrt(alpha_bar_next) * x0_hat + torch.sqrt(1 - alpha_bar_next) * eps_pred
        else:
            x_t = x0_hat
            
    return x_t

# Define diffusion schedule
num_diffusion_steps = 400
betas = cosine_beta_schedule(num_diffusion_steps)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# Run Diffusion Reconstruction
x_diff = reconstruct_with_diffusion_prior(diff_model, y_delta, K, alpha_bars, num_steps=400, rho=50.0, noise_level=noise_level)

plt.figure(figsize=(5, 5))
plt.imshow(x_diff[0,0].cpu(), cmap='gray', vmin=-1, vmax=1)
plt.title("DiffPIR Reconstruction")
plt.axis('off')
plt.show()
"""
set_cell_source(9, cell_9_source)

# === Cell 11: Quantitative Comparison ===
cell_11_source = """import sys
from pathlib import Path
project_root = Path('..').resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from src.metrics import compute_psnr, compute_ssim

# Normalize images to [0, 1] for metrics
def to_01(t):
    return (t.clamp(-1, 1) + 1.0) / 2.0

x_clean_01 = to_01(x_clean).cpu().numpy()[0, 0]
y_delta_01 = to_01(y_delta).cpu().numpy()[0, 0]
x_vae_01 = to_01(x_vae).cpu().detach().numpy()[0, 0]
x_diff_01 = to_01(x_diff).cpu().numpy()[0, 0]

metrics = {
    'Corrupted': {
        'PSNR': compute_psnr(x_clean_01, y_delta_01),
        'SSIM': compute_ssim(x_clean_01, y_delta_01)
    },
    'VAE Prior': {
        'PSNR': compute_psnr(x_clean_01, x_vae_01),
        'SSIM': compute_ssim(x_clean_01, x_vae_01)
    },
    'DiffPIR': {
        'PSNR': compute_psnr(x_clean_01, x_diff_01),
        'SSIM': compute_ssim(x_clean_01, x_diff_01)
    }
}

print("Quantitative Comparison:")
for k, v in metrics.items():
    print(f"{k:>12s} - PSNR: {v['PSNR']:.2f} dB, SSIM: {v['SSIM']:.4f}")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(x_clean_01, cmap='gray'); axes[0].set_title("Ground Truth")
axes[1].imshow(y_delta_01, cmap='gray'); axes[1].set_title(f"Degraded\\nPSNR: {metrics['Corrupted']['PSNR']:.2f}")
axes[2].imshow(x_vae_01, cmap='gray'); axes[2].set_title(f"VAE Prior\\nPSNR: {metrics['VAE Prior']['PSNR']:.2f}")
axes[3].imshow(x_diff_01, cmap='gray'); axes[3].set_title(f"DiffPIR Prior\\nPSNR: {metrics['DiffPIR']['PSNR']:.2f}")
for ax in axes: ax.axis('off')
plt.show()
"""
set_cell_source(11, cell_11_source)

# Write out the modified notebook
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook modified successfully.")
