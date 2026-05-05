import torch

class ImageNormalizer:
    """
    Gestisce la normalizzazione e denormalizzazione delle immagini per i task di Super-Risoluzione.
    
    La normalizzazione matematica è cruciale per la stabilità numerica e per preservare 
    le proprietà statistiche del rumore additivo nei problemi inversi.
    """
    def __init__(self, mode='0_1'):
        """
        mode: 
          - '0_1': (Default) normalizza i pixel nel range [0, 1]. Ottimale per stabilità numerica, 
                   calcolo del PSNR/SSIM, e per il metodo variazionale (Total Variation).
          - 'neg1_1': normalizza i pixel nel range [-1, 1]. Usato da alcune reti generative.
        """
        self.mode = mode

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applica la normalizzazione definita al tensore di input.
        Scala l'input [0, 255] al dominio desiderato.
        """
        if tensor.max() > 1.0:
            tensor = tensor.float() / 255.0
            
        if self.mode == '0_1':
            return tensor
        elif self.mode == 'neg1_1':
            return (tensor * 2.0) - 1.0
        else:
            raise ValueError(f"Modalità di normalizzazione {self.mode} non supportata.")

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Riporta il tensore dal range normalizzato al range standard [0, 1] 
        per il salvataggio su disco o calcolo metriche.
        """
        if self.mode == '0_1':
            return torch.clamp(tensor, 0.0, 1.0)
        elif self.mode == 'neg1_1':
            return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)

# Istanza globale di default per uso nel progetto
default_normalizer = ImageNormalizer(mode='0_1')
