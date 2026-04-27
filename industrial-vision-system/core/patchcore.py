import torch
import torch.nn.functional as F
import numpy as np
from huggingface_hub import hf_hub_download

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_patchcore():
    path = hf_hub_download("Filiyo/patchcore", "patchcore.pth")
    data = torch.load(path, map_location=DEVICE)

    memory_bank = data["memory_bank"].to(DEVICE)
    threshold = data["threshold"]

    return memory_bank, threshold


def run_patchcore(image, memory_bank):
    # image: tensor [1, C, H, W]

    B, C, H, W = image.shape

    # simple fast embedding (deployment-friendly)
    features = image.view(C, -1).T.to(memory_bank.device)

    distances = torch.cdist(features, memory_bank)

    dist_score, _ = torch.min(distances, dim=1)

    score = torch.max(dist_score)

    # build heatmap
    size = int(np.sqrt(len(dist_score)))
    heatmap = dist_score[:size*size].view(1,1,size,size)

    heatmap = F.interpolate(
        heatmap,
        size=(224,224),
        mode="bilinear",
        align_corners=False
    )

    return heatmap.squeeze().cpu().numpy(), score.item()
