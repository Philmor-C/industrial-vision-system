import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_patchcore():
    path = hf_hub_download("Filiyo/patchcore", "patchcore.pth")
    data = torch.load(path, map_location=DEVICE)

    memory_bank = data["memory_bank"].to(DEVICE)
    threshold = data["threshold"]

    return memory_bank, threshold


def run_patchcore(image, backbone, memory_bank):
    features = backbone(image)

    dist = torch.cdist(features, memory_bank)
    dist_score, _ = torch.min(dist, dim=1)

    score = torch.max(dist_score)

    heatmap = dist_score.view(1,1,28,28)

    heatmap = F.interpolate(
        heatmap,
        size=(224,224),
        mode="bilinear",
        align_corners=False
    )

    return heatmap.squeeze().detach().cpu().numpy(), score.item()
