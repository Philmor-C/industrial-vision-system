import torch
import numpy as np
import torch.nn.functional as F

# Load checkpoint
checkpoint = torch.load("models/patchcore.pth", map_location="cpu")

memory_bank = checkpoint["memory_bank"]
threshold = checkpoint["threshold"]

def run_patchcore(image, backbone):
    """
    image: tensor [1,3,H,W]
    backbone: feature extractor model
    """

    with torch.no_grad():
        features = backbone(image)

        # distance to memory bank
        distances = torch.cdist(features, memory_bank)
        dist_score, _ = torch.min(distances, dim=1)

        # image-level anomaly score
        score = torch.max(dist_score)

        # heatmap (patch-level)
        heatmap = dist_score.view(1, 1, 28, 28)

        heatmap = F.interpolate(
            heatmap,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

    heatmap = heatmap.squeeze().cpu().numpy()
    score = score.item()

    return heatmap, score, threshold