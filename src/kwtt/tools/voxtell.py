import numpy as np


class VoxTellTool:
    """Thin wrapper around the VoxTell predictor."""

    def __init__(self, device="cuda:0"):
        import torch
        from voxtell.inference.predictor import VoxTellPredictor
        self.predictor = VoxTellPredictor(device=torch.device(device))

    def predict(self, img, prompts):
        """img: array (1, z, y, x) in RAS. Returns a list of boolean masks, one per prompt."""
        out = self.predictor.predict_single_image(img, prompts)
        return [np.asarray(m) > 0 for m in out]