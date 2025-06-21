from collections import Counter
import logging

import numpy as np
import torch
from typing import Tuple

logger = logging.getLogger(__name__)

UNKNOWN_TAG = "unknown"


def model_predict_label(model: torch.nn.Module, img: torch.Tensor) -> Tuple[int, torch.Tensor]:
    assert img.dim() == 4
    assert img.size(0) == 1
    model.eval()
    img = img.to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(img)
        _, pred = torch.max(logits, 1)
    return pred.item(), logits[0]

def model_predict_label_batch(model: torch.nn.Module, img: torch.Tensor) -> Tuple[int, torch.Tensor]:
    assert img.dim() == 4
    model.eval()
    img = img.to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(img)
        _, pred = torch.max(logits, 1)
    return pred[0].item(), logits[0] 

class DeviceIdentifierOneVsOne:
    UNKNOWN_TAG = UNKNOWN_TAG

    def __init__(self, border_img_group: list[dict]) -> None:
        self.border_img_group = border_img_group

    @staticmethod
    def _predict_unit(
        model, border_img: torch.Tensor, tag_1: str, tag_1_label: int, tag_2: str, tag_2_label: int, eval_batch_size: int
    ) -> Tuple[int, torch.Tensor]:
        if eval_batch_size==1:
            model_pred_label, model_pred_logits = model_predict_label(model, border_img)
        else:
            model_pred_label, model_pred_logits = model_predict_label_batch(model, border_img)
        print("Predict unit: ",tag_1,tag_2,tag_1_label,tag_2_label,model_pred_label)
        if model_pred_label == tag_1_label:
            return tag_1, model_pred_logits
        elif model_pred_label == tag_2_label:
            return tag_2, model_pred_logits
        else:
            return UNKNOWN_TAG, model_pred_logits

    @torch.no_grad()
    def predict(self, model: torch.nn.Module, eval_batch_size: int, return_counter: bool = False) -> str | Counter:
        model.eval()
        pred_devices = []

        for border_img_meta in self.border_img_group:
            if "border_image" not in border_img_meta:
                border_img = torch.load(border_img_meta["border_image_path"])
            else:
                border_img = border_img_meta["border_image"]
            tag_1 = border_img_meta["model_1_tag"]
            tag_1_label = int(border_img_meta["model_1_pred_label"])
            tag_2 = border_img_meta["model_2_tag"]
            tag_2_label = int(border_img_meta["model_2_pred_label"])
            if eval_batch_size!=1:
                random_imgs = torch.randn(eval_batch_size-1, *border_img.shape[1:])
                border_img = torch.cat([border_img, random_imgs], dim=0)
            pred_devices.append(self._predict_unit(model, border_img, tag_1, tag_1_label, tag_2, tag_2_label, eval_batch_size))
        print(pred_devices)
        # remove unknown tags
        pred_devices = [device[0] for device in pred_devices if device[0] != UNKNOWN_TAG]
        pred_devices = Counter(pred_devices)

        if return_counter:
            return pred_devices

        if len(pred_devices) == 0:
            return UNKNOWN_TAG
        else:
            return pred_devices.most_common(1)[0][0]
            
