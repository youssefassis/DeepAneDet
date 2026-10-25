import torch
from torch import nn as nn
from deepanedet.utils import get_logger

logger = get_logger("Loss function")


class YOLO_Loss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.confidence_loss = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()
        logger.info(f"YOLO Loss is used ({self.confidence_loss} and {self.mse})")

    def compute_loss(self, prediction, target):
        batch_size = prediction[0].shape[0]
        mask = target[0][..., 0] == 1
        n_pos_cells = mask.sum().float()

        confidence_obj_loss = self.confidence_loss(prediction[0][..., 0][mask], target[0][..., 0][mask])
        confidence_noobj_loss = self.confidence_loss(prediction[0][..., 0][~mask], target[0][..., 0][~mask])

        confidence_obj_loss = 0.0 if torch.isnan(confidence_obj_loss) else confidence_obj_loss
        confidence_noobj_loss = 0.0 if torch.isnan(confidence_noobj_loss) else confidence_noobj_loss

        confidence_loss = confidence_obj_loss + 0.5 * n_pos_cells * confidence_noobj_loss

        regression_loss = 0.0
        if n_pos_cells > 0:
            prediction[1][..., 0:3] = self.sigmoid(
                prediction[1][..., 0:3]
            )  # Normalize x y z coords (relative to grid size)
            prediction[1][..., 3:] = torch.exp(prediction[1][..., 3:])  # radius should be positive
            regression_loss = self.mse(prediction[1][..., 0:][mask], target[1][..., 0:][mask])
        return (confidence_loss + 5 * regression_loss) / batch_size

    def forward(self, predictions, targets):
        loss = 0.0
        for i in range(len(predictions)):
            loss += self.compute_loss(prediction=predictions[i], target=targets[i])
        return loss


def get_loss_criterion(name):
    assert name is not None, "Could not find any loss function"

    if name == "YOLO_Loss":
        return YOLO_Loss()
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
