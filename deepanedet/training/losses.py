import torch
import numpy as np
from torch import nn as nn
from deepanedet.utils import get_logger

logger = get_logger("Loss function")


def intersection_over_union(prediction, truth):
    def intersection(point1, point2):
        pos1, pos2 = point1[:3], point2[:3]
        r1, r2 = point1[3], point2[3]
        d = sum((pos1 - pos2) ** 2) ** 0.5
        if d >= (r1 + r2):
            return 0.0  # no overlapping
        elif d == 0 and min(r1, r2) > 0:
            return 4.0 / 3.0 * np.pi * min(r1, r2) ** 3  # part is overlapping
        elif r1 > (d + r2):
            return 4.0 / 3.0 * np.pi * r2**3
        elif r2 > (d + r1):
            return 4.0 / 3.0 * np.pi * r1**3
        vol = (
            np.pi * (r1 + r2 - d) ** 2 * (d**2 + (2 * d * r1 - 3 * r1**2 + 2 * d * r2 - 3 * r2**2) + 6 * r1 * r2)
        ) / (12 * d)
        return vol

    ious = torch.zeros(prediction.shape[0], dtype=torch.float32)
    for i in range(ious.shape[0]):
        inter = intersection(prediction[i].cpu().detach().numpy(), truth[i].cpu().detach().numpy())
        union = (
            (4 / 3) * np.pi * (prediction[i][3].cpu().detach().numpy() ** 3)
            + (4 / 3) * np.pi * (truth[i][3].cpu().detach().numpy() ** 3)
            - inter
        )
        ious[i] = inter / union + 1e-6
    return ious


# Bounding spheres or boxes
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


class YOLO_Loss_anchors(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.confidence_loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        logger.info(f"YOLO_Loss_anchors is used ({self.confidence_loss} and {self.mse})")

    def compute_loss(self, prediction, target, anchor):  # [x y z radius confidence]
        batch_size = prediction[0].shape[0]
        mask = target[0][..., 0] == 1
        noobj = target[0][..., 0] == 0

        n_pos_cells = mask.sum().float()
        anchors = anchor.reshape(1, prediction[0].shape[1], 1, 1, 1, 1)  # (1,nb_anchors,1,1,1)

        # Confidence
        ## No object loss
        confidence_noobj_loss = self.confidence_loss(prediction[0][..., 0][noobj], target[0][..., 0][noobj])
        if torch.isnan(confidence_noobj_loss):
            print(f"Nan confidence_noobj_loss: n_pos_cells={n_pos_cells}")
            confidence_noobj_loss = 0.0

        ## object loss
        confidence_obj_loss = 0.0
        if n_pos_cells > 0:
            spheres_preds = torch.cat(
                [self.sigmoid(prediction[1][..., 0:3]), torch.exp(prediction[1][..., 3:4]) * anchors], dim=-1
            )  # Concat (center coords, radius scaled)
            ious = (
                intersection_over_union(spheres_preds[..., 0:4][mask], target[1][..., 0:4][mask])
                .detach()
                .to(prediction[1].device)
            )
            confidence_obj_loss = self.mse(
                self.sigmoid(prediction[0][..., 0][mask]), ious * target[0][..., 0][mask]
            )  # confidence

            if torch.isnan(confidence_obj_loss):
                print(f"Nan confidence_obj_loss: n_pos_cells={n_pos_cells}")
                confidence_obj_loss = 0.0

        confidence_loss = confidence_obj_loss + 0.5 * n_pos_cells * confidence_noobj_loss

        # Regression
        regression_loss = 0.0
        if n_pos_cells > 0:
            # Regression MSE
            prediction[1][..., 0:3] = self.sigmoid(
                prediction[1][..., 0:3]
            )  # Normalize x y z coords (relative to grid size)
            target[1][..., 3:4] = torch.log((1e-16 + target[1][..., 3:4] / anchors))  # radius
            regression_loss = self.mse(prediction[1][..., 0:][mask], target[1][..., 0:][mask])
            if torch.isnan(regression_loss):
                print(f"Nan regression_loss: n_pos_cells={n_pos_cells}")
                regression_loss = 0.0

        return (confidence_loss + 5 * regression_loss) / batch_size

    def forward(
        self, predictions, targets, anchors
    ):  # compute loss for each of the three scales [[Batch, 1, S, S, S, params], [Batch, 1, S, S, S, 1]]
        loss = 0.0
        for i in range(len(predictions)):
            loss += self.compute_loss(predictions[i], targets[i], anchors[i])
        return loss


def get_loss_criterion(name):
    assert name is not None, "Could not find any loss function"

    if name == "YOLO_Loss":
        return YOLO_Loss()
    elif name == "YOLO_Loss_anchors":
        return YOLO_Loss_anchors()
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
