import torch
import numpy as np
from deepanedet.utils import get_logger
from functools import reduce

logger = get_logger("Balanced Batch Sampler")


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    Randomly select a certain number of negative patches per patient and yield batches of balanced
    positive and negative samples of a given batch size.

    1- For each PATIENT, shuffle their corresponding negative patches and select a given number of
    negative patches to be added to neg_indices. If nb_negative_samples_per_patient is None, all
    negative patches for a given patient are selected.
    2- For each BATCH of a given size (batch_size), randomly select a certain number of positive
    and negative samples according to their respective ratios (pos_ratio and neg_ratio).
    If drop_last is True and there are not enough samples left to make a complete batch, skip the
    remaining samples.
    """

    def __init__(
        self, patches, batch_size, nb_negative_samples_per_patient=None, percentage_neg_patches=None, drop_last=False
    ):
        if nb_negative_samples_per_patient is not None and percentage_neg_patches is not None:
            raise ValueError("Negative patches selection should be either percentage or fixed number per patient")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.patches = patches

        self.pos_indices = np.array([idx for idx, p in enumerate(self.patches) if p["status"]], dtype="int")
        self.neg_indices = [
            [idx for idx, y in enumerate(self.patches) if y["dir"] == x and (not y["status"])]
            for x in set(map(lambda x: x["dir"], self.patches))
        ]

        nb_patients = len(self.neg_indices)

        if nb_negative_samples_per_patient is not None:  # fixed number of negative samples per patient
            self.nb_negative_samples_per_patient = nb_negative_samples_per_patient
            self.num_samples_total = self.pos_indices.shape[0] + nb_patients * self.nb_negative_samples_per_patient

        elif percentage_neg_patches is not None:  # fixed propotion of negative samples per patient
            self.nb_negative_samples_per_patient = int(
                ((self.pos_indices.shape[0] * percentage_neg_patches) / (100 - percentage_neg_patches)) / nb_patients
            )
            self.num_samples_total = self.pos_indices.shape[0] + nb_patients * self.nb_negative_samples_per_patient

        else:  # without random negative samples selection
            self.nb_negative_samples_per_patient = None
            self.num_samples_total = self.pos_indices.shape[0] + reduce(
                lambda count, patient_indices: count + len(patient_indices), self.neg_indices, 0
            )

        self.nb_batches = self.get_n_batches()

        logger.info(f"Drop last batch={drop_last}")
        logger.info(
            f"Randomly loading {self.nb_negative_samples_per_patient} negative patches/ patient (from {len(self.neg_indices)} patients)"
        )
        logger.info(
            f"{self.num_samples_total} patches = {self.pos_indices.shape[0]} ({np.round((self.pos_indices.shape[0] / self.num_samples_total) * 100, 2)}%) positives + {self.num_samples_total - self.pos_indices.shape[0]} ({round(100 - (self.pos_indices.shape[0] / self.num_samples_total) * 100, 2)}%) negatives"
        )

    def get_n_batches(self):
        return (
            self.num_samples_total // self.batch_size
            if np.remainder(self.num_samples_total, self.batch_size) == 0
            else self.num_samples_total // self.batch_size + 1
        )

    def _get_n_elements_from(self, list_indices, nb_elements):
        return np.random.choice(list_indices, nb_elements, replace=False)

    def __iter__(self):
        if self.nb_negative_samples_per_patient is None:
            logger.info("Random selection of ALL negative patches per patient")
        else:
            logger.info(f"Random selection of {self.nb_negative_samples_per_patient} negative patches per patient")

        pos_indices = self.pos_indices[:]
        neg_indices = []

        # randomly select 'nb_negative_samples_per_patient' negative patches from each patient
        for patient_idx in range(len(self.neg_indices)):  # for each patient
            if len(self.neg_indices[patient_idx]) > 0:
                np.random.shuffle(self.neg_indices[patient_idx])
                if self.nb_negative_samples_per_patient is not None:
                    neg_indices.extend(self.neg_indices[patient_idx][: self.nb_negative_samples_per_patient])
                else:
                    neg_indices.extend(self.neg_indices[patient_idx])
        neg_indices = np.array(neg_indices, dtype=int)
        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)

        for batch_idx in range(self.nb_batches):
            left_samples = pos_indices.shape[0] + neg_indices.shape[0]
            if left_samples >= self.batch_size:
                pos_ratio = pos_indices.shape[0] / left_samples
                nb_pos_samples = int(pos_ratio * self.batch_size)
                nb_neg_samples = self.batch_size - nb_pos_samples

                # randomly select 'nb_pos_samples' from positive indices + remove indices from list
                pos_selected = self._get_n_elements_from(pos_indices, nb_pos_samples)
                pos_indices = np.setdiff1d(
                    pos_indices, pos_selected
                )  # np.delete(pos_indices, pos_indices[pos_selected])

                # randomly select 'nb_neg_samples' from negative indices + remove indices from list
                neg_selected = self._get_n_elements_from(neg_indices, nb_neg_samples)
                neg_indices = np.setdiff1d(neg_indices, neg_selected)

                indices = np.concatenate([pos_selected, neg_selected])
            else:
                if not self.drop_last:
                    indices = np.concatenate([pos_indices, neg_indices])
                else:
                    indices = None

            if indices is not None:
                np.random.shuffle(indices)
                yield indices.tolist()

    def __len__(self) -> int:
        return self.get_n_batches()
