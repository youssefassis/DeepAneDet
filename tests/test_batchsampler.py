from deepanedet.training.batch_sampler import BalancedBatchSampler


def test_batches_hold_integer_indices_without_negative_patches():
    patches = [{"dir": "P0001", "status": True}] * 4 + [{"dir": "P0001", "status": False}]
    sampler = BalancedBatchSampler(patches, batch_size=2, percentage_neg_patches=15)  # rounds to 0 negatives

    batches = list(sampler)

    assert sorted(i for batch in batches for i in batch) == [0, 1, 2, 3]
    assert all(isinstance(i, int) for batch in batches for i in batch)
