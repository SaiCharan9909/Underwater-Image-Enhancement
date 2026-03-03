"""
Dataset and DataLoader creation utilities.
Supports both training and validation phases.
"""

import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    """
    Create DataLoader for training or validation.
    """

    if phase == "train":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt.get("batch_size", 1),
            shuffle=dataset_opt.get("use_shuffle", True),
            num_workers=dataset_opt.get("num_workers", 4),
            pin_memory=True
        )

    elif phase == "val":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

    else:
        raise ValueError(f"Unsupported phase: {phase}")


def create_dataset(dataset_opt, phase):
    """
    Create dataset instance.
    """

    from data.dataset import UIEDataset

    dataset = UIEDataset(
        dataroot=dataset_opt["dataroot"],
        resolution=dataset_opt.get("resolution", 256),
        split=phase,
        data_len=dataset_opt.get("data_len", None),
    )

    print(f"Dataset [{dataset.__class__.__name__}] created for {phase} phase.")

    return dataset