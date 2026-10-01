from torch.utils.data import DataLoader, random_split
import torch
from typing import Tuple

import config

def get_dataloaders(strategy) -> Tuple[DataLoader, DataLoader, DataLoader]:
    full_train_dataset, test_dataset = strategy.get_datasets(config.DATA_DIR)

    validation_size = int(0.15 * len(full_train_dataset))
    train_size = len(full_train_dataset) - validation_size

    train_dataset, validation_dataset = random_split(full_train_dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(config.RANDOM_SEED))

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"[{strategy.name}] Train: {len(train_dataset)}, Validation: {len(validation_dataset)}, Test: {len(test_dataset)}")
    return train_loader, validation_loader, test_loader