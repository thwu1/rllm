"""
Train a math agent using TinkerAgentTrainer.

This version uses TinkerAgentTrainer which internally uses the separated
architecture (TinkerTrajectoryGenerator + TinkerPolicyTrainer) while providing
a simplified API similar to the original trainer.
"""

from torch.utils.data import DataLoader

from rllm.data.dataset import DatasetRegistry

train_dataset = DatasetRegistry.load_dataset("gsm8k", "train")
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=64,
    shuffle=True,
    collate_fn=lambda x: x  # Return batches as lists
)

class SimpleDataLoader:
    """Simple reusable dataloader."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            yield self.dataset[i : i + self.batch_size]


def create_dataloader(dataset, batch_size):
    """Create a simple reusable dataloader from dataset."""
    return SimpleDataLoader(dataset, batch_size)




if __name__ == "__main__":
    num_batchs = 0
    for batch in train_dataloader:
        num_batchs += 1
        if num_batchs == 1:
            print(batch)
    
    print(num_batchs)

    for batch in train_dataloader:
        num_batchs += 1

    print(num_batchs)

    dataloader = create_dataloader(train_dataset, 64)
    num_batchs = 0
    for batch in dataloader:
        num_batchs += 1
        if num_batchs == 1:
            print(batch)
    
    print(num_batchs)
    for batch in dataloader:
        num_batchs += 1
    print(num_batchs)
