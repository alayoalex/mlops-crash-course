import torch
from torch.utils.data import DataLoader
import tiktoken
from spam_dataset import SpamDataset

tokenizer = tiktoken.get_encoding("gpt2")
num_workers = 0
batch_size = 8

torch.manual_seed(42)

train_dataset = SpamDataset(
    csv_file="data/train.csv",
    max_length=None,
    tokenizer=tokenizer
)

val_dataset = SpamDataset(
    csv_file="data/validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

test_dataset = SpamDataset(
    csv_file="data/test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False,
)

print("Train Loader")
for input_batch, label_batch in train_loader:
    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions:", label_batch.shape)
    break
# print("Input batch dimensions:", input_batch.shape)

print(f"{len(train_loader)} batches in the train loader")
print(f"{len(val_loader)} batches in the validation loader")
print(f"{len(test_loader)} batches in the test loader")
