import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW
from datasets import load_dataset



DATA_FILE = "data/cards_encoded.csv"
MODEL_OUT = "mtg-gpt2-model-ver10"

SPECIAL_TOKENS = [
    "<|name|>", "<|manaCost|>", "<|type|>", "<|text|>",
    "<|power|>", "<|toughness|>", "<|loyalty|>"
]
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset...")
dataset = load_dataset("text", data_files={"train": DATA_FILE})
dataset = dataset["train"].select(range(20000))  # Example: use only 5k samples


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

class MTGDataset(Dataset):
    def __init__(self, dataset):
        self.input_ids = dataset["input_ids"]
        self.attn = dataset["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = torch.tensor(self.input_ids[idx])
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.tensor(self.attn[idx])
        }

train_ds = MTGDataset(dataset)
loader = DataLoader(train_ds, batch_size=8, shuffle=True)

print("Loading GPT-2 model...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 3
print("Starting training...")

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Batch {i}  Loss: {loss.item():.4f}")

print("Saving model...")
model.save_pretrained(MODEL_OUT)
tokenizer.save_pretrained(MODEL_OUT)
print("Done!")
