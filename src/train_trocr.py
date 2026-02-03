# ----------------------------------------------------------------------------- #

import os
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_scheduler
)
from torch.optim import AdamW

from tqdm import tqdm

# ----------------------------------------------------------------------------- #

def crop_from_corners(image, corners):
    xs = [p[0] for p in corners.values()]
    ys = [p[1] for p in corners.values()]
    return image.crop((min(xs), min(ys), max(xs), max(ys)))

class LicensePlateDataset(Dataset):
    def __init__(self, root_dirs, processor, split="train"):
        self.samples = []
        self.processor = processor
        self.split = split

        self.transform = transforms.Compose([
            transforms.Resize((64, 256)),  # TrOCR likes wide images
            transforms.ColorJitter(contrast=0.2),
        ])

        for root in root_dirs:
            for track_dir in sorted(Path(root).glob("track_*")):
                track_id = int(track_dir.name.split("_")[1])

                is_val = (track_id % 10 == 0)
                if split == "train" and is_val:
                    continue
                if split == "val" and not is_val:
                    continue

                ann_path = track_dir / "annotations.json"
                with open(ann_path, "r") as f:
                    ann = json.load(f)

                text = ann["plate_text"]

                for i in range(1, 6):
                    img_name = f"lr-00{i}_c.png"
                    img_path = track_dir / img_name

                    corner_key = img_name.replace("_c", "")

                    if corner_key not in ann["corners"]:
                        # In case there are no annotations.
                        continue

                    corners = ann["corners"][corner_key]
                    self.samples.append((img_path, corners, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, corners, text = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = crop_from_corners(image, corners)
        image = self.transform(image)

        pixel_values = self.processor(
            image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=10,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text
        }

# ----------------------------------------------------------------------------- #

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.vocab_size = model.config.decoder.vocab_size

model.to(device)

roots = [
    r"C:\Users\bgat\Desktop\Antonis\competition\train\Scenario-A\Brazilian",
    r"C:\Users\bgat\Desktop\Antonis\competition\train\Scenario-A\Mercosur",
]

train_ds = LicensePlateDataset(roots, processor, split="train")
val_ds   = LicensePlateDataset(roots, processor, split="val")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

num_epochs = 15
num_training_steps = num_epochs * len(train_loader)

lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps,
)

scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# ----------------------------------------------------------------------------- #

def train():
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in loop:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()

            loop.set_postfix(loss=loss.item())

        acc = evaluate()
        print(f"Validation accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained("trocr_lp_best")
            processor.save_pretrained("trocr_lp_best")

# ----------------------------------------------------------------------------- #

def evaluate():
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)

            generated_ids = model.generate(
                pixel_values,
                max_length=10,
                num_beams=5
            )

            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            gts = batch["text"]

            for p, g in zip(preds, gts):
                total += 1
                if p == g:
                    correct += 1

    return correct / total

# ----------------------------------------------------------------------------- #

def main():
    train()
    evaluate()

# ----------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
