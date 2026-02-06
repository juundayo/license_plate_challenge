# ----------------------------------------------------------------------------- #

import os
import json
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    get_scheduler
)
from torch.optim import AdamW

import matplotlib.pyplot as plt
import random

# ----------------------------------------------------------------------------- #

class Config:
    train_roots = [
        r"C:\Users\bgat\Desktop\Antonis\competition\train\Scenario-A\Brazilian",
        r"C:\Users\bgat\Desktop\Antonis\competition\train\Scenario-A\Mercosur",
    ]
    
    batch_size = 8
    num_epochs = 100
    learning_rate = 5e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    
    model_name = "microsoft/trocr-base-printed"

    # TrOCR config - not needed anymore. Original images are 33x19.
    #image_size = (384, 384) 
    
    val_split_ratio = 10
    
    patience = 5
    min_delta = 0.001
    
    output_dir = "./trocr_license_plate"
    best_model_path = "./trocr_license_plate_best"
    
config = Config()

# ----------------------------------------------------------------------------- #

def initialize_model(device):
    print("Initializing TrOCR model...")
    
    try:
        processor = TrOCRProcessor.from_pretrained(config.model_name)
        model = VisionEncoderDecoderModel.from_pretrained(config.model_name)
        
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.eos_token_id = processor.tokenizer.sep_token_id
        
        model.config.decoder.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.decoder.pad_token_id = processor.tokenizer.pad_token_id
        model.config.decoder.eos_token_id = processor.tokenizer.sep_token_id

        model.to(device)
        
        print(f"  Model loaded: {config.model_name}")
        print(f"  Device: {device}")
        print(f"  Model config properly set")
        
        return processor, model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# ----------------------------------------------------------------------------- #

def get_transforms(augment=True):
    '''Small augmentations to avoid overfitting to the training set.'''
    if augment:
        return transforms.Compose([
            transforms.Pad((0, (384-64)//2, 0, (384-64)//2)),  
            transforms.Resize((384, 384)),  
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Pad((0, (384-64)//2, 0, (384-64)//2)),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

# ----------------------------------------------------------------------------- #

def correct_corners(corners):
    '''
    Image cropping based on the bounding boxes provided 
    by the 1st set. Will use YOLO for set 2.
    '''
    pts = [
        corners["top-left"],
        corners["top-right"],
        corners["bottom-right"],
        corners["bottom-left"]
    ]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)

    return (left, top, right, bottom)

class LicensePlateDataset(Dataset):
    def __init__(self, root_dirs, processor, split="train", augment=True):
        self.samples = []
        self.processor = processor
        self.split = split
        self.augment = augment and split == "train"

        print(f"\n{'='*60}")
        print(f"Building {split.upper()} dataset")
        print(f"{'='*60}")

        for root in root_dirs:
            root_path = Path(root)
            if not root_path.exists():
                print(f"Warning: Path {root} does not exist!")
                continue

            track_dirs = sorted(list(root_path.glob("track_*")))
            print(f"Found {len(track_dirs)} track directories in {root_path.name}")

            for track_dir in track_dirs:
                track_id = int(track_dir.name.split("_")[1])

                is_val = (track_id % config.val_split_ratio == 0)
                if split == "train" and is_val:
                    continue
                if split == "val" and not is_val:
                    continue

                ann_path = track_dir / "annotations.json"
                if not ann_path.exists():
                    continue

                try:
                    with open(ann_path, "r") as f:
                        ann = json.load(f)

                    text = ann.get("plate_text", "").strip()
                    if len(text) != 7:
                        continue

                    for i in range(1, 6):
                        img_name = f"lr-00{i}.png"
                        img_path = track_dir / img_name
                        if not img_path.exists():
                            continue

                        corner_key = img_name
                        if "corners" in ann and corner_key in ann["corners"]:
                            corners = ann["corners"][corner_key]
                            self.samples.append({
                                "image_path": str(img_path),
                                "corners": corners,
                                "text": text,
                                "track_id": track_id
                            })
                except Exception:
                    continue

        print(f"Samples loaded: {len(self.samples)}")
        print(f"{'='*60}\n")

        if split == "train":
            np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            # Image loading.
            image = Image.open(sample["image_path"]).convert("RGB")
            crop_coords = correct_corners(sample["corners"])
            image = image.crop(crop_coords)

            if self.augment:
                from torchvision import transforms as T
                color_jitter = T.ColorJitter(brightness=0.1, contrast=0.1)
                image = color_jitter(image)

            # Image encoding.
            pixel_encoding = self.processor(
                images=image,
                return_tensors="pt"
            )
            pixel_values = pixel_encoding.pixel_values.squeeze(0)

            # Text encoding.
            text_encoding = self.processor.tokenizer(
                sample["text"],
                padding="max_length",
                max_length=10,
                truncation=True,
                return_tensors="pt"
            )
            labels = text_encoding.input_ids.squeeze(0)

            # Replacing pad tokens with -100 (TrOCR needs this).
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            return {
                "pixel_values": pixel_values,
                "labels": labels,
                "text": sample["text"]
            }

        except Exception:
            # Dummy fallback for debugging.
            dummy_image = Image.new('RGB', (128, 64), (0, 0, 0))
            pixel_encoding = self.processor(
                images=dummy_image,
                return_tensors="pt"
            )
            pixel_values = pixel_encoding.pixel_values.squeeze(0)

            dummy_text = "ERROR123"
            text_encoding = self.processor.tokenizer(
                dummy_text,
                padding="max_length",
                max_length=10,
                truncation=True,
                return_tensors="pt"
            )
            labels = text_encoding.input_ids.squeeze(0)
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            return {
                "pixel_values": pixel_values,
                "labels": labels,
                "text": dummy_text
            }

# ----------------------------------------------------------------------------- #

def visualize_dataset_samples(dataset, num_samples=5):
    """Visualizing the original image, bounding box, and cropped plate."""
    print("\n" + "=" * 60)
    print("VISUAL DATASET CHECK")
    print("=" * 60)

    indices = random.sample(range(len(dataset.samples)), num_samples)

    for idx in indices:
        sample = dataset.samples[idx]

        img_path = sample["image_path"]
        corners = sample["corners"]
        text = sample["text"]

        try:
            image = Image.open(img_path).convert("RGB")
            crop_coords = correct_corners(corners)
            cropped = image.crop(crop_coords)

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            # Original image
            axs[0].imshow(image)
            axs[0].set_title("Original")
            axs[0].axis("off")

            # Bounding box overlay
            axs[1].imshow(image)
            left, top, right, bottom = crop_coords
            rect = plt.Rectangle(
                (left, top),
                right - left,
                bottom - top,
                edgecolor="red",
                facecolor="none",
                linewidth=2
            )
            axs[1].add_patch(rect)
            axs[1].set_title(f"BBox | {text}")
            axs[1].axis("off")

            # Cropped image
            axs[2].imshow(cropped)
            axs[2].set_title("Cropped Plate")
            axs[2].axis("off")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Failed to visualize sample {idx}: {e}")
          

# ----------------------------------------------------------------------------- #

def collate_fn(batch):
    '''Simple collate function.'''
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    texts = [item["text"] for item in batch]
    
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "text": texts
    }

# ----------------------------------------------------------------------------- #

def debug_model_and_device():
    """Debug function to test everything before training."""
    print("=" * 60)
    print("DEBUG MODE")
    print("=" * 60)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n1. Testing model loading...")
    try:
        processor = TrOCRProcessor.from_pretrained(config.model_name)
        print("✓ Processor loaded")
        
        model = VisionEncoderDecoderModel.from_pretrained(config.model_name)
        print("✓ Model loaded")
        
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.eos_token_id = processor.tokenizer.sep_token_id
        
        model.config.decoder.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.decoder.pad_token_id = processor.tokenizer.pad_token_id
        model.config.decoder.eos_token_id = processor.tokenizer.sep_token_id
        
        model.to(device)
        print(f"✓ Model moved to {device}")
        
        print("\n2. Testing processor on license plate image size...")
        dummy_image = Image.new('RGB', (128, 64), (255, 255, 255))
        processed = processor(images=dummy_image, return_tensors="pt")
        print(f"✓ Processor works on 128x64 image")
        print(f"  Input shape: 128x64")
        print(f"  Output shape: {processed.pixel_values.shape}")
        print(f"  Model image size: {model.encoder.config.image_size}")
        
        print("\n3. Testing forward pass...")
        test_input = torch.randn(1, 3, 384, 384).to(device)
        test_labels = torch.randint(0, processor.tokenizer.vocab_size, (1, 10)).to(device)
        
        with torch.no_grad():
            output = model(pixel_values=test_input, labels=test_labels)
            print(f"✓ Forward pass successful!")
            print(f"  Loss: {output.loss.item():.4f}")
        
        print("\n4. Testing generation...")
        with torch.no_grad():
            generated_ids = model.generate(
                test_input,
                max_length=10,
                num_beams=2
            )
            pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            print(f"✓ Generation successful!")
            print(f"  Generated: {pred_texts[0]}")
        
        print("\n" + "=" * 60)
        print("ALL DEBUG TESTS PASSED! ✓")
        print("=" * 60)
        
        return True, processor, model, device
        
    except Exception as e:
        print(f"\n✗ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


# ----------------------------------------------------------------------------- #

def train_epoch(model, dataloader, optimizer, scheduler, processor, device, epoch):
    """Training pipeline!"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass.
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        # Backward pass.
        loss.backward()
        
        # Gradient clipping.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Loss calculation.
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

# ----------------------------------------------------------------------------- #

def validate(model, dataloader, processor, device):
    """Model validation."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    num_batches = 0
    
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Loss calculation.
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1
            
            # Prediction generation.
            generated_ids = model.generate(
                pixel_values,
                max_length=10,
                num_beams=4,
                early_stopping=True
            )
            
            pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            true_texts = batch["text"]
            
            predictions.extend(pred_texts)
            ground_truths.extend(true_texts)
            
            # Accuracy calculation.
            for pred, true in zip(pred_texts, true_texts):
                total += 1
                if pred == true:
                    correct += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy, predictions, ground_truths

# ----------------------------------------------------------------------------- #

def main_training():
    """Main training function"""
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    debug_success, processor, model, device = debug_model_and_device()
    
    if not debug_success:
        print("Debug failed. Exiting.")
        return
    
    os.makedirs(config.output_dir, exist_ok=True)

    print("\nLoading datasets...")
    try:
        train_dataset = LicensePlateDataset(
            config.train_roots, 
            processor, 
            split="train",
            augment=True
        )
        
        val_dataset = LicensePlateDataset(
            config.train_roots, 
            processor, 
            split="val",
            augment=False
        )
    except Exception as e:
        print(f"Error creating datasets: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    visualize_dataset_samples(train_dataset, num_samples=5)
    
    print("\n" + "=" * 60)
    print("DEBUGGING DATASET SAMPLES")
    print("=" * 60)
    
    for i in range(5):
        try:
            sample = train_dataset[i]
            print(f"\nSample {i}:")
            print(f"  Text: {sample['text']}")
            print(f"  Pixel values shape: {sample['pixel_values'].shape}")
            print(f"  Labels: {sample['labels']}")
            
            # Decoding labels to text (need to handle -100 values).
            labels_for_decoding = sample['labels'].clone()
            # Replacing -100 with pad_token_id for decoding.
            labels_for_decoding[labels_for_decoding == -100] = processor.tokenizer.pad_token_id
            decoded = processor.decode(labels_for_decoding, skip_special_tokens=True)
            print(f"  Decoded labels: {decoded}")
            
            img = sample['pixel_values']
            print(f"  Image min/max: {img.min():.3f}, {img.max():.3f}")
            print(f"  Image mean/std: {img.mean():.3f}, {img.std():.3f}")
        except Exception as e:
            print(f"\n✗ Error loading sample {i}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("CHECKING FOR NEGATIVE LABEL VALUES")
    print("=" * 60)
    
    # This checks for negative labels (the -100s).
    for i in range(min(10, len(train_dataset))):
        try:
            sample = train_dataset[i]
            labels = sample['labels']
            if (labels < 0).any():
                print(f"Sample {i} has negative labels")
                print(f"  Number of negative values: {(labels < 0).sum().item()}")
        except Exception as e:
            print(f"Error checking sample {i}: {e}")
            continue

    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True
        )
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Optimizer and scheduler.
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    best_accuracy = 0.0
    patience_counter = 0
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Training steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print("=" * 60)

    # Training loop!!
    for epoch in range(config.num_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{config.num_epochs}")
        print(f"{'='*60}")
        
        # Train.
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, 
            processor, device, epoch
        )
        
        # Validate.
        val_loss, val_acc, preds, truths = validate(
            model, val_loader, processor, device
        )
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        
        # Sample predictions per epoch.
        print(f"\nSample predictions:")
        for i in range(min(5, len(preds))):
            status = "✓" if preds[i] == truths[i] else "✗"
            print(f"  True: {truths[i]:<10} | Pred: {preds[i]:<10} | {status}")
        
        # Saves the best model.
        if val_acc > best_accuracy + config.min_delta:
            best_accuracy = val_acc
            patience_counter = 0
            
            print(f"\n  New best model! Accuracy: {val_acc:.4f}")
            
            model.save_pretrained(config.best_model_path)
            processor.save_pretrained(config.best_model_path)

            checkpoint_path = f"{config.output_dir}/epoch_{epoch+1}"
            model.save_pretrained(checkpoint_path)
            processor.save_pretrained(checkpoint_path)
            
            print(f"Model saved to: {config.best_model_path}")
        else:
            patience_counter += 1
            print(f"\nNo improvement for {patience_counter} epoch(s)")
        
        # Early stopping!
        if patience_counter >= config.patience:
            print(f"\n  Early stopping triggered after {epoch + 1} epochs")
            break
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Total epochs trained: {epoch + 1}")
    
    # Loading and testing the best model.
    if os.path.exists(config.best_model_path):
        print(f"\nTesting best model...")
        try:
            best_model = VisionEncoderDecoderModel.from_pretrained(config.best_model_path)
            best_processor = TrOCRProcessor.from_pretrained(config.best_model_path)
            best_model.to(device)
            
            print("\nBest model predictions:")
            print("-" * 40)
            
            test_indices = list(range(min(10, len(val_dataset))))
            for idx in test_indices:
                sample = val_dataset[idx]
                pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    generated_ids = best_model.generate(
                        pixel_values,
                        max_length=10,
                        num_beams=4,
                        early_stopping=True
                    )
                
                pred_text = best_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                status = "✓" if pred_text == sample["text"] else "✗"
                print(f"  True: {sample['text']:<10} | Pred: {pred_text:<10} | {status}")
        
        except Exception as e:
            print(f"Could not test best model: {e}")
    
    print(f"\nTraining completed successfully!")
    print(f"Best model saved to: {config.best_model_path}")

# ----------------------------------------------------------------------------- #

def simplified_training():
    """Simplified training approach"""
    print("=" * 60)
    print("SIMPLIFIED TRAINING")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = TrOCRProcessor.from_pretrained(config.model_name)
    
    print("\nLoading model with custom config...")
    
    config_model = VisionEncoderDecoderConfig.from_pretrained(config.model_name)
    
    config_model.decoder_start_token_id = processor.tokenizer.cls_token_id
    config_model.pad_token_id = processor.tokenizer.pad_token_id
    config_model.eos_token_id = processor.tokenizer.sep_token_id
    
    model = VisionEncoderDecoderModel.from_pretrained(
        config.model_name,
        config=config_model
    )
    
    model.to(device)
    print("  Model loaded and configured")
    
    print("\nTesting forward pass...")
    test_input = torch.randn(1, 3, 384, 384).to(device)
    test_labels = torch.randint(0, processor.tokenizer.vocab_size, (1, 10)).to(device)
    
    with torch.no_grad():
        output = model(pixel_values=test_input, labels=test_labels)
        print(f"  Test loss: {output.loss.item():.4f}")
    
    print("\nModel is ready for training!")
    return processor, model, device

# ----------------------------------------------------------------------------- #

def debug_dataset_samples():
    """More debugging to ensure preprocessing is correct."""
    processor, _, _ = simplified_training()
    dataset = LicensePlateDataset(config.train_roots, processor, split="train")
    
    for i in range(5):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Text: {sample['text']}")
        print(f"  Pixel values shape: {sample['pixel_values'].shape}")
        print(f"  Labels: {sample['labels']}")
        
        decoded = processor.decode(sample['labels'], skip_special_tokens=True)
        print(f"  Decoded labels: {decoded}")
        
        img = sample['pixel_values']
        print(f"  Image min/max: {img.min():.3f}, {img.max():.3f}")
        print(f"  Image mean/std: {img.mean():.3f}, {img.std():.3f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print("=" * 60)
    print("LICENSE PLATE RECOGNITION WITH TrOCR")
    print("=" * 60)
    
    print("\nQuick test: Loading processor and testing on one image...")
    try:
        processor = TrOCRProcessor.from_pretrained(config.model_name)
        
        test_path = r"C:\Users\bgat\Desktop\Antonis\competition\train\Scenario-A\Brazilian\track_00001\lr-001_c.png"
        if os.path.exists(test_path):
            image = Image.open(test_path).convert("RGB")
            print(f"Test image loaded: {image.size}")
            
            encoding = processor(images=image, text="TEST123", return_tensors="pt")
            print(f"✓ Processor works!")
            print(f"  Pixel values shape: {encoding.pixel_values.shape}")
        else:
            print("Test image not found, using dummy")
            dummy_image = Image.new('RGB', (128, 64), (255, 255, 255))
            encoding = processor(images=dummy_image, text="TEST123", return_tensors="pt")
            print(f"✓ Processor works on dummy!")
            print(f"  Pixel values shape: {encoding.pixel_values.shape}")
            
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    try:
        main_training()
    except Exception as e:
        print(f"\nMain training failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nTrying simplified approach...")
        
        try:
            processor, model, device = simplified_training()
            
            print("\nSimplified model loaded successfully!")
            print("You can now proceed with training using this model.")
            
        except Exception as e2:
            print(f"\nSimplified approach also failed: {e2}")
            import traceback
            traceback.print_exc()
            print("\nPlease check your installation and try again.")
