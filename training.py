import os
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer, AutoModelForMaskedLM
from PIL import Image
import torch
from dataclasses import dataclass

# Configuration classes
@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE: int = 16
    EPOCHS: int = 15
    LEARNING_RATE: float = 0.00005

@dataclass(frozen=True)
class DatasetConfig:
    DATA_ROOT: Path = Path(os.getenv("DATA_ROOT", "dataset"))

@dataclass(frozen=True)
class ModelConfig:
    TROCR_MODEL_NAME: str = 'microsoft/trocr-small-printed'
    BERT_MODEL_PATH: Path = Path(
        os.getenv("BERT_MODEL_PATH", "pretrained_model/bert_module_khmer")
    )

# Custom OCR Dataset
class CustomOCRDataset(Dataset):
    def __init__(self, root_dir: Path, df, processor):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir / self.df.iloc[idx, 0]
        image = Image.open(img_name).convert("RGB")
        text = str(self.df.iloc[idx, 1])

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.squeeze()

        return {"pixel_values": pixel_values, "labels": labels}

# Load the TrOCR model and processor
def load_trocr_model(model_name: str):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model

# Load the Khmer BERT model and tokenizer
def load_bert_model(bert_model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    model = AutoModelForMaskedLM.from_pretrained(bert_model_path)
    return tokenizer, model

# Recognize text from image using TrOCR
def recognize_text_from_image(image_path: str, processor, model):
    image = Image.open(image_path)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    outputs = model.generate(pixel_values.to('cuda') if torch.cuda.is_available() else pixel_values)
    predicted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return predicted_text

if __name__ == "__main__":
    # Load train and test DataFrames (ensure these are properly loaded and have 'image_path' and 'text' columns)
    train_df = pd.read_csv(DatasetConfig.DATA_ROOT / "train_labels.csv")
    test_df = pd.read_csv(DatasetConfig.DATA_ROOT / "test_labels.csv")

    # Load the TrOCR processor and model
    processor, model = load_trocr_model(ModelConfig.TROCR_MODEL_NAME)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Print model and parameter information
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Create training and validation datasets
    train_dataset = CustomOCRDataset(
        root_dir=DatasetConfig.DATA_ROOT / "data_train",
        df=train_df,
        processor=processor
    )

    valid_dataset = CustomOCRDataset(
        root_dir=DatasetConfig.DATA_ROOT / "data_test",
        df=test_df,
        processor=processor
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=TrainingConfig.BATCH_SIZE)

    # Set up the training environment
    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LEARNING_RATE)
    model.train()

    # Training loop
    for epoch in range(TrainingConfig.EPOCHS):
        epoch_loss = 0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{TrainingConfig.EPOCHS}, Loss: {epoch_loss/len(train_loader)}")

    print("Training complete.")

    # Optionally, process recognized text with Khmer BERT model
    # Load the BERT model and tokenizer
    bert_tokenizer, bert_model = load_bert_model(str(ModelConfig.BERT_MODEL_PATH))

    # Example of recognizing text from an image and processing it with BERT
    example_image_path = "path/to/your/example_image.png"  # Replace with your image path
    recognized_text = recognize_text_from_image(example_image_path, processor, model)
    print("Recognized text:", recognized_text)

    bert_inputs = bert_tokenizer(recognized_text, return_tensors="pt")
    bert_outputs = bert_model(**bert_inputs)
    print("BERT model outputs:", bert_outputs)
