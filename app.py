from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import re

# ========== Fungsi Preprocessing ==========
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ========== Model Class ==========
class IndoBERTHoaxClassifier(nn.Module):
    def __init__(self, model_name='indobenchmark/indobert-base-p1', num_classes=2, freeze_layers=8):
        super(IndoBERTHoaxClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for i, layer in enumerate(self.bert.encoder.layer):
            for param in layer.parameters():
                param.requires_grad = False if i < freeze_layers else True
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)


# ==========  Kelas Detektor ==========
class HoaxDetector:
    def __init__(self, model_path=None, tokenizer_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path is None:
            base_path = 'D:/ABRAR FILE/CCI/theHack'  # ubah sesuai direktori kamu
            model_path = f'{base_path}/bert_model_state_dict.pth'
            tokenizer_path = f'{base_path}/tokenizer'

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = IndoBERTHoaxClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, judul, isi_berita):
        text = f"{judul} {isi_berita}"
        clean_txt = clean_text(text)

        encoding = self.tokenizer(
            clean_txt,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        prediction = "HOAX" if predicted_class == 1 else "NON-HOAX"

        return {
            "prediction": prediction,
            "confidence": f"{confidence * 100:.2f}%",
            "probabilities": {
                "Non-Hoax": f"{probabilities[0][0]:.3f}",
                "Hoax": f"{probabilities[0][1]:.3f}",
            },
        }


# ========== FastAPI App ==========
app = FastAPI(
    title="Hoax Detector API",
    description="API untuk mendeteksi berita hoax menggunakan IndoBERT",
    version="1.0.0"
)

# Inisialisasi model (load sekali saja saat server mulai)
detector = HoaxDetector()


# ========== Schema untuk input request ==========
class NewsInput(BaseModel):
    judul: str
    isi_berita: str


# ========== Endpoint utama ==========
@app.post("/predict")
def predict_news(input_data: NewsInput):
    """
    Kirimkan JSON:
    {
      "judul": "Judul berita...",
      "isi_berita": "Isi berita..."
    }
    """
    result = detector.predict(input_data.judul, input_data.isi_berita)
    return result


# ========== Endpoint untuk cek server ==========
@app.get("/")
def home():
    return {"message": "✅ Hoax Detector API is running!"}
