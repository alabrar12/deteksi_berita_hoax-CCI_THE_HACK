import streamlit as st
import torch
from transformers import AutoTokenizer
import re


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)      
    text = re.sub(r"\s+", " ", text).strip()        
    return text


import torch.nn as nn
from transformers import AutoModel

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


class HoaxDetector:
    def __init__(self, model_path=None, tokenizer_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path is None:
            base_path = 'D:/Semester 4/CCI/The Hack 2025/deteksi_berita_hoax-CCI_THE_HACK' #Ubah ke dir tempat menyimpan model
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
            "confidence": f"{confidence*100:.2f}%",
            "probabilities": {
                "Non-Hoax": f"{probabilities[0][0]:.3f}",
                "Hoax": f"{probabilities[0][1]:.3f}",
            },
        }

st.set_page_config(page_title="Deteksi Hoax Berita", layout="centered")

st.title("📰 Deteksi Hoax Berita Menggunakan IndoBERT")
st.write("Masukkan judul dan isi berita untuk memprediksi apakah berita **HOAX** atau **NON-HOAX**.")

# Load model (cache supaya tidak re-load setiap kali)
@st.cache_resource
def load_detector():
    return HoaxDetector()

detector = load_detector()

# Form input
judul = st.text_input("Judul Berita", placeholder="Contoh: Kemenkes Wajibkan Penumpang Pesawat Tervaksinasi TBC")
isi = st.text_area("Isi Berita", height=200, placeholder="Masukkan isi berita di sini...")

if st.button("Prediksi"):
    if judul.strip() == "" or isi.strip() == "":
        st.warning("⚠️ Mohon masukkan judul dan isi berita.")
    else:
        with st.spinner("⏳ Sedang memproses..."):
            result = detector.predict(judul, isi)
        st.success("✅ Prediksi Selesai!")
        st.markdown(f"**Hasil Prediksi:** {result['prediction']}")
        st.markdown(f"**Confidence:** {result['confidence']}")

        st.subheader("Probabilitas Kelas:")
        st.write(f"🟩 Non-Hoax: {result['probabilities']['Non-Hoax']}")
        st.write(f"🟥 Hoax: {result['probabilities']['Hoax']}")
