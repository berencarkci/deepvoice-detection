"""
DeepVoice Tespit Sistemi — Streamlit GUI (ASVspoof 2019 LA modelleriyle)

Çalıştırma:
    cd <proje-dizini>
    source .venv/bin/activate
    pip install -r requirements_gui.txt      # ilk sefer (streamlit + soundfile)
    streamlit run app.py

Modeller `models/` klasöründen okunur. CNN normalizasyon mean/std değerleri,
her modelin yanındaki *.norm.json dosyasından alınır.

Etiket: 1 = Gerçek (bonafide), 0 = Sahte (spoof/deepvoice).
"""

import os
import json
import tempfile

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
import streamlit as st

# Model klasörü — bu dosyanın yanındaki models/ (çalışma dizininden bağımsız)
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


# ==========================================
# 1. SAYFA YAPILANDIRMASI VE CSS
# ==========================================
st.set_page_config(
    page_title="Deepvoice Tespit Sistemi",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; font-weight: bold; background-color: #4CAF50; color: white;}
    .fake-result {color: #d32f2f; font-size: 24px; font-weight: bold; text-align: center; padding: 20px; background-color: #ffcdd2; border-radius: 10px;}
    .real-result {color: #388e3c; font-size: 24px; font-weight: bold; text-align: center; padding: 20px; background-color: #c8e6c9; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. PYTORCH CNN MİMARİSİ
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + x)


class DeepVoiceCNN(nn.Module):
    def __init__(self, dropout_rate: float = 0.4):
        super().__init__()
        def conv_bn_lrelu(in_ch, out_ch, kernel=3, stride=1, pad=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )
        self.stem = nn.Sequential(conv_bn_lrelu(1, 32), nn.MaxPool2d(2, 2))
        self.stage1 = nn.Sequential(conv_bn_lrelu(32, 64), ResBlock(64), nn.MaxPool2d(2, 2), nn.Dropout2d(dropout_rate))
        self.stage2 = nn.Sequential(conv_bn_lrelu(64, 128), ResBlock(128), nn.MaxPool2d(2, 2), nn.Dropout2d(dropout_rate))
        self.stage3 = nn.Sequential(conv_bn_lrelu(128, 256), ResBlock(256), nn.AdaptiveAvgPool2d((4, 9)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 9, 256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(dropout_rate + 0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)


# --- MODEL YÜKLEYİCİLER (CACHE'Lİ) ---
@st.cache_resource
def load_pytorch_model(model_path):
    model = DeepVoiceCNN(dropout_rate=0.4)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_resource
def load_ml_model(model_path):
    return joblib.load(model_path)


@st.cache_resource
def load_norm_stats(model_path):
    """<model>.norm.json dosyasından eğitim mean/std değerlerini okur."""
    sidecar = os.path.splitext(model_path)[0] + ".norm.json"
    with open(sidecar, "r", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["mean"]), float(d["std"])


# ==========================================
# 3. TAHMİN (INFERENCE) FONKSİYONU
# ==========================================
def predict_audio(file_path, model_choice):
    # Ses okuma parametreleri (22.05 kHz, 5 saniye)
    SR = 22050
    DURATION = 5.0
    y, sr = librosa.load(file_path, sr=SR, duration=DURATION)

    # ----------------------------------------
    # A) GELENEKSEL ML (194 Öznitelik)
    # ----------------------------------------
    if model_choice in ["Random Forest", "Support Vector Machines (SVM)"]:
        features = []

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        features.extend(np.min(mfccs, axis=1))
        features.extend(np.max(mfccs, axis=1))

        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])

        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.extend([np.mean(cent), np.std(cent)])

        band = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.extend([np.mean(band), np.std(band)])

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.extend([np.mean(rolloff), np.std(rolloff)])

        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        X_input = np.array(features, dtype=np.float32).reshape(1, -1)  # (1, 194)

        if model_choice == "Random Forest":
            scaler = load_ml_model(os.path.join(MODELS_DIR, "asvspoof_random_forest_scaler.pkl"))
            model = load_ml_model(os.path.join(MODELS_DIR, "asvspoof_random_forest_model.pkl"))
        else:
            scaler = load_ml_model(os.path.join(MODELS_DIR, "asvspoof_svm_scaler.pkl"))
            model = load_ml_model(os.path.join(MODELS_DIR, "asvspoof_svm_model.pkl"))

        X_scaled = scaler.transform(X_input)

        # classes_=[0,1] → kolon 0 = P(Fake). is_fake = P(fake) > 0.5
        prob_fake = float(model.predict_proba(X_scaled)[0][0])
        is_fake = prob_fake > 0.5
        return is_fake, prob_fake

    # ----------------------------------------
    # B) CNN — Mel-Spektrogram
    # ----------------------------------------
    elif model_choice == "CNN (Mel-Spektrogram)":
        HOP_LENGTH = 512
        N_MELS = 128
        MAX_PAD_LEN = int((SR * DURATION) / HOP_LENGTH) + 1  # 216

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if mel_spec_db.shape[1] > MAX_PAD_LEN:
            mel_spec_db = mel_spec_db[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode="constant")

        MODEL_PATH = os.path.join(MODELS_DIR, "asvspoof_cnn_mel_final.pth")
        mean, std = load_norm_stats(MODEL_PATH)            # sidecar'dan global mean/std
        S_norm = (mel_spec_db - mean) / (std + 1e-8)
        input_tensor = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        model = load_pytorch_model(MODEL_PATH)
        with torch.no_grad():
            logits = model(input_tensor)
            prob_real = torch.sigmoid(logits).item()       # sigmoid = P(Real)
            prob_fake = 1.0 - prob_real
        is_fake = prob_fake > 0.5
        return is_fake, prob_fake

    # ----------------------------------------
    # C) CNN — Standart Spektrogram
    # ----------------------------------------
    elif model_choice == "CNN (Standart Spektrogram)":
        HOP_LENGTH = 512
        N_FFT = 512  # 257 frekans bandı
        MAX_PAD_LEN = int((SR * DURATION) / HOP_LENGTH) + 1  # 216

        stft_matrix = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        spec_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)

        if spec_db.shape[1] > MAX_PAD_LEN:
            spec_db = spec_db[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - spec_db.shape[1]
            spec_db = np.pad(spec_db, pad_width=((0, 0), (0, pad_width)), mode="constant")

        MODEL_PATH = os.path.join(MODELS_DIR, "asvspoof_cnn_spec_final.pth")
        mean, std = load_norm_stats(MODEL_PATH)
        D_norm = (spec_db - mean) / (std + 1e-8)
        input_tensor = torch.tensor(D_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        model = load_pytorch_model(MODEL_PATH)
        with torch.no_grad():
            logits = model(input_tensor)
            prob_real = torch.sigmoid(logits).item()
            prob_fake = 1.0 - prob_real
        is_fake = prob_fake > 0.5
        return is_fake, prob_fake


# ==========================================
# 4. GÖRSELLEŞTİRME (XAI)
# ==========================================
@st.cache_data
def plot_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=5.0)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    librosa.display.waveshow(y, sr=sr, ax=ax[0], color="blue", alpha=0.5)
    ax[0].set_title("Ses Dalga Formu (Waveform)")
    ax[0].set_ylabel("Genlik")

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, hop_length=512, ax=ax[1], fmax=8000)
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    ax[1].set_title("Mel-Spektrogram Analizi")

    plt.tight_layout()
    return fig


# ==========================================
# 5. UI: YAN MENÜ VE SEKMELER
# ==========================================
try:
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/8603/8603816.png", width=100)
except Exception:
    st.sidebar.markdown("# 🎙️")
st.sidebar.title("Kontrol Paneli")
st.sidebar.info(
    "Proje: Eğiticili Makine Öğrenmesi Yöntemleri ile Deepvoice Tahmini\n\n"
    "Veri Seti: ASVspoof 2019 LA"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Seçimi")
selected_model = st.sidebar.selectbox(
    "Tahmin için kullanılacak modeli seçin:",
    ("Random Forest", "Support Vector Machines (SVM)", "CNN (Mel-Spektrogram)", "CNN (Standart Spektrogram)"),
)

tab1, tab2, tab3 = st.tabs(["🎙️ Canlı Tespit Sistemi", "📊 Performans ve Metrikler", "🧠 Nasıl Çalışır?"])

with tab1:
    st.header("Deepvoice Tespit Sistemi")
    st.write("Şüpheli bir ses kaydını yükleyin; sistem çıkarılan öznitelikleri analiz ederek sentetik (Deepvoice) mi yoksa gerçek mi olduğunu tespit etsin.")

    uploaded_file = st.file_uploader("Bir ses dosyası yükleyin (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Dinle")
            st.audio(uploaded_file)
        with col2:
            st.subheader("Analiz")
            analyze_button = st.button("🔍 Ses Dosyasını Analiz Et")

        if analyze_button:
            with st.spinner(f"{selected_model} modeli ile öznitelikler çıkarılıyor ve analiz ediliyor..."):
                # Dosya formatını koruyarak geçici kayıt (decode/gürültü sorununu önler)
                file_extension = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    is_fake, fake_prob = predict_audio(tmp_file_path, selected_model)
                    st.markdown("---")
                    if is_fake:
                        st.markdown(
                            f'<div class="fake-result">🚨 TESPİT EDİLDİ: SENTETİK SES (DEEPVOICE) <br>'
                            f'<span style="font-size: 18px;">Güven Skoru: %{fake_prob*100:.2f}</span></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        real_prob = 1 - fake_prob
                        st.markdown(
                            f'<div class="real-result">✅ ONAYLANDI: GERÇEK İNSAN SESİ <br>'
                            f'<span style="font-size: 18px;">Güven Skoru: %{real_prob*100:.2f}</span></div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("### 📈 Sinyal Görselleştirmesi (XAI)")
                    fig = plot_audio_features(tmp_file_path)
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Tahmin sırasında bir hata oluştu: {e}")
                finally:
                    os.unlink(tmp_file_path)

with tab2:
    st.header("📊 Model Performans Raporu")
    st.write("ASVspoof 2019 LA veri setinde **saldırı-gruplu 5 katlı çapraz doğrulama** "
             "(her katta görülmemiş saldırı türleriyle test) ile elde edilen sonuçlar.")
    st.markdown("""
    | Yöntem | EER | Dengeli Doğruluk (BalACC) | AUC |
    |---|---|---|---|
    | Random Forest | 0.1708 ± 0.0671 | 0.6641 | 0.9066 |
    | SVM | 0.1374 ± 0.0549 | 0.8500 | 0.9355 |
    | **CNN (Mel)** | **0.0370 ± 0.0407** | **0.9566** | **0.9904** |
    | CNN (Spec) | 0.0954 ± 0.1022 | 0.9002 | 0.9424 |
    """)
    st.caption("EER düşük = daha iyi; diğer metrikler yüksek = daha iyi. Değerler 5 katın "
               "ortalaması ± standart sapmadır. Sınıf dengesizliği (~1:9) nedeniyle dengeli "
               "doğruluk ve AUC öne çıkarılır. En iyi model: CNN (Mel).")
    st.info("Detaylı tablo ve grafikler: results/asvspoof_results.md / results/asvspoof_results.csv")

with tab3:
    st.header("🧠 Sistem Nasıl Çalışır?")
    st.markdown("""
    **Siber güvenlik ve bilgi güvenilirliği** için geliştirilen bu sistem ses sinyallerini inceler.
    - **ML Modelleri (RF, SVM):** Sesten 194 spektral istatistik (MFCC, ZCR, RMS, Chroma vb.) çıkarılır, StandardScaler ile ölçeklenip sınıflandırılır.
    - **Derin Öğrenme (CNN):** Mel/Standart Spektrogram çıkarılıp 216 zaman dilimine sabitlenir; Residual block içeren CNN, sentetik sesin bıraktığı **artefaktları** yakalar.
    - **Modeller:** ASVspoof 2019 LA veri setiyle eğitildi; CNN normalizasyon istatistikleri model yanındaki `.norm.json` dosyasından okunur.
    """)
