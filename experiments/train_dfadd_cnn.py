import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan donanım birimi: {device}")

# 1. Bellek Dostu Veri Yükleyici (Custom PyTorch Dataset)
class DFADD_Dataset(Dataset):
    def __init__(self, x_path, y_path, indices):
        # mmap_mode='r' veriyi RAM'e almaz, sadece diskteki adresini tutar
        self.X = np.load(x_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x_val = self.X[real_idx].astype(np.float32)
        y_val = self.y[real_idx].astype(np.float32)
        
        # --- EKLENEN NORMALİZASYON KODU ---
        # Değerleri 0 ile 1 arasına ölçeklendiriyoruz (Min-Max Scaling)
        x_min = x_val.min()
        x_max = x_val.max()
        if x_max - x_min > 1e-8: # Sıfıra bölünme hatasını önlemek için
            x_val = (x_val - x_min) / (x_max - x_min)
        
        # PyTorch için Channel boyutunu ekle (1, 128, 216)
        x_val = np.expand_dims(x_val, axis=0)
        
        return torch.tensor(x_val), torch.tensor([y_val])

# 2. İndeksleri Böl (%80 Eğitim, %20 Test)
total_samples = 200465 # Terminalden aldığın sayı
all_indices = np.arange(total_samples)
# Verinin kendisini değil, sadece indeks numaralarını bölüyoruz (RAM tasarrufu)
train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

print(f"Eğitim için ayrılan: {len(train_indices)} | Test için ayrılan: {len(test_indices)}")

# DataLoader'ları oluştur
BATCH_SIZE = 64 # Veri seti çok büyük olduğu için paketi 64'e çıkardık
train_dataset = DFADD_Dataset("X_train_dfadd_mel.npy", "y_train_dfadd_mel.npy", train_indices)
test_dataset = DFADD_Dataset("X_train_dfadd_mel.npy", "y_train_dfadd_mel.npy", test_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. CNN Mimarisini Tanımla

class DeepVoiceCNN(nn.Module):
    def __init__(self):
        super(DeepVoiceCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1, self.pool1 = nn.ReLU(), nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2, self.pool2 = nn.ReLU(), nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3, self.pool3 = nn.ReLU(), nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 27, 128)
        self.relu4, self.dropout = nn.ReLU(), nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.dropout(self.relu4(self.fc1(self.flatten(x))))
        return self.sigmoid(self.fc2(x))

model = DeepVoiceCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Eğitim Döngüsü
EPOCHS = 3 # Veri seti 4 kat büyük olduğu için epoch sayısını düşük tutuyoruz
print("\nDFADD veri seti ile model eğitimi başlıyor...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i+1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
    print(f"--- Epoch {epoch+1} Tamamlandı. Ortalama Loss: {running_loss/len(train_loader):.4f} ---")

# 5. Test ve Değerlendirme
print("\nDFADD test seti üzerinde değerlendirme yapılıyor...")
model.eval()
all_preds, all_probs, all_labels = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        probs = model(inputs).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend((probs > 0.5).astype(int))
        all_labels.extend(labels.cpu().numpy())

all_labels = np.array(all_labels).flatten()
all_preds = np.array(all_preds).flatten()
all_probs = np.array(all_probs).flatten()

accuracy = accuracy_score(all_labels, all_preds)
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]

print("\n" + "="*40)
print(f"DFADD VERİ SETİ - CNN SONUÇLARI")
print("="*40)
print(f"Accuracy (Doğruluk):   {accuracy:.4f}")
print(f"EER (Equal Error Rate):{eer:.4f}")
print("="*40)

# Yeni modeli kaydet
torch.save(model.state_dict(), "deepvoice_dfadd_cnn_model.pth")
print("\nYeni model 'deepvoice_dfadd_cnn_model.pth' olarak kaydedildi.")