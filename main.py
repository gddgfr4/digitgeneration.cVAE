import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------
# モデル定義（Decoderのみ）
# -------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(latent_dim + 16, 128)
        self.fc_out = nn.Linear(128, 28 * 28)

    def forward(self, latent_vector, label):
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        x = torch.cat([latent_vector, label_embedding], dim=1)
        x = F.relu(self.fc_hidden(x))
        x = torch.sigmoid(self.fc_out(x))
        x = x.view(-1, 1, 28, 28)
        return x

# -------------------------
# デバイス設定
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# モデル読み込み
# -------------------------
decoder = Decoder(latent_dim=3)
decoder.load_state_dict(torch.load("cvae.pth", map_location=device))
decoder.to(device)
decoder.eval()

# -------------------------
# Streamlit UI
# -------------------------
st.title("🎨 条件付き画像生成アプリ（cVAE）")

# ラベル選択
digit = st.selectbox("生成したい数字ラベル", list(range(10)))

# 潜在変数スライダー（z ∈ ℝ³）
z_values = []
for i in range(3):
    z_values.append(st.slider(f"z[{i}]", -3.0, 3.0, 0.0, 0.1))

# テンソル変換
z_tensor = torch.tensor(z_values, dtype=torch.float32).unsqueeze(0).to(device)
label_tensor = torch.tensor([digit], dtype=torch.long).to(device)

# 推論
with torch.no_grad():
    generated = decoder(z_tensor, label_tensor)

# 画像表示
image = generated[0][0].cpu().numpy()
st.image(image, width=200, clamp=True, caption=f"生成された画像（{digit}）")
