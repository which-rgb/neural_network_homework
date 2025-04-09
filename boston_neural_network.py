import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 配置设备和随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ----------------------
# 数据准备与特征选择
# ----------------------
# 加载数据（保持原始顺序）
data = pd.read_excel("BostonHousingData.xlsx", sheet_name="Sheet1")

# 特征选择（阈值调整为0.5）
correlation_matrix = data.corr()
medv_correlation = correlation_matrix['MEDV'].abs().sort_values(ascending=False)
selected_features = medv_correlation[medv_correlation > 0.5].index.tolist()
selected_features.remove('MEDV')
print(f"Selected features: {selected_features}")

# 数据集分割（前450训练，后56测试）
X = data[selected_features].values
y = data['MEDV'].values.reshape(-1, 1)

X_train, X_test = X[:450], X[450:]
y_train, y_test = y[:450], y[450:]

# ----------------------
# 数据预处理
# ----------------------
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# 仅用训练集拟合scaler
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

# 目标变量标准化（保持二维结构）
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

# 划分验证集
X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
    X_train_scaled, y_train_scaled,
    test_size=0.1,
    random_state=42
)

# 转换为Tensor
X_train_t = torch.FloatTensor(X_train_sub).to(device)
y_train_t = torch.FloatTensor(y_train_sub).to(device)
X_val_t = torch.FloatTensor(X_val_sub).to(device)
y_val_t = torch.FloatTensor(y_val_sub).to(device)
X_test_t = torch.FloatTensor(X_test_scaled).to(device)


# ----------------------
# 神经网络模型
# ----------------------
class BostonModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # 新增正则化
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


model = BostonModel(input_size=len(selected_features)).to(device)

# ----------------------
# 训练配置
# ----------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 新增L2正则化
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
early_stop_patience = 15
best_loss = np.inf
patience_counter = 0

# ----------------------
# 训练循环
# ----------------------
train_losses = []
val_losses = []

for epoch in range(500):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss = criterion(val_preds, y_val_t)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    scheduler.step(val_loss)

    # 早停机制
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1

    if patience_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# ----------------------
# 最终测试评估
# ----------------------
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_t).cpu().numpy()

# 反标准化（保持二维结构）
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_orig = y_test  # 使用原始值

# 计算指标
mse = mean_squared_error(y_test_orig, y_pred)
mae = mean_absolute_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)
max_error = np.max(np.abs(y_test_orig - y_pred))

print("\n=== Final Evaluation ===")
print(f"Test MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"MAE: ${mae:.2f}k")  # 修正单位显示
print(f"Max Error: ${max_error:.2f}k")

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()