import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from extrai_dados import x, y
import matplotlib.pyplot as plt

torch.manual_seed(42)

X_np = x.astype("float32").values
y_np = y.values.astype("float32")

# 70% treino, 15% validação, 15% teste
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_np, y_np, test_size=0.30, stratify=y_np, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

scaler = StandardScaler().fit(X_tr)  # só no treino
X_tr, X_val, X_te = [torch.tensor(scaler.transform(a), dtype=torch.float32) for a in (X_tr, X_val, X_te)]
y_tr, y_val, y_te = [torch.tensor(a) for a in (y_tr, y_val, y_te)]

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

n_features = X_tr.shape[1]

model = nn.Sequential(
    nn.Linear(n_features, 32), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 1),  # 1 saída: logit da classe "sobreviveu"
)
loss_fn = nn.BCEWithLogitsLoss()  # já inclui a sigmoide
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

best_val, paciencia, sem_melhora = float("inf"), 15, 0
hist = {"treino": [], "val": []}

for epoca in range(200):
    model.train()
    perdas = []
    for xb, yb in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(xb).squeeze(1), yb)
        loss.backward()
        opt.step()
        perdas.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val).squeeze(1), y_val).item()
    hist["treino"].append(sum(perdas) / len(perdas))
    hist["val"].append(val_loss)

    if val_loss < best_val:
        best_val, sem_melhora = val_loss, 0
        melhor_estado = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        sem_melhora += 1
        if sem_melhora >= paciencia:
            print("Early stopping na época", epoca)
            break

model.load_state_dict(melhor_estado)

plt.plot(hist["treino"], label="treino")
plt.plot(hist["val"], label="validação")
plt.xlabel("época");
plt.ylabel("loss");
plt.legend()

model.eval()
with torch.no_grad():
    prob = torch.sigmoid(model(X_te).squeeze(1)).numpy()
pred = (prob >= 0.5).astype(int)

print(classification_report(y_te.numpy(), pred))
print("ROC-AUC:", roc_auc_score(y_te.numpy(), prob))
