from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from extrai_dados import x, y

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

# Escalar: ajusta somente no treino
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_s, y_train)


print("Acurácia treino:", log_reg.score(X_train_s, y_train))
print("Acurácia teste: ", log_reg.score(X_test_s, y_test))