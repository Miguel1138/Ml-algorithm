from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from extrai_dados import x, y

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

modelos = {
    "Regressão logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
}

for nome, modelo in modelos.items():
    scores = cross_val_score(modelo, x, y, cv=cv, scoring='roc_auc')
    print(f"{nome}: {scores.mean():.3f} ± {scores.std():.3f}")