from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay

from treina_modelo import log_reg, X_test_s, y_test

y_pred = log_reg.predict(X_test_s)
y_prob = log_reg.predict_proba(X_test_s)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)