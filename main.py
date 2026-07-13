import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    f1_score,
)

RANDOM_STATE = 42


iris = load_iris()
X = iris.data          
y = iris.target        
feature_names = iris.feature_names
target_names = iris.target_names

print("=" * 60)
print("STEP 1: INPUT - Dataset Overview")
print("=" * 60)
print(f"Samples: {X.shape[0]}")
print(f"Features ({X.shape[1]}): {feature_names}")
print(f"Classes ({len(target_names)}): {list(target_names)}")
print(f"Class balance: {dict(zip(target_names, np.bincount(y)))}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y, shuffle=True
)

print("\n" + "=" * 60)
print("STEP 2: PROCESS - Train/Test Split")
print("=" * 60)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("STEP 2: PROCESS - Tuning K (Error Rate vs K)")
print("=" * 60)
k_range = range(1, 21)
error_rates = []
for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    pred_temp = knn_temp.predict(X_test_scaled)
    error_rates.append(np.mean(pred_temp != y_test))


search_start = 2  
best_idx = search_start + int(np.argmin(error_rates[search_start:]))
best_k = k_range[best_idx]
print(f"Best K found: {best_k} (lowest error rate = {error_rates[best_idx]:.4f}, K=1 excluded to avoid overfitting)")

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), error_rates, marker="o", linestyle="--", color="steelblue")
plt.axvline(best_k, color="orange", linestyle=":", label=f"Optimal K = {best_k}")
plt.title("Tuning the Engine: Error Rate vs K Value")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("k_tuning_curve.png", dpi=150)
plt.close()


model = KNeighborsClassifier(n_neighbors=best_k)   
model.fit(X_train_scaled, y_train)                 
predictions = model.predict(X_test_scaled)          


acc = accuracy_score(y_test, predictions)
f1_macro = f1_score(y_test, predictions, average="macro")
cm = confusion_matrix(y_test, predictions)

print("\n" + "=" * 60)
print(f"STEP 4: OUTPUT - Model Validation (K={best_k})")
print("=" * 60)
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score (macro avg): {f1_macro:.4f}\n")
print("Confusion Matrix:")
print(cm)
print("\nFull Classification Report:")
print(classification_report(y_test, predictions, target_names=target_names))


fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix (KNN, K={best_k})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  
new_flower_scaled = scaler.transform(new_flower)
new_pred = model.predict(new_flower_scaled)
print("\n" + "=" * 60)
print("STEP 5: Predicting a brand-new sample")
print("=" * 60)
print(f"Input measurements: {new_flower[0]}")
print(f"Predicted species: {target_names[new_pred[0]]}")

print("\nDone. Saved: k_tuning_curve.png, confusion_matrix.png")