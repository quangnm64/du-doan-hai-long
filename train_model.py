
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("PHASE 1: BUSINESS UNDERSTANDING")
print("=" * 60)
print("Mục tiêu: Dự đoán mức độ hài lòng của khách hàng")
print("Bài toán: Classification (Hài lòng / Không hài lòng)")
print()

print("PHASE 2: DATA UNDERSTANDING")
print("=" * 60)


df = None

if df is None:
    print("   Đang tạo dữ liệu mẫu để demo...")
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(18, 80, n_samples),
        'Flight_Distance': np.random.randint(100, 5000, n_samples),
        'Inflight_wifi_service': np.random.randint(0, 6, n_samples),
        'Departure_Arrival_time_convenient': np.random.randint(0, 6, n_samples),
        'Ease_of_Online_booking': np.random.randint(0, 6, n_samples),
        'Gate_location': np.random.randint(0, 6, n_samples),
        'Food_and_drink': np.random.randint(0, 6, n_samples),
        'Online_boarding': np.random.randint(0, 6, n_samples),
        'Seat_comfort': np.random.randint(0, 6, n_samples),
        'Inflight_entertainment': np.random.randint(0, 6, n_samples),
        'On_board_service': np.random.randint(0, 6, n_samples),
        'Leg_room_service': np.random.randint(0, 6, n_samples),
        'Baggage_handling': np.random.randint(0, 6, n_samples),
        'Checkin_service': np.random.randint(0, 6, n_samples),
        'Inflight_service': np.random.randint(0, 6, n_samples),
        'Cleanliness': np.random.randint(0, 6, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Customer_Type': np.random.choice(['Loyal Customer', 'disloyal Customer'], n_samples),
        'Type_of_Travel': np.random.choice(['Business travel', 'Personal Travel'], n_samples),
        'Class': np.random.choice(['Business', 'Eco', 'Eco Plus'], n_samples, p=[0.2, 0.6, 0.2]),
    }
    
    df = pd.DataFrame(data)
    
    service_scores = df[['Inflight_wifi_service', 'Food_and_drink', 'Seat_comfort', 
                         'Inflight_entertainment', 'On_board_service', 'Inflight_service']].sum(axis=1)
    df['satisfaction'] = ((service_scores > 20) & 
                          (df['Customer_Type'] == 'Loyal Customer') & 
                          (df['Class'].isin(['Business', 'Eco Plus']))).astype(int)
    
    noise = np.random.random(n_samples) < 0.15
    df.loc[noise, 'satisfaction'] = 1 - df.loc[noise, 'satisfaction']

print(f"Số lượng mẫu: {len(df)}")
print(f"Số lượng features: {len(df.columns) - 1}")
print(f"Phân bố target:")
print(df['satisfaction'].value_counts())
print(f"\nThông tin dataset:")
print(df.info())
print(f"\nThống kê mô tả:")
print(df.describe())

print("PHASE 3: DATA PREPARATION")
print("=" * 60)

X = df.drop('satisfaction', axis=1)
y = df['satisfaction']

X_encoded = pd.get_dummies(X, columns=['Gender', 'Customer_Type', 'Type_of_Travel', 'Class'], 
                          drop_first=True)

feature_names = X_encoded.columns.tolist()

print(f"\nMissing values: {X_encoded.isnull().sum().sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Đã chuẩn hóa dữ liệu")

print("PHASE 4: MODELING")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\nĐang huấn luyện các mô hình...")
for name, model in models.items():
    print(f"\n{name}:")
    model.fit(X_train_scaled, y_train)
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {auc:.4f}")

best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

print(f"\n{'='*60}")
print(f"Mô hình tốt nhất: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"{'='*60}")

print("PHASE 5: EVALUATION")
print("=" * 60)

y_pred_best = best_model.predict(X_test_scaled)
y_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png')
print("\nĐã lưu confusion matrix vào static/confusion_matrix.png")

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features quan trọng nhất:")
    print(feature_importance.head(10))
    
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    print("Đã lưu feature importance vào static/feature_importance.png")

print("PHASE 6: DEPLOYMENT PREPARATION")
print("=" * 60)

os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)

joblib.dump(best_model, 'models/customer_satisfaction_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

with open('models/feature_names.json', 'w') as f:
    json.dump(feature_names, f)

print("\nĐã lưu:")
print("  - Model: models/customer_satisfaction_model.pkl")
print("  - Scaler: models/scaler.pkl")
print("  - Feature names: models/feature_names.json")

evaluation_info = {
    'best_model': best_model_name,
    'metrics': {
        'accuracy': float(results[best_model_name]['accuracy']),
        'precision': float(results[best_model_name]['precision']),
        'recall': float(results[best_model_name]['recall']),
        'f1': float(results[best_model_name]['f1']),
        'auc': float(results[best_model_name]['auc'])
    },
    'all_models': {name: {
        'accuracy': float(results[name]['accuracy']),
        'f1': float(results[name]['f1'])
    } for name in results}
}

with open('models/evaluation_info.json', 'w') as f:
    json.dump(evaluation_info, f, indent=2)

print("  - Evaluation info: models/evaluation_info.json")
print("\nHoàn thành huấn luyện mô hình!")
