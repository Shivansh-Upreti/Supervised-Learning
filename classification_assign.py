import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

df=pd.read_csv("waterQuality1.csv")
df=df[df["is_safe"]!="#NUM!"]
df["is_safe"]=df["is_safe"].astype(int)

X=df.drop(columns=["is_safe"])
y=df["is_safe"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

print("# Logistic Regression #")
lr=LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr=lr.predict(X_test_scaled)
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred_lr))
print("Classification Report:")
print(classification_report(y_test,y_pred_lr))
print("\n\n")

print("# K-Nearest Neighbors #")
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled,y_train)
y_pred_knn=knn.predict(X_test_scaled)
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred_knn))
print("Classification Report:")
print(classification_report(y_test,y_pred_knn))
print("\n\n")

print("# Random Forest #")
rf=RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled,y_train)
y_pred_rf=rf.predict(X_test_scaled)
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred_rf))
print("Classification Report:")
print(classification_report(y_test,y_pred_rf))
print("\n\n")

print("# Decision Tree #")
dt=DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled,y_train)
y_pred_dt=dt.predict(X_test_scaled)
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred_dt))
print("Classification Report:")
print(classification_report(y_test,y_pred_dt))