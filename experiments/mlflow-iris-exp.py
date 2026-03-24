import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-classification")

iris = load_iris()
X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

for n in [10, 50, 100]:
    with mlflow.start_run(run_name=f"rf_n{n}"):
        clf = RandomForestClassifier(n_estimators=n, random_state=42)
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
        mlflow.log_param("n_estimators", n)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")
        print(f"n={n}: accuracy={acc:.3f}")

print("Done — check localhost:5000")
