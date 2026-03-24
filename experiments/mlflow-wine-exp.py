import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("diabetes-regression")

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

runs = [
    ("ridge_alpha_0.1",  Ridge(alpha=0.1)),
    ("ridge_alpha_1.0",  Ridge(alpha=1.0)),
    ("ridge_alpha_10.0", Ridge(alpha=10.0)),
    ("rf_50_trees",      RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)),
    ("rf_100_trees",     RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)),
    ("rf_200_deep",      RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)),
    ("gbm_lr_0.05",      GradientBoostingRegressor(learning_rate=0.05, n_estimators=100, random_state=42)),
    ("gbm_lr_0.1",       GradientBoostingRegressor(learning_rate=0.1,  n_estimators=100, random_state=42)),
    ("gbm_lr_0.2",       GradientBoostingRegressor(learning_rate=0.2,  n_estimators=200, random_state=42)),
]

for name, model in runs:
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_params(model.get_params())
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        r2   = r2_score(y_te, preds)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2",   r2)
        mlflow.sklearn.log_model(model, "model")
        print(f"{name:<22} RMSE={rmse:.1f}  R2={r2:.3f}")

print("\nDone — check localhost:5000")
