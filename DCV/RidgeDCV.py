#%%
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import optuna

def objective(trial):
    # parameters for Ridge regression
    alpha = trial.suggest_float("alpha", 1e-4, 1e4)
    # R2 score by cross validation
    model = Ridge(alpha=alpha)
    r2_score = cross_validate(model, X, y, cv=5, scoring='r2')['test_score'].mean()
    
    return r2_score

# Reading data
housing = fetch_california_housing()
X = housing['data']
y = housing['target']

# Instantiate KFold
kf = KFold(n_splits=2, shuffle=True, random_state=1)
y_true = []
y_pred = []

for i, (train_index, test_index) in enumerate(kf.split(X)):
    # Split data
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    # Optimize hyperparameters
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    # Optimized model
    optimized_model = Ridge(**study.best_params)
    optimized_model.fit(X_train, y_train)
    # Evaluate model
    predicted_y = optimized_model.predict(X_test)
    y_true_temp = [y_test[i] for i in range(len(y_test))]
    y_pred_temp = [predicted_y[i] for i in range(len(predicted_y))]
    y_true += y_true_temp
    y_pred += y_pred_temp

r2 = r2_score(y_true, y_pred)

# %%
