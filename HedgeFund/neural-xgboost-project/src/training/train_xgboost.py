import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.utils.device import get_device

class XGBoostModel:
    def __init__(self, params=None):
        self.params = params if params is not None else {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.1,
            'max_depth': 6,
            'alpha': 10,
            'n_estimators': 100
        }
        self.model = None

    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        self.model = xgb.train(self.params, dtrain, num_boost_round=self.params['n_estimators'], 
                                evals=[(dval, 'validation')], early_stopping_rounds=10)

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        predictions = [1 if pred > 0.5 else 0 for pred in predictions]
        return accuracy_score(y, predictions)

def main():
    # Load your dataset
    data = pd.read_csv('path/to/your/dataset.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    # Initialize the model
    xgb_model = XGBoostModel()

    # Train the model
    xgb_model.train(X, y)

    # Evaluate the model
    accuracy = xgb_model.evaluate(X, y)
    print(f'Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    main()