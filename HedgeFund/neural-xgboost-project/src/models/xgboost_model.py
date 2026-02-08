class XGBoostModel:
    def __init__(self, params=None):
        import xgboost as xgb
        self.params = params if params else {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.1,
            'max_depth': 6,
            'alpha': 10,
            'n_estimators': 100
        }
        self.model = xgb.XGBRegressor(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import mean_squared_error
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse