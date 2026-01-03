class Model:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        import joblib

        joblib.dump(self.model, filepath)

    @staticmethod
    def load(filepath):
        import joblib

        return joblib.load(filepath)
