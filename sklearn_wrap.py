import torch
from skorch import NeuralNet

from single import SingleS4Regression, SingleS4Classifier


class S4Regression(NeuralNet):
    def __init__(
        self,
        d_input,
        d_output,
        d_model,
        n_layers,
        dropout,
        transposed,
        s4d,
        max_epochs,
        optimizer,
        lr,
        batch_size,
        criterion,
        # train_split_ratio=0.2,
    ):
        super().__init__(
            SingleS4Regression(
                d_input=d_input,
                d_output=d_output,
                d_model=d_model,
                n_layers=n_layers,
                dropout=dropout,
                transposed=transposed,
                s4d=s4d,
            ),
            max_epochs=max_epochs,
            optimizer=optimizer,
            lr=lr,
            # train_split=train_split_ratio,
            batch_size=batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            criterion=criterion,
        )
        self.module.setup()

    # def fit(self, X, y):
    #     return self.module.fit(X, y)

    # def predict(self, X):
    #     return self.module.predict(X)


class S4Classifier(NeuralNet):
    def __init__(
        self,
        d_input,
        d_output,
        d_model,
        n_layers,
        dropout,
        transposed,
        s4d,
        max_epochs,
        optimizer,
        lr,
        batch_size,
        criterion,
        # train_split_ratio=0.2,
    ):
        super().__init__(
            SingleS4Classifier(
                d_input=d_input,
                d_output=d_output,
                d_model=d_model,
                n_layers=n_layers,
                dropout=dropout,
                transposed=transposed,
                s4d=s4d,
            ),
            max_epochs=max_epochs,
            optimizer=optimizer,
            lr=lr,
            # train_split=train_split_ratio,
            batch_size=batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            criterion=criterion,
        )
        self.module.setup()

    # def fit(self, X, y):
    #     return self.module.fit(X, y)

    # def predict(self, X):
    #     return self.module.predict(X)
