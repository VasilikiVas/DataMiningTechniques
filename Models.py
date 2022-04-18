# fetch data, make model, train, cross-validation, test, predict
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTM_Model, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


class LSTM_optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        model_path = f'models/LSTM_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features])
                y_batch = y_batch
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features])
                    y_val = y_val
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features])
                y_test = y_test
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.numpy())
                values.append(y_test.numpy())

        return predictions, values

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()


def train_svm_model(X, y):
    model = SVR(kernel='rbf')
    model.fit(X, y)
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=kfold)
    return model, scores


def train_lstm_model(X_train, train_loader, val_loader, test_loader_one):
    input_dim = len(X_train[1])
    output_dim = 1
    hidden_dim = 64
    layer_dim = 3
    batch_size = 64
    dropout = 0.2
    n_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-6

    model = LSTM_Model(input_dim, hidden_dim, layer_dim, output_dim, dropout)

    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    opt = LSTM_optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
    opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
    opt.plot_losses()

    predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)

    return predictions


if __name__ == "__main__":
    # Get the train data
    train_dataX = np.genfromtxt('train_input.csv', delimiter=',')
    train_dataY = np.genfromtxt('train_output.csv', delimiter=',')
    train_dataX = np.delete(train_dataX, [900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926], axis=0)

    # Get the test data
    test_dataX = np.genfromtxt('test_input.csv', delimiter=',')
    test_dataY = np.genfromtxt('test_output.csv', delimiter=',')
    test_dataX = np.delete(test_dataX, [108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134],axis=0)

    # train the svr model
    svm_model, svm_scores = train_svm_model(train_dataX, train_dataY)
    # Predict the y values for the test data
    predictions = svm_model.predict(test_dataX)
    # Calculate the accuracy
    accuracy = accuracy_score(test_dataY.astype(int), predictions.astype(int))
    print(accuracy)

    # put data into tensors
    train_features = torch.Tensor(train_dataX)
    train_targets = torch.Tensor(train_dataY)
    test_features = torch.Tensor(test_dataX)
    test_targets = torch.Tensor(test_dataY)

    # make tensor datasets
    train = TensorDataset(train_features, train_targets)
    test = TensorDataset(test_features, test_targets)

    # Split the train data into train and val
    train, val = torch.utils.data.random_split(train, [800,100])

    # make dataloaders
    train_loader = DataLoader(train, batch_size=64, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=64, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

    # train lstm
    lstm_predictions = train_lstm_model(train_dataX, train_loader, val_loader, test_loader_one)
    print(lstm_predictions)


