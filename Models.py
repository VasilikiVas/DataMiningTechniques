from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow import random

random.set_seed(5)

def grid_search_svr(trainX, trainY):
    parameters = [
        {"kernel": ["rbf"], "gamma": [1e-1, 1e-2, 1e-3, 1e-4], "C": [0.1, 1, 10, 100, 1000]},
        {"kernel": ["linear"], "C": [0.1, 1, 10, 100, 1000]},
    ]

    k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
    model = GridSearchCV(SVR(), parameters, cv=k_fold, scoring='neg_mean_squared_error')
    model.fit(trainX, trainY)

    print(model.best_params_)


def fit_predict_svr(trainX, trainY, testX, testY):
    model = SVR(kernel='rbf', C=10, gamma=0.1)
    model.fit(trainX, trainY)
    predictions = model.predict(testX)
    error = mean_absolute_error(predictions, testY)

    print(f'svr {error}')


def predict_baseline(test_temporalY):
    predictions = test_temporalY[:-1]
    error = mean_absolute_error(predictions, test_temporalY[1:])

    print(f'baseline {error}')


def create_model(lr, hidden_size, nfeatures=13):
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(3, nfeatures)))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=optimizer,metrics=["mae"])
    return model


def grid_search_lstm(trainX, trainY):
    model = KerasClassifier(build_fn=create_model)

    epochs = (50,60,70, 100, 150)
    lrs = (1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4)
    hidden_size = (32, 64, 128, 256, 500, 1000)
    batch_size = (10,30,40,50)
    param_grid = dict(nb_epoch=epochs, lr=lrs, hidden_size=hidden_size, batch_size=batch_size)

    k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=k_fold, scoring='neg_mean_squared_error')
    grid_result = grid.fit(trainX, trainY)

    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")


def fit_predict_lstm(trainX, trainY, testX, testY, batch_size, n_epochs):
    model = create_model(lr=0.005,hidden_size=32)

    model.summary()

    model.fit(trainX, trainY, epochs=n_epochs, batch_size=batch_size)
    predictions = model.predict(testX)
    print(predictions)
    error = mean_absolute_error(predictions, testY)

    print(f'lstm: {error}')


if __name__ == "__main__":
    # Get the train data
    train_dataX = np.genfromtxt('Data/Model Comparison/train_input.csv', delimiter=',', dtype=float)
    train_dataY = np.genfromtxt('Data/Model Comparison/train_output.csv', delimiter=',', dtype=float)
    train_data_temporalX = np.genfromtxt('Data/Model Comparison/train_input_temporal.csv', delimiter=',', dtype=np.float32)
    train_data_temporalY = np.genfromtxt('Data/Model Comparison/train_output_temporal.csv', delimiter=',', dtype=np.float32)

    # Get the test data
    test_dataX = np.genfromtxt('Data/Model Comparison/test_input.csv', delimiter=',', dtype=float)
    test_dataY = np.genfromtxt('Data/Model Comparison/test_output.csv', delimiter=',', dtype=float)
    test_data_temporalX = np.genfromtxt('Data/Model Comparison/test_input_temporal.csv', delimiter=',', dtype=np.float32)
    test_data_temporalY = np.genfromtxt('Data/Model Comparison/test_output_temporal.csv', delimiter=',', dtype=np.float32)

    lengths_train = [38, 29, 40, 43, 37, 39, 44, 31, 33, 42, 35, 45, 43, 38, 37, 33, 34, 46, 26, 38, 35, 33, 34, 38,
                     29, 35, 37]
    lengths_test = [8, 4, 7, 9, 8, 8, 9, 6, 7, 9, 5, 9, 9, 7, 8, 7, 6, 10, 5, 8, 7, 6, 7, 8, 4, 5, 8]
    window_size = 3

    train_input = []
    train_output = []
    test_input = []
    test_output = []

    count = 0
    for i in lengths_train:
        for start_row in range(i - window_size):
            window = train_data_temporalX[start_row + count:start_row + window_size + count]

            train_input.append(list(window))

        train_output += list(train_data_temporalY[count + window_size:count+i])
        count += i

    count = 0
    for i in lengths_test:
        for start_row in range(i - window_size):
            window = test_data_temporalX[start_row + count:start_row + window_size + count]

            test_input.append(list(window))

        test_output += list(test_data_temporalY[count + window_size:count + i])
        count += i

    train_input = np.asarray(train_input)
    train_output = np.asarray(train_output)
    test_input = np.asarray(test_input)
    test_output = np.asarray(test_output)

    print(train_input.shape, train_output.shape, test_input.shape, test_output.shape)

    # grid_search_lstm(train_input, train_output)

    fit_predict_lstm(train_input, train_output, test_input, test_output, batch_size=10, n_epochs=60)

    # Predict with baseline
    predict_baseline(test_data_temporalY)

    # Cross validate the svr model
    # grid_search_svr(train_dataX, train_dataY)

    # Fit svr model and predict the y values for the test data
    fit_predict_svr(train_dataX, train_dataY, test_dataX, test_dataY)
