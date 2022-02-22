import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


ISRAELI_DATA_PATH = "../csvs/israeli_data.csv"
MACRO_NUTRIENTS = ['protein', 'total_fat', 'carbohydrates']
MICRO_MINERALS = ['calcium', 'iron', 'magnesium', 'phosphorus', 'potassium', 'sodium', 'zinc', 'copper']
MICRO_VITAMINS = ['vitamin_a_iu', 'vitamin_e', 'vitamin_c', 'thiamin', 'riboflavin', 'niacin', 'vitamin_b6',
                  'folate', 'folate_dfe', 'vitamin_b12', 'carotene']
MICRO_NUTRIENTS = MICRO_MINERALS + MICRO_VITAMINS

FDC_DATA_PATH = "../csvs/wide_nutri_records.csv"
FDC_MACRO = ['Protein', 'Total lipid (fat)', 'Carbohydrate, by difference']
FDC_MINERALS = ['Calcium, Ca', 'Iron, Fe', 'Magnesium, Mg', 'Phosphorus, P', 'Potassium, K', 'Sodium, Na', 'Zinc, Zn',
                'Copper, Cu']
FDC_VITAMINS = ['Vitamin A, IU', 'Vitamin E (alpha-tocopherol)', 'Vitamin C, total ascorbic acid', 'Thiamin',
                'Riboflavin', 'Niacin', 'Vitamin B-6', 'Folate, food', 'Folate, DFE', 'Vitamin B-12', 'Carotene, alpha']
FDC_MICRO = FDC_MINERALS + FDC_VITAMINS
ACCURACY_THRESHOLD = 0.01


class MLP(nn.Module):
    """
    The multi layer perceptron network
    The architecture is two hidden layer followed by RelU and dropout, and output layer followed by sigmoid.
    """
    def __init__(self, input_size, output_size):
        """initializing the model"""
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """forward propagation"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def preprocess_data():
    df_israeli = pd.read_csv(ISRAELI_DATA_PATH)
    df_israeli = df_israeli[MACRO_NUTRIENTS + MICRO_NUTRIENTS]
    df_fdc = pd.read_csv(FDC_DATA_PATH, dtype='unicode')
    df_fdc = df_fdc[FDC_MACRO + FDC_MICRO].iloc[2:, :].astype(float)
    df_fdc.columns = MACRO_NUTRIENTS + MICRO_NUTRIENTS
    # df_israeli = df_israeli.dropna(subset=MACRO_NUTRIENTS) # add zero
    df_israeli = df_israeli.dropna(subset=MACRO_NUTRIENTS)
    df_fdc = df_fdc.dropna(subset=MACRO_NUTRIENTS)
    return df_israeli, df_fdc


def preprocess_micronutrients(df, micronutrients):
    df = df.copy()
    df = df.dropna(subset=micronutrients)
    for micro in micronutrients:
        if micro in MICRO_MINERALS or micro in ['vitamin_e', 'vitamin_c', 'thiamin',
                                                'riboflavin', 'niacin', 'vitamin_b6']:
            df[micro] = df[micro] / 1000
        elif micro == 'vitamin_a_iu':
            df[micro] = df[micro] * 0.3 / 1000000
        else:
            df[micro] = df[micro] / 1000000
    return df


def split_data(df_israeli, df_fdc, macros, micros):
    """
    This function splits the data frame into the feature space X and the label space y, and then splits them into
    train and test sets.
    """
    X_israeli = df_israeli[macros].values
    y_israeli = df_israeli[micros].values
    X_train, X_test, y_train, y_test = train_test_split(X_israeli, y_israeli, test_size=0.2)

    X_fdc = df_fdc[macros].values
    y_fdc = df_fdc[micros].values
    X_train = np.concatenate((X_train, X_fdc), axis=0)
    y_train = np.concatenate((y_train, y_fdc), axis=0)
    return X_train, X_test, y_train, y_test


def machine_learning_models(ml_model, X_train, X_test, y_train, y_test, model_name, multi_output=True):
    """
    This function uses a machine learning algorithm in order to make predictions, fits the data and prints
    its evaluation.
    """
    if multi_output:
        model = MultiOutputRegressor(ml_model)
    else:
        model = ml_model
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
    model.fit(X_train, y_train)
    print(f'{model_name} train score: ', end='')
    print(model.score(X_train, y_train))
    print(f'{model_name} test score: ', end='')
    print(model.score(X_test, y_test))


def correct_predictions(predictions, micros):
    prediction_counter = 0
    for i in range(predictions.shape[0]):
        for j in range(len(predictions[i])):
            if micros[i][j] - ACCURACY_THRESHOLD <= predictions[i][j] <= micros[i][j] + ACCURACY_THRESHOLD:
                prediction_counter += 1
    return prediction_counter


def train(model, criterion, optimizer, train_loader, train_size, train_losses, train_accuracy):
    """trains the autoencoder model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for macros, micros in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(macros)
        loss = criterion(outputs, micros)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += micros.size(0)*micros.size(1)
        correct += correct_predictions(outputs, micros)
    train_losses.append(running_loss / train_size)
    train_accuracy.append(100*correct/total)
    return train_losses, train_accuracy


def test(model, criterion, testloader, test_size, test_losses, test_accuracy):
    """tests an MLP model for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for macros, micros in tqdm(testloader):
            outputs = model(macros)
            loss = criterion(outputs, micros)

            running_loss += loss.item()
            total += micros.size(0)*micros.size(1)
            correct += correct_predictions(outputs, micros)
    test_losses.append(running_loss / test_size)
    test_accuracy.append(100*correct/total)
    return test_losses, test_accuracy


def deep_learning_model(X_train, X_test, y_train, y_test):
    """
    This function uses a basic deep learning model - multi layer perceptron in order to learn the features of the
    macros and make predictions. The model has two hidden layers with relu activation, dropout regularization after
    each layer, and a fully connected output layer with the number of neurons as the number of labels to predict.
    :return: The model accuracy and history to be used in plots later.
    """
    train_set = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_set = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    train_loader = DataLoader(dataset=train_set, batch_size=32, num_workers=2)
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=False, num_workers=2)
    model = MLP(X_train.shape[1], y_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses, test_losses = [], []
    train_accuracy, test_accuracy = [], []
    for epoch in range(10):
        train_losses, train_accuracy = train(model, criterion, optimizer, train_loader, len(train_set),
                                             train_losses, train_accuracy)
        test_losses, test_accuracy = test(model, criterion, test_loader, len(test_set), test_losses, test_accuracy)
    with torch.no_grad():
        print(model(torch.tensor([0.76, 0.09, 5.12])))
    return train_losses, test_losses, train_accuracy, test_accuracy


def plot_train_test(train_losses, test_losses, train_accuracy, test_accuracy, micro_name):
    """
    This function draws the accuracy and loss function of the train and validation sets of the multi layer
     perceptron model.
    :param history: The history of the model containing the accuracy and loss after every epoch.
    """
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.title(f'{micro_name} Train vs Val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train loss', 'validation loss'])
    plt.grid()
    plt.show()

    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.title(f'{micro_name} Train vs Val accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train accuracy', 'validation accuracy'])
    plt.grid()
    plt.show()


def multi_output_regression():
    df_israeli, df_fdc = preprocess_data()
    df_israeli = preprocess_micronutrients(df_israeli, MICRO_NUTRIENTS)
    df_fdc = preprocess_micronutrients(df_fdc, MICRO_NUTRIENTS)
    X_train, X_test, y_train, y_test = split_data(df_israeli, df_fdc, MACRO_NUTRIENTS, MICRO_NUTRIENTS)
    machine_learning_models(LinearRegression(), X_train, X_test, y_train, y_test, "linear regression")
    machine_learning_models(KNeighborsRegressor(), X_train, X_test, y_train, y_test, "k-nearest neighbors")
    machine_learning_models(DecisionTreeRegressor(), X_train, X_test, y_train, y_test, "decision tree")
    machine_learning_models(RandomForestRegressor(n_estimators=100, max_features="sqrt"),
                            X_train, X_test, y_train, y_test, "random forest")
    train_losses, test_losses, train_accuracy, test_accuracy = deep_learning_model(X_train, X_test, y_train, y_test)
    plot_train_test(train_losses, test_losses, train_accuracy, test_accuracy, "All micronutrients")


def single_output_regression():
    df_israeli, df_fdc = preprocess_data()
    for micro in MICRO_NUTRIENTS:
        df_israeli = preprocess_micronutrients(df_israeli, [micro])
        df_fdc = preprocess_micronutrients(df_fdc, [micro])
        X_train, X_test, y_train, y_test = split_data(df_israeli, df_fdc, MACRO_NUTRIENTS, [micro])
        machine_learning_models(LinearRegression(), X_train, X_test, y_train, y_test, "linear regression", False)
        machine_learning_models(KNeighborsRegressor(), X_train, X_test, y_train, y_test, "k-nearest neighbors", False)
        machine_learning_models(DecisionTreeRegressor(), X_train, X_test, y_train, y_test, "decision tree", False)
        machine_learning_models(RandomForestRegressor(n_estimators=100, max_features="sqrt"),
                                X_train, X_test, y_train, y_test, "random forest", False)
        train_losses, test_losses, train_accuracy, test_accuracy = deep_learning_model(X_train, X_test, y_train, y_test)
        plot_train_test(train_losses, test_losses, train_accuracy, test_accuracy, micro)


if __name__ == '__main__':
    #multi_output_regression()
    single_output_regression()
