
from tkinter import Y
import torch
import numpy as np
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 0) Prepare Data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train) #fits or calculates mean and variance form the training set
X_test = sc.transform(X_test)       #uses the mean and variance from training set to scale the test set

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) model
# f = wx + b
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self,).__init__()
        self.linear = nn.Linear(n_input_features, 1)
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = LogisticRegression(n_features)

# 2) loss and optimizer
criterion = nn.BCELoss()
learning_r = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr = learning_r)

# 3) training loop
num_epochs = 400
for epoch in range(num_epochs):
    # forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    # backward pass
    loss.backward()
    
    # update
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch+1) % 40 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
        
# evaluation
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = y_pred.round()
    accuracy = y_pred_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {accuracy:.4f}')
