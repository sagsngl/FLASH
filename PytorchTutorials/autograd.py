import torch
import numpy as np

X = np.array([1, 2, 3, 4], dtype= np.float32)
Y = 9*X
w = 0.0

# model prediction
def forward(x):
    return w*x

# loss
def loss(y,y_pred):
    return ((y-y_pred)**2).mean()

# gradiant of loss
def gradient(x,y,y_pred):
    return np.dot(2*x, y_pred-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training 
lr = 0.01
n_iters = 15

for epoch in range(n_iters):
    y_pred = forward(X)
    l = loss(Y,y_pred)
    dw = gradient(X,Y,y_pred)
    
    w -=lr*dw
    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
        
print(f'Prediction after training: f(5) = {forward(5):.3f}')