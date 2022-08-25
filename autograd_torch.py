import torch
import numpy as np

X = torch.tensor([1, 2, 3, 4], dtype= torch.float32)
Y = 9*X
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w*x

# loss
def loss(y,y_pred):
    return ((y-y_pred)**2).mean()

# gradiant of loss

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training 
lr = 0.01
n_iters = 100

for epoch in range(n_iters):
    y_pred = forward(X)
    l = loss(Y,y_pred)
    l.backward()
    dw = w.grad.data
    w.data -=lr*dw
    w.grad.data.zero_()
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
        
print(f'Prediction after training: f(5) = {forward(5):.3f}')