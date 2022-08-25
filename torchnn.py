import torch
import numpy as np
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype= torch.float32)
Y = 9*X
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
n_samples, n_features = X.shape
print(n_samples,n_features)
input_size = n_features
output_size = n_features

# model prediction
model = nn.Linear(input_size, output_size)

learning_r = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_r )
# gradiant of loss
Xtest = torch.tensor([5], dtype=torch.float32)
print(f'Prediction before training: f(5) = {model(Xtest).item():.3f}')

# Training 
# learning_r = 0.01
n_iters = 200

for epoch in range(n_iters):
    y_pred = model(X)
    l = loss(Y,y_pred)
    l.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
        
print(f'Prediction after training: f(5) = {model(Xtest).item():.3f}')