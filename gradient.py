from tkinter import N
import torch
"""
x = torch.randn(3,requires_grad=True)
print(x)
y = x*x
print(y)
z = y*y*2 + 2*x
z = z.mean()
print(z)

z.backward()
print(x.grad)
"""
n = 15
weights  = torch.ones(n, requires_grad=True)
print(weights)
model = torch.randn(n,n)
print(weights)
for epoch in range(10):
    model_output = torch.matmul(model,weights).sum()
    print(model_output.data)
    model_output.backward()
    weights.data -= weights.grad.data*0.01
    weights.grad.zero_()

print(weights)