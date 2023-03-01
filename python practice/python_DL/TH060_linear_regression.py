import torch
from torch import nn

# --- Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load data

# input data
x_data = torch.Tensor([[1.], [2.], [3.], [4.]])

# label
y_data = torch.Tensor([[2.], [4.], [6.], [8.]])

# --- Initialize neural network
model = nn.Linear(in_features=1, out_features=1, bias=True).to(device)

# --- define Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr= 0.001)

# --- Train model
for step in range(10000):
    # predict(forward)
    prediction = model(x_data)

    # loss 계산
    loss = criterion(input=prediction, target=y_data)

    # update weight(SGD)
    optimizer.zero_grad()
    loss.backward()     # gradient 계산
    optimizer.step()    # update weight

    print(f'step={step}, loss = {loss.item()}')

# --- infer
new_x = torch.Tensor([[5.]])

# predict(forward)
prediction = model(new_x)
print(f'\nnew_x = {new_x} ---> prediction result = {prediction.data.numpy()}')
