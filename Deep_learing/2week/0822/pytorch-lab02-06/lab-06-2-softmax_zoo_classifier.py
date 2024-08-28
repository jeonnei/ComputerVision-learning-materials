import torch
import numpy as np

torch.manual_seed(777)  # for reproducibility

# Load the data
xy = np.loadtxt('C:/Users/kchan/python-Deeplearning/DeepLearningZeroToAll/pytorch/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

nb_classes = 7  # 0 ~ 6

# Convert numpy arrays to PyTorch tensors
X = torch.tensor(x_data)
Y = torch.tensor(y_data, dtype=torch.long).view(-1)  # Ensure Y is 1D LongTensor for CrossEntropyLoss

# Model definition
model = torch.nn.Linear(16, nb_classes, bias=True)

# Cross entropy loss (which includes softmax)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(2001):
    optimizer.zero_grad()
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    # Prediction
    prediction = torch.argmax(hypothesis, 1)
    correct_prediction = (prediction == Y)
    accuracy = correct_prediction.float().mean()

    if step % 100 == 0:
        print(f"Step: {step}\tLoss: {cost.item():.3f}\tAcc: {accuracy.item():.2%}")

# Let's see if we can predict
pred = torch.argmax(hypothesis, 1)

for p, y in zip(pred, Y):
    print(f"[{bool(p.item() == y.item())}] Prediction: {p.item()} True Y: {y.item()}")
