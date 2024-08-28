import torch
import torch.nn as nn
import torch.optim as optim

# 데이터 생성
torch.manual_seed(0)
X = torch.randn(100, 1)
y = (X > 0).float()  # 0 또는 1로 변환

# 모델 정의
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionModel()

# 손실 함수와 옵티마이저 정의
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 모델 학습
for epoch in range(100):
    model.train()
    
    # 예측값 계산
    y_pred = model(X)
    
    # 손실 계산
    loss = criterion(y_pred, y)
    
    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 예측
pred = model(X).detach().numpy()