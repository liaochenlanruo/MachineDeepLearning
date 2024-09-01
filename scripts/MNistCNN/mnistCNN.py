# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 数据预处理
# 使用Compose组合多个变换，这里将数据转换为张量并进行标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载MNIST数据集并划分为训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 定义批次大小
batch_size = 64
# 使用DataLoader加载数据，以便在训练过程中更方便地迭代数据
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 初始化卷积层、池化层、全连接层和dropout层
        # 定义第一个卷积层，用于提取特征
        # 输入通道数为1（适用于灰度图像），输出通道数为32，卷积核大小为5x5，步长为1，padding为2
        # 第一次卷积后生成的特征图大小为32*28*28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        
        # 定义最大池化层，用于降低特征维度，减少计算量
        # 池化窗口大小为2x2，步长为2，无padding
        # 第一次池化后生成的特征图大小为32*14*14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 定义第二个卷积层，进一步提取和整合特征
        # 输入通道数为32，输出通道数为64，卷积核大小为5x5，步长为1，padding为2
        # 第二次卷积后生成的特征图大小为64*14*14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        
        # 定义第一个全连接层，用于分类前的特征转换
        # 输入大小为64*7*7（这里的尺寸为64*7*7是因为在前向传播时对第二次卷积进行了池化操作），输出大小为1024
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        
        # 定义第二个全连接层，用于最终的分类输出
        # 输入大小为1024，输出大小为10（假设分类任务有10个类别）
        self.fc2 = nn.Linear(1024, 10)

        # 定义Dropout层，用于训练过程中的正则化，防止过拟合
        # Dropout比例为0.5，即在训练过程中随机将50%的元素置为0
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 定义前向传播过程
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 实例化模型
model = Net()

# 定义优化器
# 使用Adam优化器更新模型参数
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 定义损失函数
# 使用交叉熵损失函数进行分类任务
criterion = nn.CrossEntropyLoss()

# 训练模型
# 将模型转移到GPU设备上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义训练轮数
num_epochs = 10
# 开始训练过程
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将数据转移到GPU设备上（如果可用）
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印损失信息
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 评估模型
# 将模型设置为评估模式
model.eval()
# 禁用梯度计算以减少内存消耗
with torch.no_grad():
    correct = 0
    total = 0
    # 在测试集上进行预测
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 打印测试集上的准确率
    print(f'Test Accuracy of the model on the {total} test images: {100 * correct / total}%')