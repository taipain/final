import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

val_acc_list = []
train_acc_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.cuda.amp.GradScaler()

parser = argparse.ArgumentParser()
parser.add_argument("-epoch", type=int, default=100)
parser.add_argument("-batch", type=int, default=8)
parser.add_argument("-val", type=float, default=0.3)
args = parser.parse_args()

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

all_data = datasets.ImageFolder(root='./dataset', transform=data_transform)
train_data, val_data = train_test_split(all_data, test_size=args.val)
print(len(train_data),len(val_data))
train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

num_classes = len(all_data.classes)
model = MLP(256*256*3, 256, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
best_val =0
for epoch in tqdm(range(args.epoch)):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        with torch.cuda.amp.autocast():
            pred = model(X)
            loss = criterion(pred, y)

        train_loss += loss.item()

        _, predicted = pred.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    train_acc_list.append(correct/total)  
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Train Accuracy: {correct/total}')
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            _, predicted = pred.max(1)
            val_total += y.size(0)
            val_correct += predicted.eq(y).sum().item()

    val_acc = val_correct / val_total
    val_acc_list.append(val_acc)
    print(f'Epoch: {epoch+1}, Val Accuracy: {val_acc}')

    if val_acc > best_val:
        best_val = val_acc
        print(f"Best validation Acc: {val_acc}")
        #torch.save(model.state_dict(), './best_model.pth')

print(f"Finish Best validation Acc {best_val}")
plt.figure()
plt.plot(range(1, args.epoch+1), train_acc_list, label='train')
plt.plot(range(1, args.epoch+1), val_acc_list, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig("reslut.png")
