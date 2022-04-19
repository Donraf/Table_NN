import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from m_dataset import ImageDataset
from m_nn import NeuralNetwork


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

transform = transforms.Compose([
    transforms.CenterCrop(size=(2380, 1680)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
target_transform = transforms.Lambda(lambda y: torch.zeros(
    2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

table_dataset = ImageDataset(csv_file=r'C:\Users\Максим\Desktop\table.csv',
                             img_dir=r'C:\Users\Максим\Desktop\DS_Table_PT',
                             transform=transform, target_transform=target_transform)
dataset_loader = torch.utils.data.DataLoader(table_dataset, batch_size=10,
                                             shuffle=True, num_workers=0)

# model = NeuralNetwork().to(device)

net = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(dataset_loader, 0):
        print(f'Batch {i + 1}')
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        print(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        print(loss.item())
        if i % 20 == 19:    # print every 20 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0
print('Finished Training')

PATH = r'C:\Users\Максим\Desktop\model\model.pt'
torch.save(net.state_dict(), PATH)
