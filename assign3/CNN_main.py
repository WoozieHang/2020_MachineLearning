import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import hiddenlayer as h
import torchviz
from torchviz import make_dot
from torchvision.models import AlexNet
from tensorboardX import SummaryWriter

BATCH_SIZE = 30
EPOCH = 7

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # denormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


# # get some random training images
# data_iter = iter(train_loader)
# images, labels = data_iter.next()
#
# # show images
# imshow(torchvision.utils.make_grid(images,nrow=5))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))

# define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.section1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, 3, 1, 1),
                                            torch.nn.ReLU(),
                                            torch.nn.MaxPool2d(2, 2),
                                            torch.nn.Conv2d(64, 128, 3, 1, 1),
                                            torch.nn.ReLU(),
                                            torch.nn.MaxPool2d(2, 2))

        self.section2 = torch.nn.Sequential(torch.nn.Linear(7 * 7 * 128, 1024),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=0.5),
                                            torch.nn.Linear(1024, 120),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(120, 84),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(84, 10)
                                            )

    def forward(self, x):
        x = self.section1(x)
        x = x.view(-1, 7 * 7 * 128)
        x = self.section2(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
criterion = nn.CrossEntropyLoss()

# two optimizer method SGD vs Adam
#optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(net.parameters())

#training
for epoch in range(EPOCH):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        #inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.7f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = './mnist_net.pth'
torch.save(net.state_dict(), PATH)


data_iter = iter(test_loader)
images, labels = data_iter.next()
#
# # print images
# imshow(torchvision.utils.make_grid(images,nrow=5))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))
#
net = Net()
net.load_state_dict(torch.load('./mnist_net.pth'))
outputs = net(images)
#
# _, predicted = torch.max(outputs, 1)
#
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(BATCH_SIZE)))

correct = 0
total = 0
loss=0.0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        loss+=criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
print('loss: %.7f' % (loss/total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))



# #可视化神经网络结构
# net = Net()
# print(net)
#
# x = torch.randn(1, 1, 28, 28).requires_grad_(True)  # 定义一个网络的输入值
# with SummaryWriter(comment='Net') as w:
#     w.add_graph(net, x)