import torch.nn.functional as F
import torch



class Alexnet(torch.nn.Module):
    def __init__(self, num_of_classes):
        super(Alexnet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(3, 96, 11, stride=4)
        self.pool_1 = torch.nn.MaxPool2d(3, stride=2)

        self.conv_2 = torch.nn.Conv2d(96, 256, 5)
        self.pool_2 = torch.nn.MaxPooling2d(pool_size=(3, 3), strides=(2, 2))

        self.conv_3 = torch.nn.Conv2d(256, kernel_size=(3, 3))
        self.conv_4 = torch.nn.Conv2d(384, kernel_size=(3, 3))
        self.conv_5 = torch.nn.Conv2d(384, kernel_size=(3, 3))

        self.pool_3 = torch.nn.MaxPooling2d(pool_size=(3, 3), strides=(2, 2))

        self.dense_1 = torch.nn.Linear(384, 4096)
        self.dropout_1 = torch.nn.Dropout(0.5)

        self.dense_2 = torch.nn.Linear(4096, 4096)
        self.dropout_2 = torch.nn.Dropout(0.5)

        self.output = torch.nn.Linear(4096, num_of_classes)

    def forward(self, *input):
        x = self.conv_1(input)
        x = F.relu(x)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = F.relu(x)
        x = self.pool_2(x)

        x = self.conv_3(x)
        x = F.relu(x)
        x = self.conv_4(x)
        x = F.relu(x)
        x = self.conv_5(x)
        x = self.pool_3(x)

        x = x.view(x.size(0), -1)

        x = self.dense_1(x)
        x = self.dropout_1(x)

        x = self.dense_2(x)
        x = self.dropout_2(x)
        return F.softmax(self.output(x))

    def fit(self, X, Y, epochs=100):
        for i in range(epochs):
            pass


    def predict(self, X):
        return self.forward(X)

    def evaluate(self):
        pass
