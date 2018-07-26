from alexnet.torch_ import Alexnet

alex_net = Alexnet(1000)

for param in alex_net.parameters():
    print(type(param.data), param.size())
