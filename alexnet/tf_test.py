from alexnet.tf import AlexNet

model = AlexNet(1000)

print(model.predict(5))