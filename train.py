from torchvision import transforms
from torch.optim import Adam

import config


class Training:
    def __init__(self, model):
        self.model = model
        self.transform = transforms.ToPILImage()
        self.optim = Adam(params=self.model.parameters(), lr=config.LEARNING_RATE, betas=(config.B1, config.B2))
    
    def gen_train(self, epochs, data):
        for epoch in range(epochs):
            for batch in data:
                
                x, y = batch
                x, y = x.to(device=config.DEVICE), y.to(device=config.DEVICE)
                y_gen = self.model(x)
                y_gen = self.transform(y_gen)
                y_gen.show()
                break
            break
    
    def disc_train(self, data, model):
        pass