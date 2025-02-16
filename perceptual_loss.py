import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG19_relu()
        self.criterion = nn.MSELoss()
        self.weights = [1.0/64, 1.0/64, 1.0/32, 1.0/32, 1.0/1]
        self.IN = nn.InstanceNorm2d(512, affine=False, track_running_stats=False)
        
        # Register normalization parameters as buffers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

    def forward(self, x, y):
        # Handle grayscale inputs
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        
        # Normalize inputs
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        # Get VGG features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        
        # Calculate perceptual loss
        loss  = self.weights[0] * self.criterion(self.IN(x_vgg['relu1_1']), self.IN(y_vgg['relu1_1']))
        loss += self.weights[1] * self.criterion(self.IN(x_vgg['relu2_1']), self.IN(y_vgg['relu2_1']))
        loss += self.weights[2] * self.criterion(self.IN(x_vgg['relu3_1']), self.IN(y_vgg['relu3_1']))
        loss += self.weights[3] * self.criterion(self.IN(x_vgg['relu4_1']), self.IN(y_vgg['relu4_1']))
        loss += self.weights[4] * self.criterion(self.IN(x_vgg['relu5_1']), self.IN(y_vgg['relu5_1']))

        return loss

class VGG19_relu(nn.Module):
    def __init__(self):
        super(VGG19_relu, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        
        # Define feature slices
        self.slices = {
            'relu1_1': slice(0, 2),
            'relu1_2': slice(2, 4),
            'relu2_1': slice(4, 7),
            'relu2_2': slice(7, 9),
            'relu3_1': slice(9, 12),
            'relu3_2': slice(12, 14),
            'relu3_3': slice(14, 16),
            'relu3_4': slice(16, 18),
            'relu4_1': slice(18, 21),
            'relu4_2': slice(21, 23),
            'relu4_3': slice(23, 25),
            'relu4_4': slice(25, 27),
            'relu5_1': slice(27, 30),
            'relu5_2': slice(30, 32),
            'relu5_3': slice(32, 34),
            'relu5_4': slice(34, 36),
        }
        
        # Create modules for each slice
        for name, slice_range in self.slices.items():
            self.add_module(name, nn.Sequential(*vgg[slice_range]))
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        x = self.relu1_1(x)
        features['relu1_1'] = x
        x = self.relu1_2(x)
        features['relu1_2'] = x
        x = self.relu2_1(x)
        features['relu2_1'] = x
        x = self.relu2_2(x)
        features['relu2_2'] = x
        x = self.relu3_1(x)
        features['relu3_1'] = x
        x = self.relu3_2(x)
        features['relu3_2'] = x
        x = self.relu3_3(x)
        features['relu3_3'] = x
        x = self.relu3_4(x)
        features['relu3_4'] = x
        x = self.relu4_1(x)
        features['relu4_1'] = x
        x = self.relu4_2(x)
        features['relu4_2'] = x
        x = self.relu4_3(x)
        features['relu4_3'] = x
        x = self.relu4_4(x)
        features['relu4_4'] = x
        x = self.relu5_1(x)
        features['relu5_1'] = x
        x = self.relu5_2(x)
        features['relu5_2'] = x
        x = self.relu5_3(x)
        features['relu5_3'] = x
        x = self.relu5_4(x)
        features['relu5_4'] = x
        
        return features