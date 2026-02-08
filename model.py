import torch
import torch.nn as nn
import torch.nn.functional as f

class PrintSize(nn.Module):
    """Utility module to print current shape of a Tensor in Sequential, only at the first pass."""
    first = True
    
    def forward(self, x):
        if self.first:
            print(f"Size: {x.size()}")
            self.first = False
        return x

class UNetModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = 32
        self.in_channels = 1
    
        # ENCODER
        self.e1 = self.block(self.in_channels, self.features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.e2 = self.block(self.features, 2 * self.features)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.e3 = self.block(2 * self.features, 4 * self.features)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # BOTTLENECK
        self.b1 = self.block(4 * self.features, 8 * self.features)

        # DECODER
        self.up1 = nn.ConvTranspose2d(8 * self.features, 4 * self.features, kernel_size=2, stride=2)
        self.d1 = self.block(8 * self.features, 4 * self.features) # 4 inputs from up + 4 from concat

        self.up2 = nn.ConvTranspose2d(4 * self.features, 2 * self.features, kernel_size=2, stride=2)
        self.d2 = self.block(4 * self.features, 2 * self.features) # 2 inputs from up + 2 from concat
        
        self.up3 = nn.ConvTranspose2d(2 * self.features, self.features, kernel_size=2, stride=2)
        self.d3 = self.block(2 * self.features, self.features)

        self.final = nn.Conv2d(self.features, self.in_channels, kernel_size=1)
        
    
    def block(self, input_channel, output_channel):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            PrintSize(),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            PrintSize(),
        )


    def forward(self, x):
        # ENCODER
        enc1 = self.e1(x)
        maxp1 = self.pool1(enc1)

        enc2 = self.e2(maxp1)
        maxp2 = self.pool2(enc2)

        enc3 = self.e3(maxp2)
        maxp3 = self.pool3(enc3)

        # BOTTLENECK
        bottleneck = self.b1(maxp3)

        # DECODER
        dec1 = self.up1(bottleneck)
        dec1 = torch.cat((enc3, dec1), dim=1)
        dec1 = self.d1(dec1)

        dec2 = self.up2(dec1)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.d2(dec2)

        dec3 = self.up3(dec2)
        dec3 = torch.cat((enc1, dec3), dim=1)
        dec3 = self.d3(dec3)

        # Final layer
        return self.final(dec3)


        

