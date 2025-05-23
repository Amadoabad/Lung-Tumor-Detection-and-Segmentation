import torch
import torch.nn as nn 
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same',  bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same',  bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, init_features=[64, 128, 256, 512]
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNET
        for features in init_features:
            self.downs.append(DoubleConv(in_channels, features))
            in_channels = features
            
        # Up part of UNET
        for features in reversed(init_features):
            self.ups.append(
                nn.ConvTranspose2d(
                    features * 2, features, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(features * 2, features))
        
        
        self.bottleneck = DoubleConv(init_features[-1], init_features[-1] * 2)
        
        self.final_conv = nn.Conv2d(init_features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Down part of UNET
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Up part of UNET
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            # Resize skip connection if needed ex:(161x161)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
   
            concat_skip = torch.cat((skip_connection, x), dim=1)
            
            # DoubleConv
            x = self.ups[idx + 1](concat_skip)
            
        return self.final_conv(x)
    
def test():
    x = torch.randn((3, 1, 32, 32))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    
    assert preds.shape == x.shape, f"Output shape: {preds.shape}"
            
if __name__ == "__main__":
    test()