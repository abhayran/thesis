import torch
import torch.nn.functional as F


class DoubleConvTorch(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetTorch(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.double_convs_encoder = torch.nn.ModuleList([
            DoubleConvTorch(3, 6),
            DoubleConvTorch(6, 12),
            DoubleConvTorch(12, 24),
            DoubleConvTorch(24, 48),
            DoubleConvTorch(48, 96),
        ])
        self.double_convs_decoder = torch.nn.ModuleList([
            DoubleConvTorch(96, 48),
            DoubleConvTorch(48, 24),
            DoubleConvTorch(24, 12),
            DoubleConvTorch(12, 6),
        ])
        self.upsamplers = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(96, 48, 2, stride=2),
            torch.nn.ConvTranspose2d(48, 24, 2, stride=2),
            torch.nn.ConvTranspose2d(24, 12, 2, stride=2),
            torch.nn.ConvTranspose2d(12, 6, 2, stride=2),
        ])
        self.out_conv = torch.nn.Conv2d(6, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0 = self.double_convs_encoder[0](x)
        x_1 = F.max_pool2d(x_0, 2)

        x_1 = self.double_convs_encoder[1](x_1)
        x_2 = F.max_pool2d(x_1, 2)

        x_2 = self.double_convs_encoder[2](x_2)
        x_3 = F.max_pool2d(x_2, 2)

        x_3 = self.double_convs_encoder[3](x_3)
        x = F.max_pool2d(x_3, 2)

        x = self.double_convs_encoder[4](x)

        x = torch.cat([x_3, self.upsamplers[0](x)], dim=1)
        x = self.double_convs_decoder[0](x)

        x = torch.cat([x_2, self.upsamplers[1](x)], dim=1)
        x = self.double_convs_decoder[1](x)

        x = torch.cat([x_1, self.upsamplers[2](x)], dim=1)
        x = self.double_convs_decoder[2](x)

        x = torch.cat([x_0, self.upsamplers[3](x)], dim=1)
        x = self.double_convs_decoder[3](x)

        x = self.out_conv(x)
        return x
