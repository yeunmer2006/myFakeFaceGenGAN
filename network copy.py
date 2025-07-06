import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, feature_map=64, img_channels=3):
        super(Generator, self).__init__()  # 初始化生成器
        # 定义生成器的主网络结构
        # 输入: N x noise_dim x 1 x 1
        # 使用反卷积层将噪声向量转换为图像
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, feature_map * 16, 4, 1, 0, bias=False),  # -> 4x4
            nn.BatchNorm2d(feature_map * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map * 16, feature_map * 8, 4, 2, 1, bias=False),  # ->8x8
            nn.BatchNorm2d(feature_map * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map * 8, feature_map * 4, 4, 2, 1, bias=False),  # -> 16x16
            nn.BatchNorm2d(feature_map * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map * 4, feature_map * 2, 4, 2, 1, bias=False),  # -> 32x32
            nn.BatchNorm2d(feature_map * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map * 2, feature_map, 4, 2, 1, bias=False),      # -> 64x64
            nn.BatchNorm2d(feature_map),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map, img_channels, 4, 2, 1, bias=False),         #-> 128x128
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)



class Discriminator(nn.Module):
    def __init__(self,feature_map=64, img_channels=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入: 3 x 128 * 128
            nn.Conv2d(img_channels, feature_map, 4, 2, 1, bias=False),   # 128x128 -> 64x64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map, feature_map * 2, 4, 2, 1, bias=False),   # -> 32x32
            nn.BatchNorm2d(feature_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map * 2, feature_map * 4, 4, 2, 1, bias=False),  # -> 16x16
            nn.BatchNorm2d(feature_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map * 4, feature_map * 8, 4, 2, 1, bias=False),  # -> 8x8
            nn.BatchNorm2d(feature_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map * 8, feature_map * 16, 4, 2, 1, bias=False),  # -> 4x4
            nn.BatchNorm2d(feature_map * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map * 16, feature_map * 32, 4, 2, 1, bias=False),  # -> 2x2
            nn.BatchNorm2d(feature_map * 32),
            nn.LeakyReLU(0.2, inplace=True),

            # 最后使用自适应平均池化将特征图大小调整为 1x1
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_map * 32, 1, 1, 1, 0, bias=False),  # -> 1x1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(x.size(0)) # 输出为(batch_size, 1)

