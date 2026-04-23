import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class RegionSimamModule(nn.Module):
    """Placeholder region-aware attention block."""

    def __init__(self):
        super().__init__()
        self.attention = nn.Conv2d(64, 64, kernel_size=3, padding=1)

    def forward(self, x, region_mask=None):
        return self.attention(x)


class MultiLevelAttentionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_local = nn.Sequential(
            nn.Conv2d(960, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            RegionSimamModule(),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.encoder_global = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x, region_mask=None):
        x = self.encoder_local(x)
        x = self.middle(x)
        x = self.encoder_global(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x))))
        return torch.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x


class ResidualCBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        residual = x
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x + residual


class MultiViewMobileNetV3WithResidualCBAM(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.mobilenetv3_axial = models.mobilenet_v3_large(pretrained=pretrained)
        self.mobilenetv3_sagittal = models.mobilenet_v3_large(pretrained=pretrained)
        self.mobilenetv3_coronal = models.mobilenet_v3_large(pretrained=pretrained)

        self.attention_net = MultiLevelAttentionNet()
        self.cbam_axial = ResidualCBAM(960)
        self.cbam_sagittal = ResidualCBAM(960)
        self.cbam_coronal = ResidualCBAM(960)

        self.conv_1x1 = nn.Conv2d(768, 2880, kernel_size=1)
        self.conv3d_1 = nn.Conv3d(in_channels=2880, out_channels=512, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_2 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_3 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3, 3, 3), padding=1)

        self.bn1 = nn.BatchNorm3d(512)
        self.bn2 = nn.BatchNorm3d(256)
        self.bn3 = nn.BatchNorm3d(128)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, axial, sagittal, coronal):
        axial_features = self.mobilenetv3_axial.features(axial)
        sagittal_features = self.mobilenetv3_sagittal.features(sagittal)
        coronal_features = self.mobilenetv3_coronal.features(coronal)

        axial_features = self.cbam_axial(axial_features)
        sagittal_features = self.cbam_sagittal(sagittal_features)
        coronal_features = self.cbam_coronal(coronal_features)

        axial_features = self.attention_net(axial_features)
        sagittal_features = self.attention_net(sagittal_features)
        coronal_features = self.attention_net(coronal_features)

        combined_features = torch.cat([axial_features, sagittal_features, coronal_features], dim=1)
        adjusted_features = self.conv_1x1(combined_features)
        adjusted_features = adjusted_features.unsqueeze(2)

        x = F.relu(self.bn1(self.conv3d_1(adjusted_features)))
        x = F.relu(self.bn2(self.conv3d_2(x)))
        x = F.relu(self.bn3(self.conv3d_3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

