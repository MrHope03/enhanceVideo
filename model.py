import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F
from  torchvision.ops.deform_conv import DeformConv2d

class BiLSTMForAudio(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
        super(BiLSTMForAudio, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class UpsamplingResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.conv_skip = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )  # 1x1 conv for matching dimensions

    def forward(self, x):
        skip = self.conv_skip(self.upsample(x))

        x = self.upsample(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x + skip


class ASubnet(nn.Module):
    def __init__(self, input_size, initial_feature_dim):
        super(ASubnet, self).__init__()
        self.fc = nn.Linear(input_size, initial_feature_dim * 4 * 4)
        self.upres_blocks = nn.Sequential(
            UpsamplingResBlock(initial_feature_dim, initial_feature_dim),
            UpsamplingResBlock(initial_feature_dim, initial_feature_dim // 2),
            UpsamplingResBlock(initial_feature_dim // 2, initial_feature_dim // 2),
            UpsamplingResBlock(initial_feature_dim // 2, initial_feature_dim // 4),
            UpsamplingResBlock(initial_feature_dim // 4, initial_feature_dim // 4),
            UpsamplingResBlock(initial_feature_dim // 4, initial_feature_dim // 4),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)
        x = self.upres_blocks(x)
        return x


class Convolution(nn.Module):
    def __init__(
        self,
        inChannel,
        outChannel,
        kernel_size=3,
        padding=1,
        stride=1,
        upsampling=False,
        downsampling=False,
        sampling_factor=2,
    ):
        super(Convolution, self).__init__()
        self.upsampling = upsampling
        self.downsampling = downsampling
        self.conv = nn.Conv2d(
            inChannel,
            outChannel,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.upsample = nn.Upsample(scale_factor=sampling_factor, mode="nearest")
        self.downsample = nn.Conv2d(
            outChannel,
            outChannel,
            kernel_size=kernel_size,
            padding=padding,
            stride=sampling_factor,
        )
        self.batchnorm = nn.BatchNorm2d(outChannel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.upsampling:
            x = self.upsample(x)
        x = self.relu(self.batchnorm(self.conv(x)))
        if self.downsampling:
            x = self.downsample(x)
        return x


class LSubnet(nn.Module):
    def __init__(self):
        super(LSubnet, self).__init__()
        self.layers = nn.Sequential(
            Convolution(1, 64),
            Convolution(64, 128, downsampling=True),
            Convolution(128, 256, downsampling=True),
            Convolution(256, 256, downsampling=True),
            Convolution(256, 256, downsampling=True),
            Convolution(256, 256, upsampling=True),
            Convolution(256, 256, upsampling=True),
            Convolution(256, 128, upsampling=True),
            Convolution(128, 64, upsampling=True),
        )

    def forward(self, x):
        return self.layers(x)


class GuidedCnn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(GuidedCnn, self).__init__()
        self.offset_conv = nn.Conv2d(
            in_channels=8,
            out_channels=18,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.deformable_conv = DeformConv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            bias=False,
        )

    def forward(self, pf, cf, mv):
        x = torch.cat([pf, cf, mv], dim=1)
        offset = self.offset_conv(x)
        return self.deformable_conv(pf, offset)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return x


class VSubnet(nn.Module):
    def __init__(self):
        super(VSubnet, self).__init__()
        self.initial_layers = nn.Sequential(
            Convolution(3, 64),
            Convolution(64, 128, downsampling=True),
            Convolution(128, 256, downsampling=True),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
        )
        self.final_layers = nn.Sequential(
            Convolution(256, 128, upsampling=True),
            Convolution(128, 64, upsampling=True),
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.res_blocks(x)
        x = self.final_layers(x)
        return x


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )
    return conv


def addPadding(srcShapeTensor, tensor_whose_shape_isTobechanged):

    if srcShapeTensor.shape != tensor_whose_shape_isTobechanged.shape:
        target = torch.zeros(srcShapeTensor.shape)
        target[
            :,
            :,
            : tensor_whose_shape_isTobechanged.shape[2],
            : tensor_whose_shape_isTobechanged.shape[3],
        ] = tensor_whose_shape_isTobechanged
        return target
    return tensor_whose_shape_isTobechanged


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(3, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up_conv_1 = double_conv(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

    def forward(self, image):
        # expected size
        # encoder (Normal convolutions decrease the size)
        x1 = self.down_conv_1(image)
        # print("x1 "+str(x1.shape))
        x2 = self.max_pool_2x2(x1)
        # print("x2 "+str(x2.shape))
        x3 = self.down_conv_2(x2)
        # print("x3 "+str(x3.shape))
        x4 = self.max_pool_2x2(x3)
        # print("x4 "+str(x4.shape))
        x5 = self.down_conv_3(x4)
        # print("x5 "+str(x5.shape))
        x6 = self.max_pool_2x2(x5)
        # print("x6 "+str(x6.shape))
        x7 = self.down_conv_4(x6)
        # print("x7 "+str(x7.shape))
        x8 = self.max_pool_2x2(x7)
        # print("x8 "+str(x8.shape))
        x9 = self.down_conv_5(x8)
        # print("x9 "+str(x9.shape))
        x = self.up_trans_1(x9)
        x = addPadding(x7, x)
        x = self.up_conv_1(torch.cat([x7, x], 1))

        x = self.up_trans_2(x)
        x = addPadding(x5, x)
        x = self.up_conv_2(torch.cat([x5, x], 1))

        x = self.up_trans_3(x)
        x = addPadding(x3, x)
        x = self.up_conv_3(torch.cat([x3, x], 1))

        x = self.up_trans_4(x)
        x = addPadding(x1, x)
        x = self.up_conv_4(torch.cat([x1, x], 1))

        x = self.out(x)
        tanh = nn.Tanh()
        x = tanh(x)
        # print(x.shape)
        return x


class SpatialAttentionFusion(nn.Module):
    def __init__(self):
        super(SpatialAttentionFusion, self).__init__()
        self.speech_conv = nn.Conv2d(64 * 2, 64, kernel_size=7, padding=3)
        self.landmark_conv = nn.Conv2d(64 * 2, 64, kernel_size=7, padding=3)
        self.fusion_conv = nn.Conv2d(64 * 3, 64, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, speech_features, video_features, landmark_features):
        speech_video = torch.cat([speech_features, video_features], dim=1)
        landmark_video = torch.cat([landmark_features, video_features], dim=1)
        speech_map = self.sigmoid(self.speech_conv(speech_video))
        landmark_map = self.sigmoid(self.landmark_conv(landmark_video))
        landmark_map = landmark_map * landmark_features
        speech_map = speech_map * speech_features

        fused_features = torch.cat([speech_map, video_features, landmark_map], dim=1)
        fused_features = self.fusion_conv(fused_features)
        return fused_features


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bilstm = BiLSTMForAudio(input_size=40, hidden_size=128, num_layers=3)
        self.l_subnet = LSubnet()
        self.a_subnet = ASubnet(256, 256)
        self.v_subnet = VSubnet()
        self.galign = GuidedCnn(8, 3)
        self.fusion = SpatialAttentionFusion()
        self.conv = nn.Conv2d(64, 3, kernel_size=3, padding=1, stride=1)
        self.unet = UNet()

    def forward(self, frame, mfccs, mv, landmarks, prev):
        bi = self.bilstm(mfccs)
        audio_features = self.a_subnet(bi)
        aligned_features = self.galign(prev, frame, mv)
        video_features = self.v_subnet(aligned_features)
        landmarks = landmarks.unsqueeze(1)
        landmark_features = self.l_subnet(landmarks)
        len = landmark_features.shape[0]
        out = self.fusion(audio_features[:len], video_features, landmark_features)
        out = self.conv(out)
        out = self.unet(out)
        return out
