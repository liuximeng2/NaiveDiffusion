import torch
import torch.nn as nn
import torch.nn.functional as F

def render_image_size(input_size, kernel_size, stride, padding):
    output_size = ((input_size - kernel_size + 2 * padding) // stride) + 1
    return output_size

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(SelfAttentionBlock, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads=4)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2) 
        # We need transpose because the multi-head attention mechanism expects the input tensor to have the shape [batch_size, sequence_length, embedding_dim]
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln) # self-attention
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, residule=False):
        super(DoubleConv, self).__init__()

        self.residule = residule
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, residule=None):
        x = self.norm1(F.gelu(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        if self.residule:
            return F.gelu(x + residule)
        else:
            return x
    
class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernal_size=2, kernel_size=3, stride=1, padding=1, time_dim=256):
        super(DownSampleBlock, self).__init__()

        self.process1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )

        self.process2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )    

        self.maxpool = nn.MaxPool2d(pool_kernal_size)  # reduve the size of the image by kernal_size

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_dim,
                out_channels
            ),
        )          

    def forward(self, x, t):
        x = self.maxpool(x) # reduce the image size by a factor of pool_kernal_size
        x = self.process1(x) # adds the image size by kernal size
        x = self.process2(x) 
        time_emb = self.emb_layer(t)[:, :, None, None] # equalling to unsqueeze(1).unsqueeze(2)
        return x + time_emb
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleConv(in_channels * 2, in_channels, residule=True)
        self.conv2 = DoubleConv(in_channels, out_channels)
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1) # is this also residule concat?
        x = self.conv1(x, residule=skip_x)
        x = self.conv2(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256, device="cuda"):
        super(UNet, self).__init__()

        self.time_dim = time_dim

        self.DoubleConv = DoubleConv(in_channels, out_channels)

        self.down1 = DownSampleBlock(out_channels, out_channels, time_dim=time_dim)
        self.sa1 = SelfAttentionBlock(out_channels)
        self.down2 = DownSampleBlock(out_channels, out_channels, time_dim=time_dim)
        self.sa2 = SelfAttentionBlock(out_channels)
        self.down3 = DownSampleBlock(out_channels, out_channels, time_dim=time_dim)
        self.sa3 = SelfAttentionBlock(out_channels)

        self.mid_conv1 = DoubleConv(out_channels, out_channels)
        self.mid_conv2 = DoubleConv(out_channels, out_channels)
        self.mid_conv3 = DoubleConv(out_channels, out_channels)

        self.up1 = UpSampleBlock(out_channels, out_channels, time_dim=time_dim)
        self.sa4 = SelfAttentionBlock(out_channels)
        self.up2 = UpSampleBlock(out_channels, out_channels, time_dim=time_dim)
        self.sa5 = SelfAttentionBlock(out_channels)
        self.up3 = UpSampleBlock(out_channels, out_channels, time_dim=time_dim)
        self.sa6 = SelfAttentionBlock(out_channels)

        self.decoder = nn.Conv2d(out_channels, 3, 1)

        self.device = device

    def pos_encoding(self, t, channels): # input t size: (batch_size/tobe_generated), want to generate 5 image at time step 5, [5, 5, 5, 5, 5]
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        ) # of shape (channels // 2)

        pos_enc_a = torch.sin(t.repeat(1, channels // 2 // t.shape[0]) * inv_freq) # shape (1, channels // 2)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2 // t.shape[0]) * inv_freq) # shape (1, channels // 2)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1) # shape (1, channels)
        return pos_enc

    def forward(self, x, t):

        t = self.pos_encoding(t, self.time_dim) # shape (1, time_dim)

        x1 = self.DoubleConv(x) # shape (batch_size, out_channels, img_size, img_size)
        x2 = self.down1(x1, t)  # shape (batch_size, out_channels, img_size/2, img_size/2)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t) # shape (batch_size, out_channels, img_size/4, img_size/4)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t) # shape (batch_size, out_channels, img_size/8, img_size/8)
        x4 = self.sa3(x4)

        x4 = self.mid_conv1(x4)
        x4 = self.mid_conv2(x4)
        x4 = self.mid_conv3(x4)

        x = self.up1(x4, x3, t) # shape (batch_size, out_channels, img_size/4, img_size/4)
        x = self.sa4(x)
        x = self.up2(x, x2, t) # shape (batch_size, out_channels, img_size/2, img_size/2)
        x = self.sa5(x)
        x = self.up3(x, x1, t) # shape (batch_size, out_channels, img_size, img_size)
        x = self.sa6(x)
        x = self.decoder(x) # shape (batch_size, 3, img_size, img_size)
        return x