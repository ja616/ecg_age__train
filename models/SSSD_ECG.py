import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sssd.utils.util import calc_diffusion_step_embedding
from sssd.models.S4Model import S4Layer

def swish(x):
    return x * torch.sigmoid(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)
    def forward(self, x):
        return self.conv(x)
    
class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
    def forward(self, x):
        return self.conv(x)

class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels,
                 diffusion_step_embed_dim_out, in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 label_embed_dim=None):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)
        
        self.S41 = S4Layer(features=2*self.res_channels, 
                           lmax=s4_lmax,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm)
        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)
        self.S42 = S4Layer(features=2*self.res_channels, 
                           lmax=s4_lmax,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm)
        
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)
        
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)
        
        # Layer-specific fc for label embedding (conditional case)
        self.fc_label = nn.Linear(label_embed_dim, 2 * self.res_channels) if label_embed_dim is not None else None

    def forward(self, input_data):
        x, label_embed, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels

        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, self.res_channels, 1])
        h = h + part_t

        h = self.conv_layer(h)
        h = self.S41(h.permute(2,0,1)).permute(1,2,0)

        if self.fc_label is not None:
            #print("label_embed before fc_label:", label_embed.shape)
            
            # Flatten label embedding to 2D (batch_size, label_embed_dim) if needed
            label_embed = label_embed.view(label_embed.size(0), -1)
            
            label_embed = self.fc_label(label_embed).unsqueeze(2)  # output shape (B, 2C, 1)
            #print("label_embed after fc_label:", label_embed.shape)
            
            #print('h shape:', h.shape)
            #print('label_embed shape:', label_embed.shape)
            
            h = h + label_embed


        h = self.S42(h.permute(2,0,1)).permute(1,2,0)

        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])
        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)
        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers, 
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 label_embed_dim=None):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)
        
        self.residual_blocks = nn.ModuleList()
        for _ in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels, 
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm,
                                                       label_embed_dim=label_embed_dim))
            
    def forward(self, input_data):
        noise, label_embed, diffusion_steps = input_data
        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))
        h = noise
        skip = 0
        for block in self.residual_blocks:
            h, skip_n = block((h, label_embed, diffusion_step_embed))  
            skip += skip_n  
        return skip * math.sqrt(1.0 / self.num_res_layers)

class SSSD_ECG(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers,
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 label_embed_classes=0,
                 label_embed_dim=128):
        super(SSSD_ECG, self).__init__()
        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())
        
        # Embedding layer for categorical conditioning (not used if label_embed_classes=0)
        self.embedding = nn.Embedding(label_embed_classes, label_embed_dim) if label_embed_classes > 0 else None

        # Age embedding layer for continuous conditioning
        # Define only if label_embed_classes == 0 (continuous age)
        self.age_embedding = nn.Linear(1, label_embed_dim) if label_embed_classes == 0 else None
                
        self.residual_layer = Residual_group(res_channels=res_channels, 
                                             skip_channels=skip_channels, 
                                             num_res_layers=num_res_layers, 
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm,
                                             label_embed_dim=label_embed_dim if label_embed_classes == 0 else None)
        
        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(),
                                        ZeroConv1d(skip_channels, out_channels))

    def forward(self, input_data):
        noise, label, diffusion_steps = input_data
        if self.embedding is not None:
            # Categorical conditioning (not used in age conditioning)
            label_embed = label @ self.embedding.weight
        elif self.age_embedding is not None:
            # Continuous age conditioning: embed the scalar
            label_embed = self.age_embedding(label)
        else:
            label_embed = None
        
        x = noise
        x = self.init_conv(x)
        x = self.residual_layer((x, label_embed, diffusion_steps))
        y = self.final_conv(x)
        return y
