import torch
import torch.nn as nn
from models_sd.blocks import DownBlock, MidBlock, UpBlock


class VAE(nn.Module):
    def __init__(self, im_channels, model_config):
        super(VAE, self).__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']

        # disable self attention in downblock and upblock of the encoder
        self.attns = model_config['attn_down']

        # Latent dimension
        self.z_channels = model_config['z_channels']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']

        # Assert to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) ==len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        self.up_sample = list(reversed(self.down_sample))


        # Encoder
        self.encoder_conv_in = nn.Conv2d(in_channels=im_channels, out_channels=self.down_channels[0], kernel_size=3, padding=(1, 1))

        # Downblock + Midblock
        self.encoder_downs = nn.ModuleList([])
        for i in range(len(self.down_channels)):
            self.encoder_blocks.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                                 t_emb_dim=None, down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 attn=self.attns[i],
                                                 norm_channels=self.norm_channels))
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        self.encoder_norm_out = nn.GroupNorm(num_groups=self.norm_channels, num_channels=self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(in_channels=self.down_channels[-1], out_channels=2*self.z_channels, kernel_size=3, padding=1)

        # Latent dimension is 2 * latent dim as we are predicting mean and log variance
        self.pre_quant_conv = nn.Conv2d(in_channels=2*self.z_channels, out_channels=2*self.z_channels, kernel_size=1)
        ###############################################################################



        # Decoder
        self.post_quant_conv = nn.Conv2d(in_channels=self.z_channels, out_channels=self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(in_channels=self.z_channels, out_channels=self.mid_channels[-1], kernel_size=3, padding=(1, 1))

        #Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        self.decoder_ups = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.decoder_ups.append(UpBlock(self.down_channels[i], self.down_channels[i - 1],
                                               t_emb_dim=None, up_sample=self.down_sample[i - 1],
                                               num_heads=self.num_heads,
                                               num_layers=self.num_up_layers,
                                               attn=self.attns[i - 1],
                                               norm_channels=self.norm_channels))

        self.decoder_norm_out = nn.GroupNorm(num_groups=self.norm_channels, num_channels=self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(in_channels=self.down_channels[0], out_channels=im_channels, kernel_size=3, padding=1)


    def encode(self, x):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_downs):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)

        mean, logvar = torch.chunk(input=out, chunks=2, dim=1)
        std = torch.exp(0.5*logvar)
        sample = mean + std * torch.randn(mean.shape).to(device=x.device)
        return sample, out

    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_ups):
            out = up(out)
        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out

    def forward(self, x):
        z, encoder_output = self.encode(x)
        out = self.decode(z)
        return out, encoder_output

