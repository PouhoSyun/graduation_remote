import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, args, mid_channels=64, layer_cnt=3):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(args.imagemult_para, mid_channels, 4, 2, 1),
                  nn.LeakyReLU(0.2)]
        mult_para = 1

        for layer in layers(1, layer_cnt + 1):
            mult_para_last = mult_para
            mult_para = min(2 ** layer, 8)
            layers += [
                nn.Conv2d(mid_channels * mult_para_last, mid_channels * mult_para, 4, 
                          2 if layer < layer_cnt else 1, 1, bias=False),
                nn.BatchNorm2d(mid_channels * mult_para),
                nn.LeakyReLU(0.2, True)
            ]
        
        layers.append(nn.Conv2d(mid_channels * mult_para, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)