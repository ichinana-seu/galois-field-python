import numpy as np
import torch
import torch.nn as nn


def calc_normality_test(residual_noise: torch.Tensor, batch_size, batch_size_for_norm_test, Y_label_vectorlen):
    groups = int(batch_size // batch_size_for_norm_test)
    residual_noise = residual_noise.view(
        groups, Y_label_vectorlen * batch_size_for_norm_test)
    mean = residual_noise.mean(dim=1).view(groups, 1)
    variance = (residual_noise - mean).pow(2).mean(dim=1).view(groups, 1)
    moment_3rd = (residual_noise - mean).pow(3).mean(dim=1).view(groups, 1)
    moment_4th = (residual_noise - mean).pow(4).mean(dim=1).view(groups, 1)
    skewness = moment_3rd / (variance.pow(3 / 2.0) + 1e-10)
    kurtosis = moment_4th / (variance.pow(2) + 1e-10)
    norm_test = (skewness.pow(2) + 0.25 * (kurtosis - 3).pow(2)).mean()
    return norm_test


# physical_layers = 16
# filter_sizes = 9              ( Filter Dimension
# feature_map_nums = 64         ( Number of filters
class CorDnCNN(nn.Module):
    def __init__(self, logical_layers, feature_map_nums, filter_sizes, X_feature_vectorlen, Y_label_vectorlen):
        super(CorDnCNN, self).__init__()
        self.logical_layers = logical_layers                # BP25
        self.hidden_layers_ModuleList = nn.ModuleList()                  # 0 - physical_layers - 1
        self.X_feature_vectorlen = X_feature_vectorlen                  # 输出向量长度，等于LDPC_n
        self.Y_label_vectorlen = Y_label_vectorlen                      # 输出向量长度，等于LDPC_n

        # important arguments
        self.arg__kernel_nums = feature_map_nums         #   所谓的Kernel的数量 [64, 32, 16, 1] 。 对应1,2,3,4层的Kernel数量， 【会以64深度、32深度、...的形式反映在该层卷积层的输出上】【注意，最后一层一定是1】
        self.arg__kernel_height = filter_sizes           #   纵向卷积子Kernel的高度 [9,3,3,15]

        ############# BPCNN
                                                        # lambda=0 就是 base loss

        # 向 self.conv_layers 中添加卷积层
        # layer 1
        in_channels_Depth = 1
        out_channels_Depth = self.arg__kernel_nums
        add_layer = nn.Conv2d(in_channels_Depth, out_channels_Depth, (self.arg__kernel_height, 1), padding='same')
        self.hidden_layers_ModuleList.append(add_layer)
        self.hidden_layers_ModuleList.append(nn.ReLU(inplace=True))
        # layer 2 to 15
        for layer in range(2, self.logical_layers):         # layer=1，表示的是第1个layer
            in_channels_Depth =  self.arg__kernel_nums            # 所谓的Kernel的数量 [64, 32, 16, 1] 。 对应1,2,3,4层的Kernel数量（最后一定要是1）
            out_channels_Depth = self.arg__kernel_nums
            add_layer = nn.Conv2d(in_channels_Depth, out_channels_Depth,(self.arg__kernel_height, 1), padding='same')         # padding=same，保证第一层的输出高度为n，和第0层输出（即输入）的高度n一致！
                                                                                    # 和论文给的图有所区别！ 这里1到k个kernel是 深度方向堆叠的！，因此kernel的size是 高9，长1，深in_channels_Depth 的！
                                                                                    # 【卷积核的深度（也就是卷积核在通道维度上的大小）是由输入特征图的通道数 （in_channels_Depth） 自动确定的 = （上一层的out_channels_Depth）=（上一层的feature_map_num）】
            self.hidden_layers_ModuleList.append(add_layer)
            self.hidden_layers_ModuleList.append(nn.BatchNorm2d(out_channels_Depth))
            self.hidden_layers_ModuleList.append(nn.ReLU(inplace=True))
        # layer 16
        in_channels_Depth = self.arg__kernel_nums
        out_channels_Depth = 1
        add_layer = nn.Conv2d(in_channels_Depth, out_channels_Depth, (self.arg__kernel_height, 1), padding='same')
        self.hidden_layers_ModuleList.append(add_layer)


    def forward(self, x: torch.Tensor):
        x = x.reshape(-1, 1, self.X_feature_vectorlen, 1)       # 样本量（自动确定），通道数为1（没有深度），x向量长度为LDPC_n（长度），输入向量的宽度为1（不是2维的，是1维的）
        for physical_layer_id, this_layer in enumerate(self.hidden_layers_ModuleList):        
            x = this_layer(x)
        x = x.view(-1, self.Y_label_vectorlen)
        return x

    def enhancedLoss(self, output: torch.Tensor, target: torch.Tensor, batch_size: int, arg_a: int=1, arg_c: int=1) -> torch.Tensor:
        """lambda=0 就是 base loss"""
        lossA = arg_a * torch.nn.functional.mse_loss(output, target, reduction="mean")
        lossC = arg_c * calc_normality_test(target - output, batch_size, 1, self.Y_label_vectorlen)
        loss =  lossA + lossC
        return loss