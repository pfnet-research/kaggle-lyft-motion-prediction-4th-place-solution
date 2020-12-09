try:
    from efficientnet_pytorch import EfficientNet
    from efficientnet_pytorch.utils import get_same_padding_conv2d, round_filters

    _efficientnet_pytorch_available = True
except ImportError as e:
    _efficientnet_pytorch_available = False


from torch import nn


class EfficientNetWrapper(nn.Module):

    def __init__(self, model_name='efficientnet-b0', use_pretrained=False, in_channels=3):
        super(EfficientNetWrapper, self).__init__()
        if use_pretrained:
            self.model = EfficientNet.from_pretrained(model_name=model_name, in_channels=in_channels)
        else:
            model = EfficientNet.from_name(model_name, num_classes=1000)
            if in_channels != 3:
                Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
                out_channels = round_filters(32, model._global_params)
                model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
            self.model = model

    def custom_extract_features(self, model, inputs, block_start_index=0, apply_stem=True):
        """ Returns output of the final convolution layer """

        if apply_stem:
            # Stem
            x = model._swish(model._bn0(model._conv_stem(inputs)))
        else:
            x = inputs

        # Blocks
        for idx, block in enumerate(model._blocks):
            if idx < block_start_index:
                continue
            drop_connect_rate = model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = model._swish(model._bn1(model._conv_head(x)))
        return x

    def forward(self, x):
        x = self.custom_extract_features(self.model, x)
        x = self.model._avg_pooling(x)
        bs = x.shape[0]
        x = x.view(bs, -1)
        return x
