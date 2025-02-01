from torchvision.models import resnet50, ResNet50_Weights
from torch import nn, hub, std, cat, randn
from torch.nn import AdaptiveAvgPool2d


activation = {}
def getActivation(name):
    """Hook function to extract the output of a layer in the model."""
    def hook(model, input, output):
        try:
            activation[name] = output.detach()

        except AttributeError:
            for i, layer in enumerate(output):
                activation[name] = layer.detach()

    return hook
    

class ResNet50(nn.Module):
    """
    ResNet-50 backbone network for the dual-stage attention model.

    Fi = B(vi) = (f i,1 , f i,2 , f i,3 , f i,4),
    alphai = (GAP f i,1 ⊕ · · · ⊕ GAP f i,4 ),
    βi = (GSP f i,1 ⊕ · · · ⊕ GSP f i,4 ),
    Fi' = (alphai ⊕ βi)

    Output from the last 4 stages extracted following:
    https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/#accessing-a-particular-layer-from-the-model
    https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html

    Global Average Pooling:
    https://medium.com/@benjybo7/7-pytorch-pool-methods-you-should-be-using-495eb00325d6
    https://pytorch.org/docs/main/generated/torch.nn.AdaptiveAvgPool2d.html#adaptiveavgpool2d

    Pooling Standard Deviation:
    https://resources.wolframcloud.com/FormulaRepository/resources/Pooled-Standard-Deviation
    https://pytorch.org/docs/stable/generated/torch.std.html
    """
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.layers = list(self.model.children())[-6:-2]

        self.hooks = [layer.register_forward_hook(getActivation(
            str(layer[-1].conv1.in_channels)
        )) for layer in self.layers]

        self.avgpool = AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        _ = self.model(x[0][:, :, 0, :, :]) # TODO: check correct sample input.

        Fi = [
            activation[
                str(layer[-1].conv1.in_channels)
            ] for layer in self.layers
        ]

        alpha = cat([self.avgpool(fi).flatten() for fi in Fi])
        beta = cat([std(fi, dim=(2, 3)).flatten() for fi in Fi])

        semantic_features = cat([alpha, beta], dim=1)

        return semantic_features


class SlowFast(nn.Module):
    """
    SlowFast backbone network for the dual-stage attention model.

    Module documentation and example implementation @ https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/
    """
    def __init__(self):
        super(SlowFast, self).__init__()
        self.model = hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.layers = [layer for layer in self.model.blocks]

        self.hooks = [layer.register_forward_hook(getActivation(
            str(i)
        )) for i, layer in enumerate(self.layers)]

        self.avgpool = AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        _ = self.model(x)

        Fi = [activation[str(i)] for i, _ in enumerate(self.layers)]
        
        motion_features = self.avgpool(Fi[4]).flatten()

        return motion_features
    
class Backbone(nn.Module):
    """
    Backbone network for the dual-stage attention model.
    """
    def __init__(self):
        super(Backbone, self).__init__()
        self.resnet = ResNet50()
        self.slowfast = SlowFast()

    def forward(self, x):
        semantic_features = self.resnet(x)
        motion_features = self.slowfast(x)

        return cat([semantic_features, motion_features], dim=1)