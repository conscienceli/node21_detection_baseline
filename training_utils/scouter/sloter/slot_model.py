import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.slot_attention import SlotAttention
from .utils.position_encode import build_position_encoding
# from timm.models import create_model
from collections import OrderedDict


class Identical(nn.Module):
    def __init__(self):
        super(Identical, self).__init__()

    def forward(self, x):
        return x


def load_backbone(args):
    bone = create_model(
        args.model,
        pretrained=args.pre_trained,
        num_classes=args.num_classes)
    if args.dataset == "MNIST":
        bone.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)
    if args.use_slot:
        if args.use_pre:
            checkpoint = torch.load(f"saved_model/{args.dataset}_no_slot_checkpoint.pth")
            new_state_dict = OrderedDict()
            for k, v in checkpoint["model"].items():
                name = k[9:] # remove `backbone.`
                new_state_dict[name] = v
            bone.load_state_dict(new_state_dict)
            print("load pre dataset parameter over")
        if not args.grad:
            if 'seresnet' in args.model:
                bone.avg_pool = Identical()
                bone.last_linear = Identical()
            elif 'res' in args.model:
                bone.global_pool = Identical()
                bone.fc = Identical()
            elif 'efficient' in args.model:
                bone.global_pool = Identical()
                bone.classifier = Identical()
            elif 'densenet' in args.model:
                bone.global_pool = Identical()
                bone.classifier = Identical()
            elif 'mobilenet' in args.model:
                bone.global_pool = Identical()
                bone.conv_head = Identical()
                bone.act2 = Identical()
                bone.classifier = Identical()
    return bone


class SlotModel(nn.Module):
    def __init__(self):
        super(SlotModel, self).__init__()

        self.feature_size = 100

        self.channel = 1024
        self.slots_per_class = 1
        self.hidden_dim = 64
        self.num_classes = 2
        self.conv1x1 = nn.Conv2d(self.channel, self.hidden_dim, kernel_size=(1, 1), stride=(1, 1))
        self.slot = SlotAttention(self.num_classes, self.slots_per_class, self.hidden_dim, vis=False,
                                        vis_id=0, loss_status=1, power=2., to_k_layer=3)
        self.position_emb = build_position_encoding('sine', hidden_dim=self.hidden_dim)
        self.lambda_value = 0.01
        

    def dfs_freeze(self, model, freeze_layer_num):
        if freeze_layer_num == 0:
            return

        unfreeze_layers = ['layer4', 'layer3', 'layer2', 'layer1'][:4-freeze_layer_num]
        for name, child in model.named_children():
            skip = False
            for freeze_layer in unfreeze_layers:
                if freeze_layer in name:
                    skip = True
                    break
            if skip:
                continue
            for param in child.parameters():
                param.requires_grad = False
            self.dfs_freeze(child, freeze_layer_num)

    def dfs_freeze_bnorm(self, model):
        for name, child in model.named_children():
            if 'bn' not in name:
                self.dfs_freeze_bnorm(child)
                continue
            for param in child.parameters():
                param.requires_grad = False
            self.dfs_freeze_bnorm(child)

    def forward(self, x, target):
        x = self.conv1x1(x.view(x.size(0), self.channel, self.feature_size, self.feature_size))
        x = torch.relu(x)
        pe = self.position_emb(x)
        x_pe = x + pe

        b, n, r, c = x.shape
        x = x.reshape((b, n, -1)).permute((0, 2, 1))
        x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
        x, attn_loss = self.slot(x_pe, x)

        # output = F.sigmoid(x)[:,0]
        output = F.log_softmax(x, dim=1)

        if target is not None:
            # print(output, target)
            loss = F.nll_loss(output, target.long()) + self.lambda_value * attn_loss
            # loss = F.binary_cross_entropy(output, target) + self.lambda_value * attn_loss
            return output, loss
        else:
            return output, 0

