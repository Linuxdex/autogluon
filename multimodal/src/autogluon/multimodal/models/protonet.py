"""
    Support protonet and protonet_finetune for few-shot learning.
    Refer to https://github.com/hushell/pmf_cvpr22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import random
from ..constants import  LOGITS

def random_hflip(tensor, prob):
    """
    random hflip for images.

    Parameters
    ----------
    tensor
        The images input.
    prob
        The probability of augmentation.

    Return
    ------
    The images after hflip augmentation.
    """

    if prob > random.random():
        return tensor
    return torch.flip(tensor, dims=(3,))

def rand_brightness(x):
    """
    random brightness augmentation for images.

    Parameters
    ----------
    x
        The images input.

    Return
    ------
    The images after brightness augmentation.
    """

    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x

def rand_saturation(x):
    """
    random saturation augmentation for images.

    Parameters
    ----------
    x
        The images input.

    Return
    ------
    The images after saturation augmentation.
    """

    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x

def rand_contrast(x):
    """
    random saturation augmentation for images.

    Parameters
    ----------
    x
        The images input.

    Return
    ------
    The images after saturation augmentation.
    """

    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x

def rand_translation(x, ratio=0.125):
    """
    random translation augmentation for images.

    Parameters
    ----------
    x
        The images input.
    ratio
        The ratio of image translation.

    Return
    ------
    The images after translation augmentation.
    """

    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

def rand_offset(x, ratio=1, ratio_h=1, ratio_v=1):
    """
    random offset augmentation for images.

    Parameters
    ----------
    x
        The images input.
    ratio
        The offset ratio for images.
    ratio_h
        The offset ratio for h.
    retio_v
        The offset ratio for v.

    Return
    ------
    The images after offset augmentation.
    """

    w, h = x.size(2), x.size(3)

    imgs = []
    for img in x.unbind(dim = 0):
        max_h = int(w * ratio * ratio_h)
        max_v = int(h * ratio * ratio_v)

        value_h = random.randint(0, max_h) * 2 - max_h
        value_v = random.randint(0, max_v) * 2 - max_v

        if abs(value_h) > 0:
            img = torch.roll(img, value_h, 2)

        if abs(value_v) > 0:
            img = torch.roll(img, value_v, 1)

        imgs.append(img)

    return torch.stack(imgs)

def rand_offset_h(x, ratio=1):
    """
    random offset augmentation for the h of images.

    Parameters
    ----------
    x
        The images input.
    ratio
        The offset ratio for h.

    Return
    ------
    The images after h offset augmentation.
    """

    return rand_offset(x, ratio=1, ratio_h=ratio, ratio_v=0)

def rand_offset_v(x, ratio=1):
    """
    random offset augmentation for the v of images.

    Parameters
    ----------
    x
        The images input.
    ratio
        The offset ratio for v.

    Return
    ------
    The images after v offset augmentation.
    """

    return rand_offset(x, ratio=1, ratio_h=0, ratio_v=ratio)

def rand_cutout(x, ratio=0.5):
    """
    random cutout augmentation for images.

    Parameters
    ----------
    x
        The images input.
    ratio
        The ratio of cutout.

    Return
    ------
    The images after cutout augmentation.
    """

    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

def DiffAugment(x, types=[], prob = 0.5, detach=True):
    """
    The augmentations for tensorsupport in protonet_finetune.

    Parameters
    ----------
    x
        The images input.
    types
        The augmentation types.
    probs
        The probability of augmentations.
    detach
        Whether the augmentation is detached with the model.

    Return
    ------
    The images after augmentations
    """

    AUGMENT_FNS = {
        "color": [rand_brightness, rand_saturation, rand_contrast],
        "offset": [rand_offset],
        "offset_h": [rand_offset_h],
        "offset_v": [rand_offset_v],
        "translation": [rand_translation],
        "cutout": [rand_cutout],
    }
    if random.random() < prob:
        with torch.set_grad_enabled(not detach):
            x = random_hflip(x, prob=0.5)
            for p in types:
                for f in AUGMENT_FNS[p]:
                    x = f(x)
            x = x.contiguous()
    return x

class ProtoNet(nn.Module):
    """
    The ProtoNet for few-shot learning.
    Refer to https://github.com/hushell/pmf_cvpr22
    """

    def __init__(self, prefix, backbone):
        """
        Parameters
        ----------
        prefix
            The model prefix.
        backbone
            The Multimodal backbone used for feature extraction.
        """

        super().__init__()

        # bias & scale of cosine classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

        # backbone
        self.prefix = prefix
        self.backbone = backbone
        self.outputprefix = backbone.prefix
        self.name_to_id = self.get_layer_ids()

    def cos_classifier(self, w, f):
        """
        Parameters
        ----------
        w
            The prototype features.
        f
            The features to be classified.
        """

        f = F.normalize(f, p=2, dim=f.dim()-1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim()-1, eps=1e-12)

        cls_scores = f @ w.transpose(1, 2) # B, M, nC
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        return cls_scores

    def forward(self, supp_x, supp_y, x):
        """
        Parameters
        ----------
        supp_x
            The support tensor of few-shot.
        supp_y
            The few-shot label of support tensor.
        x
            The query tensor of few-shot.

        Return
        ------
        A dictionary with logits.
        """

        B, nSupp= supp_y.shape
        num_classes = int(supp_y[0].max() + 1)
        nQry = -1

        for k in supp_x:
            xsize = list(supp_x[k].shape[1:])
            xsize[0] = -1
            supp_x[k] = torch.reshape(supp_x[k], xsize)

        for k in x:
            xsize = list(x[k].shape[1:])
            nQry = xsize[0]
            xsize[0] = -1
            x[k] = torch.reshape(x[k], xsize)

        supp_f = self.backbone.forward(supp_x)
        supp_f = supp_f[self.outputprefix]["features"].view(B, nSupp, -1)

        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2)  # B, nC, nSupp

        prototypes = torch.bmm(supp_y_1hot.float(), supp_f)  # B, nC, d
        prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True)  # NOTE: may div 0

        # compute feature for z
        feat = self.backbone.forward(x)
        feat = feat[self.outputprefix]["features"].view(B, nQry, -1)  # B, nQry, d

        # classification
        logits = self.cos_classifier(prototypes, feat)  # B, nQry, nC
        return {
            self.prefix: {
                LOGITS: logits,
            }
        }

    def get_layer_ids(
        self,
    ):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end.

        It assumes that each individual model has the "name_to_id" attribute storing
        the already computed model's layer ids. This function only collects those layer ids.
        It also add prefixes for each model's parameter names since the fusion model wraps
        those individual models, making the name scope changed. Configuring the optimizer
        requires a full name of each parameter.

        The layers defined in this class, e.g., head, adapter,
        and, fusion_mlp, have id 0.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "backbone"
        names = [n for n, _ in self.named_parameters()]

        outer_layer_names = [n for n in names if not n.startswith(model_prefix)]
        name_to_id = {}
        for n in outer_layer_names:
            name_to_id[n] = 0

        for n, layer_id in self.backbone.name_to_id.items():
            full_n = f"{model_prefix}.{n}"
            name_to_id[full_n] = layer_id

        # double check each parameter has been assigned an id
        for n in names:
            assert n in name_to_id

        return name_to_id

class ProtoNet_Finetune(ProtoNet):
    def __init__(self,
                 prefix,
                 backbone,
                 num_iters=5,
                 lr=1e-5,
                 aug_prob=0.9,
                 aug_types=["color", "translation"],
    ):
        """
        Parameters
        ----------
        prefix
            The model prefix.
        backbone
            The Multimodal backbone used for feature extraction.
        num_iters
            The max epoch of self-training in prediction.
        lr
            The learning rate of fine-tuning.
        aug_prob
            The probability of augmentations.
        aug_types
            The types of augmentations.
        """

        super().__init__(prefix, backbone)
        self.num_iters = num_iters
        self.lr = lr
        self.aug_types = aug_types
        self.aug_prob = aug_prob
        self.backbone_state = deepcopy(backbone.state_dict())


    def forward(self, supp_x, supp_y, x):
        """
        Parameters
        ----------
        supp_x
            The support tensor of few-shot.
        supp_y
            The few-shot label of support tensor.
        x
            The query tensor of few-shot.

        Return
        ------
        A dictionary with logits.
        """

        self.backbone.load_state_dict(self.backbone_state, strict=True)

        if self.lr == 0:
            return super().forward(supp_x, supp_y, x)

        B, nSupp= supp_y.shape
        nQry = -1
        num_classes = int(supp_y[0].max() + 1)

        criterion = nn.CrossEntropyLoss()
        for k in supp_x:
            xsize = list(supp_x[k].shape[1:])
            xsize[0] = -1
            supp_x[k] = torch.reshape(supp_x[k], xsize)
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2) # B, nC, nSupp
        supp_y = supp_y.view(-1)
        for k in x:
            xsize = list(x[k].shape[1:])
            nQry = xsize[0]
            xsize[0] = -1
            x[k] = torch.reshape(x[k], xsize)

        # create optimizer
        opt = torch.optim.Adam(self.backbone.parameters(),
                               lr=self.lr,
                               betas=(0.9, 0.999),
                               weight_decay=0.)

        def single_step(z, mode=True):
            '''
            Parameters
            ----------
            z
                The augmented support information or the query information for prediction
            mode
                Whether to be the training stage.

            Return
            ------
            The logits of the classification result and the loss.
            '''
            nSize = nSupp if mode else nQry
            with torch.set_grad_enabled(mode):
                # recalculate prototypes from supp_x with updated backbone
                supp_f = self.backbone.forward(supp_x)
                supp_f = supp_f[self.outputprefix]["features"].view(B, nSupp, -1)
                prototypes = torch.bmm(supp_y_1hot.float(), supp_f) # B, nC, d
                prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0

                # compute feature for z
                feat = self.backbone.forward(z)
                feat = feat[self.outputprefix]["features"].view(B, nSize, -1) # B, nQry, d

                # classification
                logits = self.cos_classifier(prototypes, feat) # B, nQry, nC
                loss = None

                if mode: # if enable grad, compute loss
                    loss = criterion(logits.view(B*nSupp, -1), supp_y)

            return logits, loss

        # main loop
        for i in range(self.num_iters):
            opt.zero_grad()
            z = supp_x
            z[self.outputprefix + "_image"] = DiffAugment(supp_x[self.outputprefix + "_image"].squeeze(1), self.aug_types, self.aug_prob, detach=True).unsqueeze(1)
            _, loss = single_step(z, True)
            loss.backward()
            opt.step()

        logits, _ = single_step(x, False)
        return {
            self.prefix: {
                LOGITS: logits,
            }
        }
