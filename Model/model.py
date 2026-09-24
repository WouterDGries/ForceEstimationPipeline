"""R(2+1)D-18 backbone + force-regression head for the fingertip force
estimator. Expects a (B, 4, T, 112, 112) RGB+relative-depth clip tensor
(see dataset.py for how that's built) and produces a (B, 1) normalized
force prediction (see dataset.py for the normalization scale).

Everything below is fixed by the model spec, not operator-tunable, so it
lives here as plain module constants rather than in config.yaml - compare
FRAME_FIELDS/FORCE_FIELDS in dataAcquisition/dataWriter.py, which are the
same kind of fixed structural constant.
"""

import torch
import torch.nn as nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

NUM_INPUT_CHANNELS = 4      # R, G, B, relative depth
RGB_CHANNELS = 3
DEPTH_CHANNEL_INDEX = 3
POOLED_FEATURE_DIM = 512    # r2plus1d_18's feature size after avgpool
HEAD_HIDDEN_DIM = 128
HEAD_DROPOUT = 0.4
FROZEN_LAYER_NAMES = ("layer1", "layer2")   # stem/layer3/layer4/fc stay trainable


def _adapt_stem_for_depth(model):
    """Input: an r2plus1d_18 model with its pretrained 3-channel stem.
    Replaces stem[0] (the 1x7x7 conv to 45 channels) with a 4-input-channel
    version: input channels 0-2 copy the pretrained RGB kernels unchanged,
    channel 3 (depth) starts as the mean of the three RGB kernels. Mutates
    model in place. Returns nothing.
    """
    old_conv = model.stem[0]
    new_conv = nn.Conv3d(
        NUM_INPUT_CHANNELS, old_conv.out_channels, kernel_size=old_conv.kernel_size,
        stride=old_conv.stride, padding=old_conv.padding, bias=False,
    )
    with torch.no_grad():
        new_conv.weight[:, :RGB_CHANNELS] = old_conv.weight
        new_conv.weight[:, DEPTH_CHANNEL_INDEX] = old_conv.weight.mean(dim=1)
    model.stem[0] = new_conv


def _build_head(mean_normalized_label):
    """Input: the mean normalized training label (from dataset.py). Returns
    the regression head (Dropout -> Linear -> GELU -> Dropout -> Linear ->
    1) that replaces r2plus1d_18's fc. The final linear layer starts with
    small weights and its bias set to mean_normalized_label, so the head
    starts out predicting the average training force instead of noise.
    """
    head = nn.Sequential(
        nn.Dropout(HEAD_DROPOUT),
        nn.Linear(POOLED_FEATURE_DIM, HEAD_HIDDEN_DIM),
        nn.GELU(),
        nn.Dropout(HEAD_DROPOUT),
        nn.Linear(HEAD_HIDDEN_DIM, 1),
    )
    final_linear = head[-1]
    nn.init.normal_(final_linear.weight, mean=0.0, std=0.01)
    nn.init.constant_(final_linear.bias, mean_normalized_label)
    return head


def _freeze_layers(model):
    """Input: the adapted model. Sets requires_grad=False on every
    parameter in FROZEN_LAYER_NAMES. Returns nothing.
    """
    for name in FROZEN_LAYER_NAMES:
        for param in getattr(model, name).parameters():
            param.requires_grad = False


def set_frozen_layers_eval(model):
    """Input: the adapted model - call this right after every
    model.train(). requires_grad=False alone does not stop a frozen
    layer's BatchNorm running statistics from updating, and .train() turns
    that back on every epoch; this puts FROZEN_LAYER_NAMES back into eval
    mode so their stats stay frozen too. The stem's BatchNorm is left in
    training mode (the stem itself is trainable and must adapt to the new
    depth channel). Returns nothing.
    """
    for name in FROZEN_LAYER_NAMES:
        getattr(model, name).eval()


def build_model(mean_normalized_label):
    """Input: the mean normalized training label (see dataset.py). Builds
    the Kinetics-400 pretrained r2plus1d_18, adapts its stem to 4 input
    channels, replaces its classification head with a force-regression
    head initialized to predict the mean label, and freezes layer1/layer2.
    Returns the model.
    """
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    _adapt_stem_for_depth(model)
    model.fc = _build_head(mean_normalized_label)
    _freeze_layers(model)
    return model


def trainable_parameters(model):
    """Input: the adapted model. Returns the list of parameters with
    requires_grad=True, for passing to the optimizer - frozen layer1/layer2
    parameters must never reach it.
    """
    return [param for param in model.parameters() if param.requires_grad]


def _pooled_features(model, clip):
    """Input: a video backbone (pretrained or adapted) and a clip tensor
    matching its stem's input channels. Runs stem -> layer1-4 -> avgpool
    and returns the flattened (B, 512) feature vector, i.e. the model's
    output just before its head/fc.
    """
    x = model.stem(clip)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


def check_shapes(model):
    """Input: the adapted model. Pushes a dummy (2, 4, 20, 112, 112) tensor
    through and asserts the feature-map shape after every stage against the
    spec. Returns nothing; raises AssertionError on a mismatch.
    """
    model.eval()
    clip = torch.zeros(2, NUM_INPUT_CHANNELS, 20, 112, 112)
    with torch.no_grad():
        x = model.stem(clip)
        assert x.shape == (2, 64, 20, 56, 56), f"stem shape {tuple(x.shape)}"
        x = model.layer1(x)
        assert x.shape == (2, 64, 20, 56, 56), f"layer1 shape {tuple(x.shape)}"
        x = model.layer2(x)
        assert x.shape == (2, 128, 10, 28, 28), f"layer2 shape {tuple(x.shape)}"
        x = model.layer3(x)
        assert x.shape == (2, 256, 5, 14, 14), f"layer3 shape {tuple(x.shape)}"
        x = model.layer4(x)
        assert x.shape == (2, 512, 3, 7, 7), f"layer4 shape {tuple(x.shape)}"
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        assert x.shape == (2, POOLED_FEATURE_DIM), f"pooled shape {tuple(x.shape)}"
        out = model.fc(x)
        assert out.shape == (2, 1), f"head output shape {tuple(out.shape)}"
    print("  [model] shape trace: stem/layer1/layer2/layer3/layer4/pool/head all match spec (PASS)")


def check_zero_depth_equivalence(model):
    """Input: the adapted 4-channel model. Independently builds a fresh,
    unmodified 3-channel r2plus1d_18 with the same pretrained weights, runs
    one random normalized RGB clip through its pooled features, and
    compares that to the adapted model's pooled features on the same clip
    with an all-zero depth channel appended. A zero channel contributes
    nothing to a bias-free convolution regardless of how the depth kernel
    was initialized, so any difference means the RGB kernels were not
    copied into the adapted stem correctly. Prints the result. Returns
    nothing; raises AssertionError if the difference exceeds tolerance.
    """
    reference_model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    reference_model.eval()
    model.eval()

    clip_rgb = torch.randn(1, RGB_CHANNELS, 20, 112, 112)
    clip_rgbd = torch.cat([clip_rgb, torch.zeros(1, 1, 20, 112, 112)], dim=1)

    with torch.no_grad():
        reference_features = _pooled_features(reference_model, clip_rgb)
        adapted_features = _pooled_features(model, clip_rgbd)

    max_abs_diff = (reference_features - adapted_features).abs().max().item()
    print(f"  [model] zero-depth equivalence: max abs diff = {max_abs_diff:.3e}")
    assert max_abs_diff < 1e-4, "adapted stem's RGB kernels were not copied correctly"


if __name__ == "__main__":
    print("[model] Building r2plus1d_18 with 4-channel stem and force-regression head")
    built_model = build_model(mean_normalized_label=0.0)   # placeholder label; train.py passes the real one

    num_trainable = sum(param.numel() for param in trainable_parameters(built_model))
    num_frozen = sum(param.numel() for param in built_model.parameters() if not param.requires_grad)
    print(f"[model] Trainable parameters: {num_trainable:,}  Frozen parameters: {num_frozen:,}")

    print("[model] Running checks")
    check_shapes(built_model)
    check_zero_depth_equivalence(built_model)
    print("[model] All checks passed")
