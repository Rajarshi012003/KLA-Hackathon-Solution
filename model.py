import torch
import torch.nn as nn

class LayerNorm2d(nn.Module):
    """
    Channel-wise LayerNorm for 4D spatial feature maps ``(N, C, H, W)``.

    Normalizes each channel independently using its own spatial statistics
    (mean/var over H*W) and applies a per-channel affine ``(weight, bias)`` of
    shape ``(C, 1, 1)``. This is the standard fast LayerNorm2d idiom used in
    restoration networks and is correct for the NCHW layout.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, 1, 1) per-channel statistics over spatial dims (H, W).
        mean = x.mean(dim=(-2, -1), keepdim=True)
        var = x.var(dim=(-2, -1), unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias


class SimpleGate(nn.Module):
    """
    Activation-free non-linearity (multiply-based).

    Splits the input along the channel dimension into two halves and returns
    their Hadamard product:

        Y[c] = X[c] * X[c + C//2],  for c in [0, C//2)

    Requires an even number of channels. Replaces GELU/ReLU while lowering the
    memory-bandwidth cost of element-wise activations.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
      

class SimplifiedChannelAttention(nn.Module):
    """
    Lightweight global context: compress the channel descriptor via Global
    Average Pooling (GAP), pass it through two 1x1 convs (with a SimpleGate for
    non-linearity), gate with a sigmoid, and scale the input element-wise.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(
                "SimplifiedChannelAttention requires an even number of channels."
            )
        self.conv_a = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_b = nn.Conv2d(channels // 2, channels, kernel_size=1)
        self.act = SimpleGate()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        desc = x.mean(dim=(2, 3), keepdim=True)       # (B, C, 1, 1) GAP
        desc = self.conv_a(desc)                      # (B, C, 1, 1)
        desc = self.act(desc)                         # SimpleGate -> C//2
        desc = self.conv_b(desc)                      # (B, C, 1, 1)
        scale = self.sigmoid(desc)                    # per-channel gate in [0,1]
        return x * scale


class NAFBlock(nn.Module):
    """
    Standard NAFNet block: a pointwise/3x3 locality module plus an FFN
    refinement, both free of explicit activations, stabilized by Pre-LayerNorm.
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
    ) -> None:
        super().__init__()
        c = channels
        dw_c = c * dw_expand
        ff_c = c * ffn_expand

        # Pre-Normalization 
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # Locality module
        self.conv1 = nn.Conv2d(c, dw_c, kernel_size=1)
        self.conv2 = nn.Conv2d(dw_c, dw_c, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(dw_c // 2, c, kernel_size=1)
        self.sca = SimplifiedChannelAttention(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # FFN module
        self.conv4 = nn.Conv2d(c, ff_c, kernel_size=1)
        self.conv5 = nn.Conv2d(ff_c, ff_c, kernel_size=3, padding=1, groups=ff_c)
        self.conv6 = nn.Conv2d(ff_c // 2, c, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.act = SimpleGate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x

        # Locality 
        x = self.norm1(x)             # Apply Pre-Norm
        x = self.conv1(x)             # C -> dw_c
        x = self.conv2(x)             # 3x3 conv
        x = self.act(x)               # SimpleGate: dw_c -> dw_c // 2
        x = self.conv3(x)             # dw_c // 2 -> C
        x = self.sca(x)               # channel attention scale
        x = self.beta * x + inp       # learned residual

        # FFN 
        res = x
        y = self.norm2(x)             # Apply Pre-Norm
        y = self.conv4(y)             # C -> ff_c
        y = self.conv5(y)             # depthwise 3x3
        y = self.act(y)               # SimpleGate: ff_c -> ff_c // 2
        y = self.conv6(y)             # ff_c // 2 -> C
        y = self.gamma * y + res

        return y


class NAFNetSR(nn.Module):
    """
    Lightweight single-channel super-resolution NAFNet.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        width: int = 32,
        num_blocks: int = 4,
        scale: int = 2,
    ) -> None:
        super().__init__()
        if width % 2 != 0:
            raise ValueError("width must be even (SimpleGate splits channels).")

        self.width = width
        self.num_blocks = num_blocks
        self.scale = scale

        self.intro = nn.Conv2d(in_ch, width, kernel_size=3, padding=1, stride=1)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])
        self.conv_after_body = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(width, out_ch * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inp : torch.Tensor
            Degraded low-resolution grayscale ``(B, 1, H, W)``. Passed through
            RAW (no clipping/normalization); values may be <0 or >1.
        Returns
        -------
        torch.Tensor
            Restored ``(B, 1, scale*H, scale*W)``. Clamped to ``[0, 1]`` only during eval.
        """
        x = self.intro(inp)            # (B, width, H, W)
        skip = x                       # Save stem output for global residual
        
        x = self.body(x)               # (B, width, H, W)
        x = self.conv_after_body(x)    # (B, width, H, W)
        res = x + skip                 # Global residual addition

        out = self.upsample(res)

        if not self.training:
            out = torch.clamp(out, 0.0, 1.0)
            
        return out

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

if __name__ == "__main__":
    torch.manual_seed(0)
    model = NAFNetSR(in_ch=1, out_ch=1, width=32, num_blocks=4, scale=2)
    
    model.eval()

    x = torch.randn(1, 1, 128, 128) * 0.5 + 0.5
    print(f"input       : {tuple(x.shape)}  dtype={x.dtype}")
    print(f"input range : [{x.min().item():.4f}, {x.max().item():.4f}] (unbounded)")
    print(f"num_params  : {model.num_params:,}")

    with torch.no_grad():
        y = model(x)

    print(f"output      : {tuple(y.shape)}  dtype={y.dtype}")
    print(f"output range: [{y.min().item():.4f}, {y.max().item():.4f}] (clamped [0,1] in eval)")

    expected = (x.shape[0], 1, x.shape[2] * model.scale, x.shape[3] * model.scale)
    assert tuple(y.shape) == expected, f"expected {expected}, got {tuple(y.shape)}"
    assert torch.all(y >= 0.0) and torch.all(y <= 1.0), "output must be in [0, 1] during eval"
    print("PASS: (1,1,128,128) ->", tuple(y.shape), " [required (1,1,256,256)]")
