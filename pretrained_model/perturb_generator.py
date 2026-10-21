import torch
from torch import nn
from typing import Optional


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return (1 + gamma) * x + beta


class AdvGANResnetBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: Optional[int] = None):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
        )
        self.film = FiLM(cond_dim, channels) if cond_dim is not None else None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.conv_block(x)
        if self.film is not None:
            out = self.film(out, cond)
        return x + out


class AdvGANPerturbationGenerator(nn.Module):
    """AdvGAN-style ResNet generator with target-text and epsilon conditioning.

    The architecture follows the common AdvGAN/CycleGAN generator pattern:
    encoder, residual bottleneck, decoder, tanh output. The tanh output is
    scaled by a per-sample epsilon so the generated perturbation satisfies
    the L-infinity budget.
    """

    def __init__(
        self,
        image_channels: int = 3,
        target_dim: int = 768,
        base_channels: int = 64,
        num_res_blocks: int = 4,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks
        self.cond_dim = cond_dim
        self.conditioner = nn.Sequential(
            nn.Linear(target_dim + 1, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(inplace=True),
        )
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(image_channels, base_channels, kernel_size=7, bias=False),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=False),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList(
            [AdvGANResnetBlock(base_channels * 4, cond_dim=cond_dim) for _ in range(num_res_blocks)]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(base_channels * 2, affine=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(base_channels, affine=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, image_channels, kernel_size=7),
        )

    def forward(
        self,
        image: torch.Tensor,
        target_embedding: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        if epsilon.ndim == 0:
            epsilon = epsilon.reshape(1, 1).expand(image.shape[0], 1)
        elif epsilon.ndim == 1:
            epsilon = epsilon[:, None]
        epsilon = epsilon.to(device=image.device, dtype=image.dtype)
        target_embedding = target_embedding.to(device=image.device, dtype=image.dtype)
        if target_embedding.ndim == 1:
            target_embedding = target_embedding[None, :].expand(image.shape[0], -1)
        cond = self.conditioner(torch.cat([target_embedding, epsilon], dim=1))
        h = self.encoder(image)
        for block in self.res_blocks:
            h = block(h, cond)
        raw_delta = self.decoder(h)
        return torch.tanh(raw_delta) * epsilon[:, :, None, None]


class PerturbationGenerator(nn.Module):
    """A small image-conditioned generator for initializing visual perturbations."""

    def __init__(self, channels: int = 3, hidden_channels: int = 32):
        super().__init__()
        group_count = min(8, hidden_channels)
        while hidden_channels % group_count != 0:
            group_count -= 1
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(group_count, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(group_count, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(group_count, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, epsilon: float) -> torch.Tensor:
        return self.net(image) * epsilon


def get_attack_frame(input_x: torch.Tensor, model_name: str) -> torch.Tensor:
    if model_name == "open_flamingo":
        return input_x[:, -1, 0]
    return input_x


def format_attack_noise(delta: torch.Tensor, model_name: str) -> torch.Tensor:
    if model_name == "open_flamingo":
        return delta.unsqueeze(1)
    return delta


def load_generator_checkpoint(
    checkpoint_path: str,
    device: str,
    hidden_channels: int = 32,
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and checkpoint.get("generator_type") == "advgan_resnet_film":
        generator = AdvGANPerturbationGenerator(
            target_dim=checkpoint.get("target_dim", 768),
            base_channels=checkpoint.get("base_channels", 64),
            num_res_blocks=checkpoint.get("num_res_blocks", 4),
            cond_dim=checkpoint.get("cond_dim", 256),
        ).to(device)
    else:
        generator = PerturbationGenerator(hidden_channels=hidden_channels).to(device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator
