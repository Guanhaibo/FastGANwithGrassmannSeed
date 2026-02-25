import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from dataclasses import dataclass
from typing import Tuple, Union, Sequence

@dataclass
class GConfig:
    z_dim: int = 256
    field_res: int = 16
    K: int = 16
    num_blocks: int = 2
    ch_256: int = 64
    ch_512: int = 32
    
# ============================================================
# 2. 生成器 (Generator) 模块
# ============================================================

class GrassmannSeed(nn.Module):
    """
    通过流形投影生成初始特征图 G0
    """
    def __init__(self, z_dim: int, K: int, rank: int, field_res: int):
        super().__init__()
        self.K, self.rank = K, rank
        self.fc = nn.Linear(z_dim, K * rank)
        self.atoms = nn.Parameter(torch.randn(rank, field_res, field_res) * 0.05)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        # 生成映射矩阵并进行 QR 分解确保正交性
        w = self.fc(z).view(B, self.K, self.rank)
        # 稳定性修复：防止 W 为全零导致 QR 崩溃
        w = w + 1e-6 
        q, _ = torch.linalg.qr(w.float(), mode="reduced")
        q = q.to(z.dtype)
        
        # 投影到空间基底 [B, K, H, W]
        g0 = torch.einsum("bkr,rhw->bkhw", q, self.atoms.to(z.dtype))
        return g0,q

class Evo1(nn.Module):
    """
    几何演化块：处理 4K 通道 (标量、向量、双向量)
    """
    def __init__(self, channels: int, blade_groups: int = 4):
        super().__init__()
        self.act = nn.LeakyReLU(0.2)
        # 深度卷积捕获空间上下文
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        # 分组卷积模拟几何分量交互
        self.proj = nn.Conv2d(channels, channels, 1, groups=blade_groups)
        self.gate = nn.Conv2d(channels, channels, 1, groups=blade_groups)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.act(self.dw(x))
        # 模拟几何交互
        out = self.proj(res) 
        gate = torch.sigmoid(self.gate(x))
        return x + self.gamma * (self.act(x) + gate * out)



class Evo2(nn.Module):
    """
    roll-CAN + GGR evolution block (2D)
    Clifford Block
    Input :
        X: [B, D, H, W], where D = 4*K   (blade-stacked channels)
    Output:
        Y: [B, D, H, W]  (same shape)
    """
    def __init__(
        self,
        K: int,
        shifts: Sequence[int] = (1, 2),     # start small for speed/stability
        blade_aligned: bool = True,         # shift by *4 to reduce blade mixing (only if layout supports it)
        dw_depth: int = 2,                  # depthwise conv layers in ctx net
        gate_hidden_mul: float = 1.0,       # gate hidden width multiplier
        gamma_init: float = 1e-3,           # small init for stability
        use_dot: bool = True,
        use_wedge: bool = True,
        proj_groups: int = 1,               # set 4 if you want blade-wise grouped proj
    ):
        super().__init__()

        self.K = int(K)
        self.D = 4 * self.K
        self.shifts = tuple(int(s) for s in shifts)
        self.blade_aligned = bool(blade_aligned)

        assert use_dot or use_wedge, "At least one of dot/wedge must be enabled."
        self.use_dot = bool(use_dot)
        self.use_wedge = bool(use_wedge)

        # LN-like (stable for small batch)
        self.norm = nn.GroupNorm(1, self.D)

        # content stream
        self.det = nn.Conv2d(self.D, self.D, 1, 1, 0, bias=True)

        # learned context stream (depthwise stack)
        ctx_layers = []
        for _ in range(int(dw_depth)):
            ctx_layers.append(nn.Conv2d(self.D, self.D, 3, 1, 1, groups=self.D, bias=False))
            ctx_layers.append(nn.SiLU(inplace=True))
        ctx_layers.append(nn.Conv2d(self.D, self.D, 1, 1, 0, bias=True))
        self.ctx = nn.Sequential(*ctx_layers)

        # fuse multi-shift (dot/wedge) features -> g
        per_shift = (self.D if self.use_dot else 0) + (self.D if self.use_wedge else 0)
        in_proj = per_shift * len(self.shifts)
        self.proj = nn.Conv2d(in_proj, self.D, 1, 1, 0, bias=True, groups=int(proj_groups))

        # gate([x, g]) -> a
        gate_hid = max(16, int(self.D * gate_hidden_mul))
        self.gate = nn.Sequential(
            nn.Conv2d(self.D * 2, gate_hid, 1, 1, 0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hid, self.D, 1, 1, 0, bias=True),
        )

        # LayerScale gamma (per-channel)
        self.gamma = nn.Parameter(torch.full((1, self.D, 1, 1), float(gamma_init)))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, D, H, W]
        x0 = X                               # [B, D, H, W]
        x  = self.norm(X)                    # [B, D, H, W]

        det = F.silu(self.det(x))            # [B, D, H, W]
        ctx = self.ctx(x)                    # [B, D, H, W]

        feats = []
        for s in self.shifts:
            sh = s * 4 if self.blade_aligned else s

            ctx_s = torch.roll(ctx, shifts=sh, dims=1)  # [B, D, H, W]
            det_s = torch.roll(det, shifts=sh, dims=1)  # [B, D, H, W]

            if self.use_dot:
                dot = F.silu(det * ctx_s)               # [B, D, H, W]
                feats.append(dot)

            if self.use_wedge:
                wedge = (det * ctx_s) - (ctx * det_s)   # [B, D, H, W]
                feats.append(wedge)

        feats_cat = torch.cat(feats, dim=1)             # [B, in_proj, H, W]
        g = self.proj(feats_cat)                        # [B, D, H, W]

        xg = torch.cat([x, g], dim=1)                   # [B, 2D, H, W]
        a = torch.sigmoid(self.gate(xg))                # [B, D, H, W]

        h = det + a * g                                 # [B, D, H, W]
        return x0 + self.gamma * h                      # [B, D, H, W]


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
        
class GCSE(nn.Module):
    # cond: [B, K, Hc, Wc]  (g0)
    # feat: [B, C, H, W]
    def __init__(self, K, C, pool=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool),      # -> [B,K,pool,pool]
            nn.Conv2d(K, C, pool, 1, 0),     # -> [B,C,1,1]
            nn.ReLU(),
            nn.Conv2d(C, C, 1, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, cond, feat):
        return feat * self.net(cond)

class QFiLM(nn.Module):
    """
    q-conditioned FiLM (AdaIN-lite, no normalization)
    x: [B, C, H, W]
    q: [B, K, r]
    """
    def __init__(self, K: int, C: int, r: int, hidden_mul: float = 2.0,
                 mode: str = "mean", gamma_scale: float = 0.1):
        super().__init__()
        self.K = K
        self.C = C
        self.r = r
        self.mode = mode
        self.gamma_scale = gamma_scale

        if mode == "mean":
            in_dim = K
        elif mode == "flat":
            in_dim = K * r
        else:
            raise ValueError("mode must be 'mean' or 'flat'")

        hid = max(32, int(in_dim * hidden_mul))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.SiLU(inplace=True),
            nn.Linear(hid, 2 * C)
        )

        # init: start near identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.C

        if self.mode == "mean":
            s = q.mean(dim=2)               # [B, K]
        else:
            s = q.reshape(B, self.K * self.r)  # [B, K*r]

        gb = self.mlp(s)                    # [B, 2C]
        gamma, beta = gb[:, :C], gb[:, C:]  # [B, C], [B, C]

        gamma = self.gamma_scale * gamma    # keep modulation small/stable
        gamma = gamma.view(B, C, 1, 1)
        beta  = beta.view(B, C, 1, 1)

        return x * (1.0 + gamma) + beta

class UpBlockLite(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # PixelShuffle(2) 需要输出通道 = out_ch * 4
        self.upconv = nn.Conv2d(in_ch, out_ch * 4, 3, 1, 1, bias=True,padding_mode='reflect')
        self.ps = nn.PixelShuffle(2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.ps(self.upconv(x)))



class conv3x3(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1x1_1 = nn.Conv2d(c, 32 , 1)
        self.c1x3 = nn.Conv2d(32, 32, kernel_size=(1,3), padding=(0,1),padding_mode='reflect')
        self.c3x1 = nn.Conv2d(32, 32, kernel_size=(3,1), padding=(1,0),padding_mode='reflect')
        self.c1x1_2 = nn.Conv2d(32 ,c ,1)
        self.R = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return self.R(self.c1x1_2(self.c3x1(self.R(self.c1x3(self.R(self.c1x1_1(self.R(x))))))))+x

class Generator(nn.Module):
    """
    GrassmannSeed + Evo1/Evo2(For Fake Clifford Nets) + (QFiLM & GCSE) + tiny decoder
    Output:
      rgb1024: [B,3,1024,1024]
      rgb128 : [B,3,128,128]
    """
    def __init__(self, cfg: GConfig, rank: int = 16):
        super().__init__()
        self.cfg = cfg
        self.rank = rank

        K = cfg.K
        C0 = 4 * K  # blade-stacked channels
        self.rgb_gain = nn.Parameter(torch.tensor(5.0))
        self.loss_q = nn.Parameter(torch.tensor(5.0))
        # --- seed ---
        self.seed = GrassmannSeed(cfg.z_dim, K, rank=rank, field_res=cfg.field_res)
        self.lift = nn.Conv2d(K, C0, 1, 1, 0, bias=True)  # [B,K,128,128] -> [B,4K,128,128]

        # --- evolution ---
        self.evo1 = nn.Sequential(*[Evo1(C0) for _ in range(cfg.num_blocks)])
        self.evo2_1 = nn.Sequential(Evo2(K=K, shifts=(2, 3), blade_aligned=True),
                                  Evo2(K=K, shifts=(1, 3), blade_aligned=True))
        
        self.evo2_2 = nn.Sequential(Evo2(K=K, shifts=(1, 3), blade_aligned=True),
                                  Evo2(K=K, shifts=(1, 2), blade_aligned=True))        # --- q-conditioned modulation at 128 ---
        self.qfilm_128 = QFiLM(K=K, C=C0, r=rank, mode="mean", gamma_scale=0.3)

        # --- upsample decoder ---
        self.up256 = UpBlockLite(C0, cfg.ch_256)          # 128 -> 256
        self.up512 = UpBlockLite(cfg.ch_256, cfg.ch_512)  # 256 -> 512

        # --- grassmann-conditioned gates (use g0 as cond) ---
        self.gcse_256 = GCSE(K=K, C=cfg.ch_256, pool=4)
        self.gcse_512 = GCSE(K=K, C=cfg.ch_512, pool=4)
        self.refine256 = nn.Sequential(
            nn.Conv2d(cfg.ch_256, cfg.ch_256, 3, padding=1,padding_mode='reflect'),
            conv3x3(cfg.ch_256),
            nn.Conv2d(cfg.ch_256, cfg.ch_256, 3, padding=1,padding_mode='reflect'),
            conv3x3(cfg.ch_256),
        )

        self.to_rgb_512 = nn.Conv2d(cfg.ch_512, 3, 1, 1, 0, bias=True)
        self.to_rgb_256 = nn.Conv2d(cfg.ch_256, 3, 1, 1, 0, bias=True)
    def forward(self, z: torch.Tensor):
        # z: [B, z_dim]
        g0, q = self.seed(z)                 # g0: [B,K,128,128], q:[B,K,rank]
        #ch_energy = g0.pow(2).mean(dim=(2,3))          # [B,K]
        #loss_energy = ch_energy.std(dim=1).mean()      # 越小越均匀
        feat = self.lift(g0)                 # feat: [B,4K,128,128]

        #feat = self.evo1(feat)               # [B,4K,128,128]
        feat = self.evo2_1(feat)               # [B,4K,128,128]
        feat = self.qfilm_128(feat, q)       # [B,4K,128,128]  (q-conditioned FiLM)
        feat = self.evo2_2(feat)
        
        feat_256 = self.up256(feat)          # [B,ch_256,256,256]
        
        feat_256 = self.gcse_256(g0, feat_256)  # gate with g0 -> [B,ch_256,256,256
        
        feat_256 = self.refine256(feat_256)
        rgb_256 = self.to_rgb_256(feat_256)
        rgb_256 = self.rgb_gain * rgb_256

        
        return torch.tanh(rgb_256)#,loss_energy*self.loss_q