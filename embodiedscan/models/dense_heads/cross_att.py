import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ---------------------------
# Utils: window partition / reverse for 3D
# ---------------------------
def window_partition_3d(x: torch.Tensor, window_size: Tuple[int,int,int]):
    """
    x: (B, C, X, Y, Z)
    returns windows: (num_windows*B, C, Wx*Wy*Wz)
    and meta: (B, nx, ny, nz) - number of windows per dim
    """
    B, C, X, Y, Z = x.shape
    Wx, Wy, Wz = window_size
    assert X % Wx == 0 and Y % Wy == 0 and Z % Wz == 0, "Volume dims must be divisible by window size"
    nx, ny, nz = X // Wx, Y // Wy, Z // Wz
    x = x.view(B, C, nx, Wx, ny, Wy, nz, Wz)  # (B, C, nx, Wx, ny, Wy, nz, Wz)
    x = x.permute(0,2,4,6,1,3,5,7).contiguous()  # (B, nx, ny, nz, C, Wx, Wy, Wz)
    windows = x.view(B * nx * ny * nz, C, Wx * Wy * Wz)  # (num_windows*B, C, M)
    return windows, (nx, ny, nz)

def window_reverse_3d(windows: torch.Tensor, window_size: Tuple[int,int,int], volume_shape: Tuple[int,int,int], B: int):
    """
    windows: (num_windows*B, C, M)
    window_size: (Wx,Wy,Wz)
    volume_shape: (X,Y,Z)
    returns x: (B, C, X, Y, Z)
    """
    Wx, Wy, Wz = window_size
    X, Y, Z = volume_shape
    nx, ny, nz = X // Wx, Y // Wy, Z // Wz
    C = windows.shape[1]
    x = windows.view(B, nx, ny, nz, C, Wx, Wy, Wz)  # (B, nx, ny, nz, C, Wx, Wy, Wz)
    x = x.permute(0,4,1,5,2,6,3,7).contiguous()  # (B, C, nx, Wx, ny, Wy, nz, Wz)
    x = x.view(B, C, X, Y, Z)
    return x

# ---------------------------
# Cross-Modal Windowed Attention for 3D
# ---------------------------
class WindowCrossAttention3D(nn.Module):
    """
    Cross-attention inside 3D windows: Query from x_q, Key/Value from x_kv.
    Supports shifted windows, relative positional bias.
    """
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int,int,int] = (4,4,4),
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rel_pos: bool = True,
        shift: Tuple[int,int,int] = (0,0,0),
        linear_attn: bool = False,          # optional linear attention
        downsample_kv: Optional[Tuple[int,int,int]] = None,  # factor to downsample kv inside window
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wx,Wy,Wz)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert (self.head_dim * num_heads) == dim, "dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_rel_pos = use_rel_pos
        Wx, Wy, Wz = window_size
        M = Wx * Wy * Wz
        if use_rel_pos:
            # relative bias table: one per head, pairwise inside window
            self.relative_position_bias_table = nn.Parameter(torch.zeros((num_heads, M, M)))
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.shift = shift
        self.linear_attn = linear_attn
        self.downsample_kv = downsample_kv

    def forward(
        self,
        x_q: torch.Tensor,      # (B, C, X, Y, Z)
        x_kv: torch.Tensor,     # (B, C, X, Y, Z)
        mask: Optional[torch.Tensor] = None  # (num_windows, M, M) or None
    ):
        """
        Returns: out: (B, C, X, Y, Z)
        """
        B, C, X, Y, Z = x_q.shape
        Wx, Wy, Wz = self.window_size
        M = Wx * Wy * Wz

        # Optional cyclic shift (Swin)  —— implement shift by roll
        shift_x, shift_y, shift_z = self.shift
        if any([shift_x, shift_y, shift_z]):
            x_q = torch.roll(x_q, shifts=(-shift_x, -shift_y, -shift_z), dims=(2,3,4))
            x_kv = torch.roll(x_kv, shifts=(-shift_x, -shift_y, -shift_z), dims=(2,3,4))

        # Window partition (B*C*nx*ny*nz windows)
        q_windows, (nx, ny, nz) = window_partition_3d(x_q, self.window_size)  # (num_windows*B, C, M)
        kv_windows, _ = window_partition_3d(x_kv, self.window_size)

        # Optionally downsample KV inside each window (simple avg pooling across local groups)
        if self.downsample_kv is not None:
            # apply a local downsample inside each window: group M -> M'
            dsx, dsy, dsz = self.downsample_kv
            assert (Wx % dsx == 0 and Wy % dsy == 0 and Wz % dsz == 0), "downsample factors must divide window dims"
            # reshape kv_windows to (num_windows*B, C, Wx, Wy, Wz) then pool
            nv = kv_windows.shape[0]
            kv_resh = kv_windows.view(nv, C, Wx, Wy, Wz)
            kv_resh = kv_resh.unfold(2, dsx, dsx).unfold(3, dsy, dsy).unfold(4, dsz, dsz)
            # shape: nv, C, Wx/dsx, Wy/dsy, Wz/dsz, dsx, dsy, dsz
            nv2, C2, gx, gy, gz, dx, dy, dz = kv_resh.shape
            kv_resh = kv_resh.contiguous().view(nv2, C2, gx, gy, gz, -1)  # merge small block
            kv_resh = kv_resh.mean(-1)  # average inside block
            # now flatten spatial -> M'
            kv_windows = kv_resh.view(nv2, C2, gx * gy * gz)
            # note: this reduces M, we must accordingly adjust K,V projections

        # prepare Q,K,V
        # (num_windows*B, M, C) for Q and K,V
        q = q_windows.permute(0,2,1).contiguous()  # (NwB, M, C)
        k = kv_windows.permute(0,2,1).contiguous()
        v = kv_windows.permute(0,2,1).contiguous()

        # linear projections
        q = self.q_proj(q)  # (NwB, M, C)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # reshape for multi-head: (NwB, num_heads, M, head_dim)
        NwB = q.shape[0]
        q = q.view(NwB, M, self.num_heads, self.head_dim).permute(0,2,1,3)  # (NwB, H, M, d)
        k = k.view(NwB, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.view(NwB, -1, self.num_heads, self.head_dim).permute(0,2,1,3)

        # scaled dot-product attention
        if not self.linear_attn:
            # classic softmax attention
            attn = torch.matmul(q, k.transpose(-2,-1))  # (NwB, H, M, M)
            attn = attn * self.scale
            if self.use_rel_pos:
                # add relative bias: (H, M, M) -> broadcast to (NwB, H, M, M)
                attn = attn + self.relative_position_bias_table.unsqueeze(0)
            if mask is not None:
                # mask shape: (num_windows, M, M), Nw = num_windows*B, we tile mask for each sample
                # assume mask has num_windows = nx*ny*nz
                num_win = nx * ny * nz
                # mask: (num_win, M, M) -> expand to (B*num_win, 1, M, M)
                attn = attn.view(B, num_win, self.num_heads, M, M)
                attn = attn + mask.unsqueeze(2)  # mask broadcast
                attn = attn.view(NwB, self.num_heads, M, M)
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v)  # (NwB, H, M, d)
        else:
            # simple linear attention (feature map kernel), a fast but approximate version
            # q,k: (NwB,H,M,d) -> map to (NwB,H,M,d) after elu+1
            qk = (F.elu(q) + 1)
            kk = (F.elu(k) + 1)
            # compute kv = (kk^T * v) per head, but careful with shapes
            # compute denominator: qk @ kk.sum(dim=2).transpose? We'll do per-window per-head compute
            # WARNING: Approximate and may be numerically unstable; used only for large M
            kv = torch.einsum('nhmd,nhme->nhde', kk, v)  # (NwB, H, d, d)
            denom = torch.einsum('nhmd,nhd->nhm', qk, kk.sum(dim=2))  # (NwB, H, M)
            out = torch.einsum('nhmd,nhde->nhme', qk, kv)  # (NwB,H,M,d)
            out = out / (denom.unsqueeze(-1) + 1e-6)

        # combine heads
        out = out.permute(0,2,1,3).contiguous().view(NwB, M, self.dim)  # (NwB, M, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # revert windows: (NwB, C, M) -> (B, C, X, Y, Z)
        out = out.permute(0,2,1).contiguous()  # (NwB, C, M)
        out = window_reverse_3d(out, self.window_size, (X, Y, Z), B)

        # reverse cyclic shift
        if any([shift_x, shift_y, shift_z]):
            out = torch.roll(out, shifts=(shift_x, shift_y, shift_z), dims=(2,3,4))

        return out

# ---------------------------
# Example wrapper: Cross-Attention Swin Block (can be used inside UNet)
# ---------------------------
class CrossSwin3DBlock(nn.Module):
    """
    A Swin-like block where self-attention is replaced by cross-attention:
    Query comes from x_q, Key/Value from x_kv. Followed by MLP and residuals.
    """
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int,int,int] = (4,4,4),
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_rel_pos: bool = True,
        shift: Tuple[int,int,int] = (0,0,0),
    ):
        super().__init__()
        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_kv = nn.LayerNorm(dim)
        self.attn = WindowCrossAttention3D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_rel_pos=use_rel_pos,
            shift=shift
        )
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor):
        """
        x_q, x_kv: (B, C, X, Y, Z)
        returns: fused (B, C, X, Y, Z)
        """
        B, C, X, Y, Z = x_q.shape
        # LayerNorm expects (B, L, C) or apply per-voxel: so permute
        x_q_ln = self.norm1_q(x_q.permute(0,2,3,4,1)).permute(0,4,1,2,3)  # (B,C,X,Y,Z)
        x_kv_ln = self.norm1_kv(x_kv.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        attn_out = self.attn(x_q_ln, x_kv_ln)
        x = x_q + self.drop_path(attn_out)
        # MLP
        x_ln2 = self.norm2(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        x = x + self.drop_path(self.mlp(x_ln2.permute(0,2,3,4,1)).permute(0,4,1,2,3))
        return x




if __name__ == "__main__":
    # ====== 单元测试 ======
    def test_cross_attention_3d_forward():
        torch.manual_seed(0)
        b, c, X, Y, Z = 1, 256, 40, 40, 16

        x = torch.randn(b, c, X, Y, Z).cuda()
        y = torch.randn(b, c, X, Y, Z).cuda()

        model = CrossSwin3DBlock(dim=c, num_heads=4).cuda()
        out = model(x, y)

        # 输出形状检查
        assert out.shape == (b, c, X, Y, Z)

        # 值的范围合理性检查
        assert not torch.isnan(out).any()
        assert torch.isfinite(out).all()

        print("CrossAttention3D forward pass successful! Output shape:", out.shape)
    test_cross_attention_3d_forward()