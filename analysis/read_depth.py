# depth_pseudocolor_turbo.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os

def depth_to_turbo_colormap(depth, vmin=None, vmax=None, invert=False, gamma=1.0):
    """
    将单通道深度图映射为 turbo 伪彩色（与 Matplotlib/Google turbo 相同）。
    参数:
      depth: numpy array, 单通道深度 (uint8, uint16, float32)
      vmin, vmax: 归一化范围 (None 表示自动取最小最大值)
      invert: 是否反转映射（近处亮 → 远处暗）
      gamma: 伽马校正 (默认1.0)
    返回:
      color_img: uint8 RGB 伪彩色图像 (H, W, 3)
      norm: 归一化的灰度图 [0,1]
    """
    depth = depth.astype(np.float32)
    depth = np.nan_to_num(depth)

    if vmin is None:
        vmin = np.percentile(depth, 1)
    if vmax is None:
        vmax = np.percentile(depth, 99)
    if vmax <= vmin:
        vmax = vmin + 1e-3

    norm = (depth - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    if invert:
        norm = 1.0 - norm
    if gamma != 1.0:
        norm = np.power(norm, gamma)

    cmap = plt.get_cmap('viridis')
    color_img = (cmap(norm)[..., :3] * 255).astype(np.uint8)

    return color_img, norm


def read_depth(path):
    """读取深度图文件（支持 .png, .npy）"""
    if path.lower().endswith('.npy'):
        depth = np.load(path)
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Cannot read depth file: {path}")
        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    return depth


def show_and_save(color_img, norm, out_path=None, title='Turbo Pseudocolor'):
    """显示结果并可选保存"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(title)
    ax[0].axis('off')

    im = ax[1].imshow(norm, cmap='viridis')
    ax[1].set_title('Normalized depth')
    ax[1].axis('off')
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
        print(f"Saved to {out_path}")

    #plt.show()


def main():
    parser = argparse.ArgumentParser(description="Turbo pseudocolor visualization for depth maps")
    parser.add_argument("depth_path", help="Input depth image (.png, .npy)")
    parser.add_argument("--out", "-o", help="Output file path (.png)")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--invert", action="store_true", help="Invert color mapping")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction")
    args = parser.parse_args()

    depth = read_depth(args.depth_path)
    color_img, norm = depth_to_turbo_colormap(depth,
                                              vmin=args.vmin,
                                              vmax=args.vmax,
                                              invert=args.invert,
                                              gamma=args.gamma)
    show_and_save(color_img, norm, args.out)


if __name__ == "__main__":

    depth_path = "scene0656_00/01708.png"

    path = depth_path.split('/')
    print(path)

    out_path = path[0].replace("scene", "depth_scannet_scene")

    os.makedirs(out_path, exist_ok=True)

    depth = read_depth(depth_path)

    color_img, norm = depth_to_turbo_colormap(depth)
    show_and_save(color_img, norm, out_path= os.path.join(out_path,path[-1]))


