import numpy as np
import os
from mayavi import mlab


class_names = ('empty','floor', 'wall', 'chair', 'cabinet', 'door', 'table', 'couch',
               'shelf', 'window', 'bed', 'curtain', 'desk', 'doorframe',
               'plant', 'stairs', 'pillow', 'wardrobe', 'picture', 'bathtub',
               'box', 'counter', 'bench', 'stand', 'rail', 'sink', 'clothes',
               'mirror', 'toilet', 'refrigerator', 'lamp', 'book', 'dresser',
               'stool', 'fireplace', 'tv', 'blanket', 'commode',
               'washing machine', 'monitor', 'window frame', 'radiator', 'mat',
               'shower', 'rack', 'towel', 'ottoman', 'column', 'blinds',
               'stove', 'bar', 'pillar', 'bin', 'heater', 'clothes dryer',
               'backpack', 'blackboard', 'decoration', 'roof', 'bag', 'steps',
               'windowsill', 'cushion', 'carpet', 'copier', 'board',
               'countertop', 'basket', 'mailbox', 'kitchen island',
               'washbasin', 'bicycle', 'drawer', 'oven', 'piano',
               'exercise equipment', 'beam', 'partition', 'printer',
               'microwave', 'frame')


def get_grid_coords(dims, resolution):
    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])
    coords_grid = (coords_grid * resolution) + resolution / 2
    return coords_grid


def visualize_from_npy_interactive(pred_npy, voxel_size=[0.2, 0.2, 0.2]):
    # 加载数据
    voxels = np.load(pred_npy)
    scene_id = os.path.basename(pred_npy).replace(".npy", "")

    # 生成体素坐标
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]],
        voxel_size
    )
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # # 过滤非空体素
    # fov_voxels = grid_coords[
    #     (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 20)
    # ]
    #
    # # 开始绘制
    # figure = mlab.figure(size=(1280, 720), bgcolor=(1, 1, 1))
    # voxel_size_avg = sum(voxel_size) / 3
    #
    # plt_plot_fov = mlab.points3d(
    #     fov_voxels[:, 1],  # y
    #     fov_voxels[:, 0],  # x
    #     fov_voxels[:, 2],  # z
    #     fov_voxels[:, 3],  # label
    #     scale_factor=0.95 * voxel_size_avg,
    #     mode="cube",
    #     opacity=1.0,
    #     vmin=1,
    #     vmax=19,
    # )

    display_list = ([x for x in range(1,12)] + [29,14,15,28]
                    +[71,79,78,73,68,77,74])
    mask = np.isin(grid_coords[:, 3], display_list)
    fov_voxels = grid_coords[mask]

    #场景多少类
    unique_classes = np.unique(fov_voxels[:, 3])

    print(unique_classes)

    for cls in unique_classes:
        print(class_names[int(cls)])

    print('该场景有多少类在挑选的类中?： ',len(unique_classes))
    #重新映射
    class_to_index = {c: i for i, c in enumerate(display_list)}
    color_indices = np.array([class_to_index[c] for c in fov_voxels[:, 3]])

    # 开始绘制
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    voxel_size_avg = sum(voxel_size) / 3

    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 1],
        fov_voxels[:, 0],
        fov_voxels[:, 2],
        color_indices,
        scale_factor=0.95 * voxel_size_avg,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=21,
    )

    # 使用离屏版相同的 scale_mode
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"

    # 自定义颜色表
    common_colors = np.array(
        [
            [255, 120, 50, 255],   # floor
            [255, 192, 203, 255],  # wall
            [255, 255, 0, 255],    # chair
            [0, 150, 245, 255],    # cabinet
            [0, 255, 255, 255],    # door
            [0, 175, 0, 255],      # table
            [255, 0, 0, 255],      # couch
            [127, 127, 127, 255],  # shelf
            [135, 60, 0, 255],     # window
            [160, 32, 240, 255],   # bed
            [255, 0, 255, 255],    # curtain
            [175, 0, 75, 255],     # refrigerator
            [139, 137, 137, 255],  # plant
            [75, 0, 75, 255],      # stairs
            [150, 240, 80, 255],   # toilet
        ]
    ).astype(np.uint8)

    extra_colors = np.array([

        [222, 150, 255, 255], #bicycle

        [255, 200, 50, 255], #microwave
        [230, 230, 250, 255], #printer
        [214, 255, 50, 255], #oven
        [255, 50, 173, 255], #mailbox

        [50, 55, 255, 255], #partition
        [150, 182, 255, 255], #piano

        #drawer
        #carpet

    ]).astype(np.uint8)

    colors = np.vstack([common_colors, extra_colors])

    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    # 设置默认相机（和离屏版一致）
    scene = figure.scene
    scene.camera.position = [5.8892, 2.0419, 20.25645]
    scene.camera.focal_point = [-0.58892, -0.20419, 5.25645]
    scene.camera.view_angle = 60.0
    scene.camera.view_up = [-0.0, -1.0, -0.0]
    scene.camera.clipping_range = [0.01, 200.0]
    scene.camera.compute_view_plane_normal()


    # 打开交互窗口，用户可旋转缩放
    mlab.show()

    mlab.savefig(f"./gt/{scene_id}_vis.png")
    # ---- 打印相机参数 ----
    print(f"Saved: {scene_id}_vis.png")
    print("scene.camera.position =", list(scene.camera.position))
    print("scene.camera.focal_point =", list(scene.camera.focal_point))
    print("scene.camera.view_angle =", scene.camera.view_angle)
    print("scene.camera.view_up =", list(scene.camera.view_up))
    print("scene.camera.clipping_range =", list(scene.camera.clipping_range))
    mlab.close(all=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", type=str, help="Path to .npy occupancy file")
    args = parser.parse_args()

    visualize_from_npy_interactive(args.npy_path)
