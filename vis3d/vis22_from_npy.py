import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
from mayavi import mlab

mlab.options.offscreen = True  # 强制无窗口渲染

def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z]
    :param resolution: voxel size [rx, ry, rz]
    :return coords_grid: voxel center coords
    """
    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)

    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])
    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid


def visualize_from_npy(pred_npy, voxel_size=[0.2, 0.2, 0.2],
                       save_dir="./vis_png/",filter=False):
    """
    从 npy 文件加载 occupancy 并可视化保存 PNG
    Args:
        pred_npy (str): 保存的预测 npy 文件路径
        voxel_size (list): [x,y,z] voxel 尺寸
        save_dir (str): 保存 png 的目录
    """


    # 加载数据
    voxels = np.load(pred_npy)
    scene_id = os.path.basename(pred_npy).replace(".npy", "")

    methodname = os.path.basename(os.path.dirname(pred_npy))

    # 生成体素坐标
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]],
        voxel_size
    )

    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # 过滤非空体素
    # fov_voxels = grid_coords[
    #     (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 81)
    # ]

    display_list = [x for x in range(1,12)]+[29,14,15,28]+[71,79,78,73,68,77,74]

    if filter:
        gt_pred_npy = './gt/'+os.path.basename(pred_npy)
        gt_voxels = np.load(gt_pred_npy)
        gtcl_list = np.unique(gt_voxels)

        filter_list = gtcl_list[np.isin(gtcl_list, display_list)]

        print('filter_list: ', filter_list)

        #filter_list = [1,2,4,8,9,73,79]
        mask = np.isin(grid_coords[:, 3], filter_list)
    else:
        mask = np.isin(grid_coords[:, 3], display_list)
    fov_voxels = grid_coords[mask]

    #场景多少类
    unique_classes = np.unique(fov_voxels[:, 3])

    print('该场景有多少类:?',unique_classes)
    print('该场景有多少类在挑选的类中?： ',len(unique_classes))
    #重新映射
    class_to_index = {c: i for i, c in enumerate(display_list)}
    color_indices = np.array([class_to_index[c] for c in fov_voxels[:, 3]])

    #print(color_indices)

    # 开始绘制
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    voxel_size_avg = sum(voxel_size) / 3

    # plt_plot_fov = mlab.points3d(
    #     fov_voxels[:, 1],
    #     fov_voxels[:, 0],
    #     fov_voxels[:, 2],
    #     fov_voxels[:, 3],
    #     colormap="viridis",
    #     scale_factor=0.95 * voxel_size_avg,
    #     mode="cube",
    #     opacity=1.0,
    #     vmin=1,
    #     vmax=19,
    # )

    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 1],
        fov_voxels[:, 0],
        fov_voxels[:, 2],
        color_indices,
        scale_factor=0.95 * voxel_size_avg,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=len(display_list)-1,
    )


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
            [175, 0, 75, 255],     # refri.
            [139, 137, 137, 255],  # plant.
            [75, 0, 75, 255],      # stairs.
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


    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    scene = figure.scene


    # top_down view
    # 相机自适应
    #Saved: vis_matterport3d_JmbYfDe2QKZ_region15_vis.png
    scene.camera.position = [4.099999785423279, 3.9999998807907104, 17.174604780239665]
    scene.camera.focal_point = [4.099999785423279, 3.9999998807907104, 1.9999999701976776]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 1.0, 0.0]
    scene.camera.clipping_range = [12.64480883548178, 18.395698789620642]

    #Saved: vis_matterport3d_7y3sRwLe3Va_region12_vis.png
    # scene.camera.position = [3.9999999434221536, 4.0, 22.775831545898935]
    # scene.camera.focal_point = [3.9999999434221536, 4.0, 1.9999999701976776]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 1.0, 0.0]
    # scene.camera.clipping_range = [18.190023333484454, 24.080943956764806]

    #Saved: vis_matterport3d_29hnd4uzFmX_region9_vis.png
    # scene.camera.position = [3.5999999940395355, 3.4999999434221536, 19.89652788681168]
    # scene.camera.focal_point = [3.5999999940395355, 3.4999999434221536, 1.9999999701976776]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 1.0, 0.0]
    # scene.camera.clipping_range = [15.339512710988071, 21.15845074279124]

    #Saved: vis_matterport3d_29hnd4uzFmX_region5_vis.png
    # scene.camera.position = [3.9999998807907104, 3.9999998211860657, 16.486606495456655]
    # scene.camera.focal_point = [3.9999998807907104, 3.9999998211860657, 1.9999999701976776]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 1.0, 0.0]
    # scene.camera.clipping_range = [11.963690533546599, 17.697380530565887]

    #Saved: vis_scannet_scene0143_01_vis.png
    # scene.camera.position = [4.199999928474426, 3.9999999434221536, 20.579346733136415]
    # scene.camera.focal_point = [4.199999928474426, 3.9999999434221536, 1.899999976158142]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 1.0, 0.0]
    # scene.camera.clipping_range = [15.915503374810026, 22.203511850830008]

    #Saved: vis_scannet_scene0111_00_vis.png
    # scene.camera.position = [4.099999964237213, 3.7999998927116394, 18.218846170275903]
    # scene.camera.focal_point = [4.099999964237213, 3.7999998927116394, 1.9999999701976776]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 1.0, 0.0]
    # scene.camera.clipping_range = [13.678607811617654, 19.455603800507426]

    #Saved: vis_scannet_scene0040_00_vis.png
    # scene.camera.position = [3.5999999046325684, 3.9999998956918716, 20.65895633105022]
    # scene.camera.focal_point = [3.5999999046325684, 3.9999998956918716, 1.9999999701976776]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 1.0, 0.0]
    # scene.camera.clipping_range = [16.094316870784226, 21.93231561369336]

    #Saved: vis_scannet_scene0006_00_vis.png
    # scene.camera.position = [4.299999952316284, 3.9999999434221536, 21.183228161852266]
    # scene.camera.focal_point = [4.299999952316284, 3.9999999434221536, 1.9999999701976776]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 1.0, 0.0]
    # scene.camera.clipping_range = [16.613345983278254, 22.464451521957432]

    #Saved: vis_3rscan_0cac760f - 8d6f - 2d13 - 8d9d - 2d8df8f8cb6e_vis.png
    # scene.camera.position = [3.1999998539686203, 2.599999848054722, 17.992219013696683]
    # scene.camera.focal_point = [3.1999998539686203, 2.599999848054722, 1.9999999701976776]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 1.0, 0.0]
    # scene.camera.clipping_range = [13.454246926604226, 19.225577236579518]

    #Saved: vis_3rscan_0cac759d - 8d6f - 2d13 - 8e8b - c3730d9fe563_vis.png
    # scene.camera.position = [3.499999850988388, 4.0, 16.701601735230525]
    # scene.camera.focal_point = [3.499999850988388, 4.0, 1.899999976158142]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 1.0, 0.0]
    # scene.camera.clipping_range = [12.076535826883195, 18.26760067795553]

    # Saved: vis_3rscan_0a4b8ef6 - a83a - 21f2 - 8672 - dce34dd0d7ca_vis.png
    # scene.camera.position = [4.099999949336052, 4.799999952316284, 19.88431417913851]
    # scene.camera.focal_point = [4.099999949336052, 4.799999952316284, 1.899999976158142]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 1.0, 0.0]
    # scene.camera.clipping_range = [15.2274211463521, 21.498053808522137]

    # Saved: vis_3rscan_0ad2d38f - 79e2 - 2212 - 98d2 - 9b5060e5e9b5_vis.png
    # scene.camera.position = [2.8999999910593033, 3.2999998927116394, 15.887890243827641]
    # scene.camera.focal_point = [2.8999999910593033, 3.2999998927116394, 1.9999999701976776]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 1.0, 0.0]
    # scene.camera.clipping_range = [11.370961444433874, 17.08968353516244]

    # #vis_scannet_scene0136_00_vis.png
    # scene.camera.position = [4.199999928474426, 3.5999999046325684, 16.25914558835707]
    # scene.camera.focal_point = [4.199999928474426, 3.5999999046325684, 1.9999999701976776]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 1.0, 0.0]
    # scene.camera.clipping_range = [11.738504235518011, 17.466507709859812]




    #vis_scannet_scene0656_00.npy
    # scene.camera.position = [3.3999999910593033, 3.1999999911058694, 19.538861740070708]
    # scene.camera.focal_point = [3.3999999910593033, 3.1999999911058694, 1.9999999701976776]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 1.0, 0.0]
    # scene.camera.clipping_range = [14.98542322571451, 20.795419603849155]

    #vis_scannet_scene0605_00.npy

    # scene.camera.position = [3.9999999403953552, 4.099999949336052, 21.532051744505992]
    # scene.camera.focal_point = [3.9999999403953552, 4.099999949336052, 1.899999976158142]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 1.0, 0.0]
    # scene.camera.clipping_range = [16.858681336065906, 23.17050743737013]



    # scene.camera.position = [-0.588919997215271, -0.20418500900268555, 14.256451940914769]
    # scene.camera.focal_point = [-0.588919997215271, -0.20418500900268555, 5.256451940914769]
    # scene.camera.view_angle = 60.00
    # scene.camera.view_up = [-0.0, -1.0, -0.0]
    # scene.camera.clipping_range = [0.01, 200.0]




    scene.camera.compute_view_plane_normal()
    scene.render()

    save_dir = save_dir+scene_id

    os.makedirs(save_dir, exist_ok=True)
    # 保存 PNG
    if filter:
        save_path = os.path.join(save_dir, f"filter_{methodname}_{scene_id}.png")
    else:
        save_path = os.path.join(save_dir, f"{methodname}_{scene_id}.png")

    mlab.savefig(save_path)
    mlab.close()

    print(f"[INFO] Saved visualization to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", type=str,
                        help="Path to .npy occupancy file")
    parser.add_argument("--save_dir", type=str, default="./vis_png/",
                        help="Directory to save rendered PNG")
    parser.add_argument("--filter", type=bool, default=False,
                        help="filter")
    args = parser.parse_args()

    visualize_from_npy(args.npy_path, save_dir=args.save_dir, filter=args.filter)
