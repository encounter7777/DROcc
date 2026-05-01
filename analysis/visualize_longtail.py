import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import re

# 解析频数数据文件
def parse_freq_data(freq_file):
    data = []
    with open(freq_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                id_ = int(parts[0])
                for i in range(1, len(parts)):
                    try:
                        count = int(parts[i])
                        class_name = ' '.join(parts[1:i])
                        freq = float(parts[i+1])
                        data.append({
                            'id': id_,
                            'name': class_name,
                            'count': count,
                            'freq': freq
                        })
                        break
                    except ValueError:
                        continue
    return data

# 生成81类长尾横向柱状图（参考longtail.png样式）
def plot_longtail_line_chart(data, output_file='longtail_line_chart.png'):
    # 反转数据顺序，使最大值在顶部
    data_reversed = data[::-1]
    names = [d['name'] for d in data_reversed]
    counts = [d['count'] for d in data_reversed]

    fig, ax = plt.subplots(figsize=(8, 12))

    # 根据count值创建紫色渐变颜色
    n = len(names)
    max_count = max(counts)
    min_count = min(counts)

    # 创建紫色colormap
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    import matplotlib.cm as cm

    # 紫色渐变：更浅到更深（范围更宽）
    purple_cmap = LinearSegmentedColormap.from_list('purple_gradient',
        [(0.95, 0.95, 1.0), (0.35, 0.15, 0.55)])

    # 根据count值映射颜色（对数映射）
    log_counts = np.log10(np.array(counts) + 1)
    log_min, log_max = np.log10(min_count + 1), np.log10(max_count + 1)
    normalized = (log_counts - log_min) / (log_max - log_min)
    colors = [purple_cmap(val) for val in normalized]

    # 绘制横向柱状图，height=1.0无间隔
    bars = ax.barh(range(len(names)), counts, color=colors, edgecolor='none', height=1.0)

    # 设置X轴为对数刻度
    ax.set_xscale('log')
    ax.set_xlim(1, max(counts) * 2)

    # 设置Y轴，无间隔紧密排列
    ax.set_ylim(-0.5, len(names) - 0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)

    # 将X轴移到顶部
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # 在第一个柱形图内部添加标题（白色）
    ax.text(10**3.5, len(names) - 1.5, 'Number of Instances for Each Category',
            fontsize=12, color='white', fontweight='bold',
            ha='center', va='center')

    # 移除下边框和右边框
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加colorbar
    sm = cm.ScalarMappable(cmap=purple_cmap, norm=LogNorm(vmin=min_count, vmax=max_count))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.3, aspect=10, pad=-0.05,
                        anchor=(0.0, 0.0), location='right')
    cbar.set_label('Count', fontsize=12, rotation=-90, labelpad=15)
    cbar.ax.yaxis.set_label_position('left')
    # 旋转colorbar刻度标签，从上往下读
    cbar.ax.tick_params(labelsize=10, rotation=-90)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存长尾柱状图到: {output_file}")

if __name__ == '__main__':
    freq_data = parse_freq_data('frep.txt')

    print(f"共读取 {len(freq_data)} 个类别")

    plot_longtail_line_chart(freq_data, output_file='longtail_line_chart.png')

    print("\n长尾柱状图生成完成！")
