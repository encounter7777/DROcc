import pandas as pd
import numpy as np

# 读取 Excel 文件
df = pd.read_excel("table2_15.xlsx")
df = df.fillna("")  # 填充空白单元格，防止报错

# 分组
grouped = df.groupby("input")

classes_16 = df.columns[2:]

classes_16 = ["\\rotatebox{90}{\colorbox{"+cl+"Color}{\\rule{0pt}{0.2em}\\rule{0.2em}{0pt}}"+cl+'}' for cl in classes_16]

# === 输出列表 ===
latex_lines = []
latex_lines.append("\\begin{table}[htbp]")
latex_lines.append("    \\centering")
latex_lines.append("    \\small")
# 自动根据列数生成 tabular 对齐方式
col_count = len(df.columns) - 1  # 不含 input
align_str = "l" + "c" * (col_count - 1)
latex_lines.append(f"    \\begin{{tabular}}{{{align_str}}}")
latex_lines.append("    \\toprule")
latex_lines.append("    Input"+ " & "+"Method & " + " & ".join(classes_16) + " \\\\")
latex_lines.append("    \\midrule")

for group_name, group_df in grouped:
    print(f"\\midrule  % {group_name}")
    latex_lines.append(f"% ---- {group_name} ----")

    # 数值列
    num_cols = group_df.columns[2:]

    # 尝试把这些列都转为 float，不可转换的设为 NaN
    numeric_df = group_df[num_cols].apply(pd.to_numeric, errors='coerce')

    # 计算每列最大值所在行索引
    max_idx = numeric_df.idxmax()

    for idx, row in group_df.iterrows():
        line = []
        line.append(row["method"].replace("_", "\\_"))  # 方法名（转义 _）
        line.append(group_name)
        for col in num_cols:
            val = row[col]
            try:
                fval = float(val)
                val_str = f"{fval:.2f}"
            except:
                val_str = str(val)

            # 若该行是该列最大值，则加粗
            if idx == max_idx[col]:
                val_str = f"\\textbf{{{val_str}}}"

            line.append(val_str)

        latex_line = " & ".join(line) + " \\\\"
        #print(latex_line)
        latex_lines.append(latex_line)

# === 表格结束 ===
latex_lines.append("    \\bottomrule")
latex_lines.append("    \\end{tabular}")
latex_lines.append("    \\caption{Quantitative comparison of semantic occupancy performance.}")
latex_lines.append("    \\label{tab:arch_results}")
latex_lines.append("\\end{table}")

# 输出结果到文件
with open("latex_table2_15.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(latex_lines))

print("\n✅ 已生成 LaTeX 表格代码到 latex_table.txt")
