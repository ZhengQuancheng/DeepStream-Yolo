import os
import re
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_perf(df: pd.DataFrame):
    # streams 列为数值类型并排序
    df['streams'] = pd.to_numeric(df['streams'])
    df.sort_values(by='streams', inplace=True)
    # 设置 Seaborn 主题
    sns.set_theme(style="white")
    # 获取所有分辨率
    resolutions = df['resolution'].unique()
    # 为每个分辨率绘制单独的图表
    for res in resolutions:
        # 筛选当前分辨率的数据
        subset = df[df['resolution'] == res]
        if subset.empty:
            continue
        # 创建图表和轴
        fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
        plt.title(f"Resolution: {res}", fontsize=16, fontweight='bold', pad=15)

        # --- 左轴 (平均 FPS) ---
        # 刻度显示到 5
        fps_ticks = [0, 1, 2, 3, 4, 5]
        ax1.set_yticks(fps_ticks)
        # 范围设为 0 到 5.5
        ax1.set_ylim(0, 5.5)

        # --- 右轴 (百分比) ---
        ax2 = ax1.twinx()
        # 刻度显示到 100
        pct_ticks = [0, 20, 40, 60, 80, 100]
        ax2.set_yticks(pct_ticks)
        # 范围设为 0 到 110
        ax2.set_ylim(0, 110)

        # 开启网格
        ax1.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray')
        ax2.grid(False)

        # 绘制左轴数据
        color_fps = '#1f77b4'
        ax1.set_xlabel('Number of Streams', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average FPS', color=color_fps, fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color_fps)

        sns.lineplot(data=subset, x='streams', y='avg_fps', ax=ax1,
                     color=color_fps, marker='o', linewidth=3,
                     label='Avg FPS', legend=False)

        # 绘制右轴数据
        ax2.set_ylabel('Usage Percentage (%)', color='#333333', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#333333')

        sns.lineplot(data=subset, x='streams', y='avg_sm', ax=ax2,
                     color='#d62728', marker='s', linestyle='--',
                     label='GPU SM', legend=False)

        sns.lineplot(data=subset, x='streams', y='avg_dec', ax=ax2,
                     color='#ff7f0e', marker='^', linestyle='--',
                     label='Decoder', legend=False)

        sns.lineplot(data=subset, x='streams', y='avg_mem', ax=ax2,
                     color='#2ca02c', marker='v', linestyle=':',
                     label='Memory', legend=False)

        # 幽灵线策略
        ghost_sm  = subset['avg_sm'] / 20
        ghost_dec = subset['avg_dec'] / 20
        ghost_mem = subset['avg_mem'] / 20
        sns.lineplot(x=subset['streams'], y=ghost_sm, ax=ax1, alpha=0, legend=False)
        sns.lineplot(x=subset['streams'], y=ghost_dec, ax=ax1, alpha=0, legend=False)
        sns.lineplot(x=subset['streams'], y=ghost_mem, ax=ax1, alpha=0, legend=False)

        # 合并图例
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        final_lines = lines_1 + lines_2
        final_labels = labels_1 + labels_2

        # 绘制图例
        ax1.legend(final_lines, final_labels, loc='best',
                   fontsize=10, frameon=True, framealpha=0.9, fancybox=True)

        # 保存为图片
        png_file = f"perf_{res}.png"
        plt.savefig(png_file, dpi=500)
        print(f"[Info] Saved plot to {png_file}")

        # 关闭图表
        plt.close()

# 清洗数据，去掉开头 ratio 比例的数据
def clean_data(df: pd.DataFrame, ratio=0.1):
    if df.empty:
        return df
    cut_idx = int(len(df) * ratio)
    if cut_idx > 0 and cut_idx < len(df):
        df = df.iloc[cut_idx:].reset_index(drop=True)
    return df

# 解析 app 日志
def parse_app_log(app_log: str, streams: int) -> pd.DataFrame:
    # 按行分割
    lines = app_log.strip().split('\n')
    # 筛选出以 '**PERF:' 开头且不包含 'FPS' 的行
    lines = [line for line in lines if line.startswith('**PERF:') and 'FPS' not in line]
    # 提取 FPS 数据
    data = []
    for line in lines:
        # 提取括号内的平均 FPS 值
        avg_fps = re.findall(r'\(([\d\.]+)\)', line)
        try:
            avg_fps = [float(fps) for fps in avg_fps]
        except ValueError:
            continue
        # 如果平均 FPS 中有超过四分之一是 0, 则跳过该行
        zero_count = avg_fps.count(0.0)
        if zero_count > len(avg_fps) / 4:
            continue
        # 如果平均 FPS 数量不等于 Stream 数量, 则跳过该行
        if len(avg_fps) != streams:
            print(f"[Warning]: Expected {streams} streams, but got {len(avg_fps)}. Skipping line.")
            continue
        # 将符合条件的平均 FPS 添加到数据列表中
        data.append(avg_fps)
    # 如果没有有效数据, 返回空的 DataFrame
    if not data:
        return pd.DataFrame()
    # 转换为 DataFrame 并命名列
    return pd.DataFrame(data, columns=[f'S{i:03}' for i in range(len(data[0]))])

# 解析 usage 日志
def parse_usage_log(usage_log: str) -> pd.DataFrame:
    # 按行分割
    lines = usage_log.strip().split('\n')
    # 过滤掉以 '#' 开头的注释行
    lines = [line for line in lines if not line.startswith('#')]
    # 提取所需的列数据
    data = []
    for line in lines:
        # 按空白字符分割数据
        # Date(0), Time(1), gpu(2), sm(3), mem(4), enc(5), dec(6), jpg(7), ofa(8)
        parts = line.split()
        if len(parts) < 9:
            continue
        # 提取 gpu, sm, mem, dec 列并转换为整数
        try:
            row = (int(parts[2]), int(parts[3]), int(parts[4]), int(parts[6]))
        except ValueError:
            continue
        # 如果 gpu, sm, mem, dec 中有任何一个为 0, 则跳过该行
        if row[1] == 0 or row[2] == 0 or row[3] == 0:
            continue
        # 添加到数据列表中
        data.append(row)
    # 如果没有有效数据, 返回空的 DataFrame
    if not data:
        return pd.DataFrame()
    # 转换为 DataFrame 并命名列
    return pd.DataFrame(data, columns=['gpu', 'sm', 'mem', 'dec'])

# 处理日志文件并汇总结果
def process_logs(log_dir: str) -> pd.DataFrame:
    data = []
    for root, dirs, _ in os.walk(log_dir):
        # 遍历每个子目录
        for dir in dirs:
            # 从目录名提取元数据
            try:
                parts = dir.split('_')
                resolution = parts[0]
                streams = int(parts[1].replace('s', ''))
            except Exception as e:
                print(f"[Error] Processing directory {dir}: {e}")
                continue
            # 日志目录路径
            dir_path = os.path.join(root, dir)
            print(dir_path)
            # 读取日志文件内容
            app_log_file = os.path.join(dir_path, "app.log")
            with open(app_log_file, encoding='UTF-8', mode="r") as f:
                app_log = f.read()
            usage_log_file = os.path.join(dir_path, "usage.log")
            with open(usage_log_file, encoding='UTF-8', mode="r") as f:
                usage_log = f.read()
            # 解析日志并计算各项指标
            df_app = clean_data(parse_app_log(app_log, streams))
            avg_fps = df_app.mean(axis=0).mean()
            sum_fps = df_app.mean(axis=0).sum()
            df_usage = clean_data(parse_usage_log(usage_log))
            avg_sm = df_usage['sm'].mean()
            avg_mem = df_usage['mem'].mean()
            avg_dec = df_usage['dec'].mean()
            # 保存结果
            data.append({
                'resolution': resolution,
                'streams': streams,
                'avg_fps': avg_fps,
                'sum_fps': sum_fps,
                'avg_sm': avg_sm,
                'avg_mem': avg_mem,
                'avg_dec': avg_dec
            })
    # 将结果转为 DataFrame
    df = pd.DataFrame(data)
    # 按 resolution 和 streams 分组并计算均值
    df = df.groupby(['resolution', 'streams'], as_index=False).mean()
    # 按 resolution 和 streams 排序
    df.sort_values(by=['resolution', 'streams'], inplace=True)
    # 重置索引
    df.reset_index(drop=True, inplace=True)
    # 保留两位小数
    df = df.round(2)
    # 并保存为 CSV 文件
    csv_file = "perf_summary.csv"
    df.to_csv(csv_file, index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process log files and plot performance data.")
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.getcwd(), "perf_logs"), help="Directory containing log files")
    args = parser.parse_args()

    # 处理日志并保存结果
    df = process_logs(args.log_dir)
    print(df)
    # 绘图
    plot_perf(df)
