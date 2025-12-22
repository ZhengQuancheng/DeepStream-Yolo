import os
import sys
import time
import argparse
import subprocess
import configparser
import math
import signal
from datetime import datetime

# 生成配置文件名称
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
GEN_APP_CONFIG = f"gen_app_config_{TIMESTAMP}.txt"
GEN_GIE_CONFIG = f"gen_gie_config_{TIMESTAMP}.txt"
LOG_ROOT = "perf_logs"

# 配置文件解析器
class CaseSensitiveConfigParser(configparser.ConfigParser):
    # DeepStream 配置文件区分大小写, 需要重写 optionxform
    def optionxform(self, optionstr):
        return optionstr

# 创建日志保存目录
def setup_log_dir(width, height, streams):
    log_dir = os.path.join(LOG_ROOT, f"{width}x{height}_{streams}s_{TIMESTAMP}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

# 生成 engine 文件
def build_engine(config_path, engine_file):
    # 若模型文件已存在, 则退出
    if os.path.exists(engine_file):
        print(f"[*] Engine found: {engine_file}. Skipping build phase.")
        return
    # 若模型文件未存在, 则构建
    print(f"[!] Engine file `{engine_file}` NOT found")
    print(f"[!] Starting DeepStream in BUILD MODE. This may take a few minutes...")

    # 启动 DeepStream 以构建 engine
    ds_cmd = ["deepstream-app", "-c", config_path]
    process = subprocess.Popen(
        ds_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        # 实时读取日志, 寻找 engine 生成完成的信号
        start_time = time.time()
        engine_built = False

        while True:
            line = process.stdout.readline()
            if not line:
                break
            # 打印关键信息
            if "Building" in line in line:
                print(f"    [Build Log] {line.strip()}")
            # 构建完成
            if "**PERF" in line:
                print(f"[*] Engine build detected successful!")
                engine_built = True
                break
            # 设置 20 分钟超时
            if time.time() - start_time > 1200:
                print("[Error] Engine build timed out.")
                break

        # 终止进程
        process.terminate()
        try:
            process.wait(timeout=5)
        except:
            process.kill()

        if engine_built:
            print(f"[*] Engine {engine_file} is ready.")
        else:
            print(f"[Error] Failed to verify engine build. Check configs.")
            sys.exit(1)

    except KeyboardInterrupt:
        process.kill()
        sys.exit(1)

# 生成配置文件
def generate_configs(base_app_config_filename, base_gie_config_filename, streams, onnx_file, input_type, input_uri, width, height, output_type):
    print(f"[*] Generating configuration files in current directory...")

    # 读取 GIE 配置文件
    gie_config = CaseSensitiveConfigParser()
    gie_config.read(base_gie_config_filename)
    # 修改 batch-size 以匹配路数
    gie_config.set("property", "batch-size", str(streams))
    # 设置 network-mode 为 FP16 模式 -  0=FP32, 1=INT8, 2=FP16 mode
    gie_config.set("property", "network-mode", "2")
    # 设置 model-engine-file
    gie_config.set("property", "model-engine-file", f"model_b{streams}_gpu0_fp16.engine")
    # 设置 onnx-file
    gie_config.set("property", "onnx-file", onnx_file)
    # 写入配置文件 GEN_GIE_CONFIG
    with open(GEN_GIE_CONFIG, encoding='UTF-8', mode="w") as f:
        gie_config.write(f)

    # 读取 App 配置文件
    app_config = CaseSensitiveConfigParser()
    app_config.read(base_app_config_filename)
    # 移除旧有的 source
    for section in app_config.sections():
        if section.startswith("source"):
            app_config.remove_section(section)
    # 动态添加 source  -  input_type: 3=File, 4=RTSP
    type_id = "4" if input_type == "rtsp" else "3"
    for i in range(streams):
        section_name = f"source{i}"
        app_config.add_section(section_name)
        app_config.set(section_name, "enable", "1")
        app_config.set(section_name, "type", type_id)
        app_config.set(section_name, "uri", input_uri)
        app_config.set(section_name, "gpu-id", "0")
        app_config.set(section_name, "cudadec-memtype", "0")
        # 若为 RTSP, 可设置延迟以减少抖动
        if input_type == 'rtsp':
            app_config.set(section_name, "latency", "200")

    # 移除旧有的 sink
    for section in app_config.sections():
        if section.startswith("sink"):
            app_config.remove_section(section)
    # 设置 sink (仅支持一个)  -  1=FakeSink 4=RTSPStreaming
    type_id = "4" if output_type == "rtsp" else "1"
    section_name = f"sink{i}"
    app_config.add_section(section_name)
    app_config.set(section_name, "enable", "1")
    app_config.set(section_name, "type", type_id)
    app_config.set(section_name, "sync", "1") # 1: Synchronously
    app_config.set(section_name, "codec", "1") # 1=h264 2=h265
    app_config.set(section_name, "enc-type", "1") # 0=Hardware 1=Software
    app_config.set(section_name, "rtsp-port", "8554")
    app_config.set(section_name, "udp-port", "5400")

    # 更新 tiled display
    rows = int(math.sqrt(streams))
    cols = math.ceil(streams / rows)
    app_config.set("tiled-display", "rows", str(rows))
    app_config.set("tiled-display", "columns", str(cols))
    app_config.set("tiled-display", "width", str(width))
    app_config.set("tiled-display", "height", str(height))

    # 更新 streammux
    app_config.set("streammux", "width", str(width))
    app_config.set("streammux", "height", str(height))
    if input_type == "rtsp":
        app_config.set("streammux", "live-source", "1")

    # 更新 GIE 路径
    app_config.set("primary-gie", "config-file", GEN_GIE_CONFIG)

    # 更新 tests
    if input_type != "rtsp":
        app_config.set("tests", "file-loop", "1")

    # 写入配置文件 GEN_APP_CONFIG
    with open(GEN_APP_CONFIG, encoding='UTF-8', mode='w') as f:
        app_config.write(f)

def run_perf_test(base_app_config_filename, base_gie_config_filename, streams, onnx_file, input_type, input_uri, width, height, output_type, duration):

    # 生成配置文件
    generate_configs(base_app_config_filename, base_gie_config_filename, streams, onnx_file, input_type, input_uri, width, height, output_type)
    # 设置日志目录
    log_dir = setup_log_dir(width, height, streams)

    # 构建 engine
    conf = CaseSensitiveConfigParser()
    conf.read(GEN_GIE_CONFIG)
    engine_file = conf.get("property", "model-engine-file")
    build_engine(GEN_APP_CONFIG, engine_file)

    # 启动资源监控 (nvidia-smi)
    print(f"[*] Starting Monitor (nvidia-smi)...")
    dmon_cmd = ["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-o", "DT", "-i", "0"]
    dmon_log_file = open(os.path.join(log_dir, "usage.log"), "w")
    dmon_proc = subprocess.Popen(dmon_cmd, stdout=dmon_log_file, stderr=subprocess.STDOUT)

    # 启动 DeepStream App
    ds_cmd = ["deepstream-app", "-c", GEN_APP_CONFIG]
    app_log_file = open(os.path.join(log_dir, "app.log"), "w")

    try:
        print(f"[*] Starting DeepStream App (Duration: {duration}s)...")
        ds_proc = subprocess.Popen(ds_cmd, stdout=app_log_file, stderr=subprocess.STDOUT)
        try:
            ds_proc.wait(timeout=duration)
        except subprocess.TimeoutExpired:
            print(f"[*] Stopping DeepStream App...")
            ds_proc.send_signal(signal.SIGINT)
    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
        ds_proc.send_signal(signal.SIGINT)
    except Exception as e:
        print(f"[x] Unknown Exception: {e}")
    finally:
        # 关闭 DeepStream App
        try:
            ds_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            ds_proc.kill()
        # 关闭 nvidia-smi
        dmon_proc.terminate()
        try:
            dmon_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            dmon_proc.kill()
        # Close files
        dmon_log_file.close()
        app_log_file.close()

        print(f"[*] Test completed. Logs saved to {log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepStream-YOLO Perf Automation")

    parser.add_argument("--app_config", type=str, default="deepstream_app_config.txt", help="Base App Config")
    parser.add_argument("--gie_config", type=str, default="config_infer_primary_yoloV8.txt", help="Base GIE Config")
    parser.add_argument("--streams", type=int, required=True, help="Number of streams")
    parser.add_argument("--onnx_file", type=str, default="yolov8s.pt.onnx", help="ONNX File")
    parser.add_argument("--input_type", type=str, choices=['rtsp', 'file'], required=True, help="Input type")
    parser.add_argument("--input_uri", type=str, required=True, help="Input URI")
    parser.add_argument("--width", type=int, default=640, help="Streammux/Display width")
    parser.add_argument("--height", type=int, default=480, help="Streammux/Display height")
    parser.add_argument("--output_type", type=str, choices=['rtsp', 'fake'], default="rtsp", help="Output type")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")

    args = parser.parse_args()

    run_perf_test(
        base_app_config_filename=args.app_config,
        base_gie_config_filename=args.gie_config,
        streams=args.streams,
        onnx_file=args.onnx_file,
        input_type=args.input_type,
        input_uri=args.input_uri,
        width=args.width,
        height=args.height,
        output_type=args.output_type,
        duration=args.duration)
