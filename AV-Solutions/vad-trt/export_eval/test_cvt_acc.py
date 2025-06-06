# verify_precision.py
# 这个脚本会自动逐个验证您导出的模型模块，以定位精度问题。
# 最终修复版：移除所有链式验证逻辑，对每个模块进行完全独立的验证。

import os
import sys
import onnx
import numpy as np
import glob
import ctypes
import tensorrt as trt
from pathlib import Path

# 检查 Polygraphy 和 onnxruntime-gpu 是否已安装
try:
    from polygraphy.backend.trt import TrtRunner, EngineFromBytes
    from polygraphy.comparator import Comparator, CompareFunc, RunResults
except ImportError as e:
    print(f"[致命错误] Polygraphy 依赖库未找到或导入失败: {e}")
    print("请确认已正确安装: pip install polygraphy --extra-index-url https://pypi.ngc.nvidia.com")
    sys.exit(1)


# --- 配置 ---

# 存放所有导出模块的根目录
BASE_DIR = "scratch"
# 自定义插件库的路径
SCRIPT_DIR = Path(__file__).resolve().parent
TRT_PLUGIN_PATH = SCRIPT_DIR.parent / "plugins/build/libplugins.so"


# --- 最终修复：移除模块依赖关系，改为独立的模块列表 ---
MODULES_TO_VERIFY = [
    "vadv1.extract_img_feat",
    "vadv1.pts_bbox_head.forward",
    "vadv1_prev.pts_bbox_head.forward"
]

# FP16 对比的容忍度。
RTOL = 1e-2
ATOL = 1e-3


# --- 脚本主逻辑 ---

# 字典：将 ONNX 的数据类型枚举值映射到 NumPy 的数据类型
ONNX_TO_NUMPY_TYPE = {
    1: np.float32,   # FLOAT
    10: np.float16,  # FLOAT16
    11: np.float64,  # DOUBLE
    6: np.int32,     # INT32
    7: np.int64,     # INT64
}

def load_engine(engine_path):
    """以二进制模式读取 .engine 文件。"""
    print(f"正在从路径加载预编译的 TensorRT 引擎: {engine_path}")
    with open(engine_path, "rb") as f:
        return f.read()

def load_data(onnx_model, data_dir):
    """
    精确地加载 .bin 文件。
    """
    feed_dict = {}
    golden_outputs = {}
    
    initializer_names = {initializer.name for initializer in onnx_model.graph.initializer}
    
    # 加载真正的用户输入
    for inp in onnx_model.graph.input:
        tensor_name = inp.name
        if tensor_name in initializer_names:
            continue
        
        bin_path = os.path.join(data_dir, f"{tensor_name}.bin")
        if os.path.exists(bin_path):
            tensor_shape = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
            numpy_dtype = ONNX_TO_NUMPY_TYPE.get(inp.type.tensor_type.elem_type, np.float32)
            
            array = np.fromfile(bin_path, dtype=numpy_dtype)
            if np.prod(tensor_shape) != array.size:
                print(f"  [错误] 输入 '{tensor_name}' 的形状 {tensor_shape} 与 .bin 文件中的元素数量 {array.size} 不匹配。")
                return None, None
            feed_dict[tensor_name] = array.reshape(tensor_shape)

    # 加载黄金标准输出
    for out in onnx_model.graph.output:
        tensor_name = out.name
        bin_path = os.path.join(data_dir, f"{tensor_name}.bin")
        if os.path.exists(bin_path):
            golden_outputs[tensor_name] = np.fromfile(bin_path, dtype=np.float32)

    return feed_dict, golden_outputs

def print_detailed_statistics(trt_results, golden_results):
    """
    手动计算并打印 TRT 输出和黄金标准数据之间的误差统计。
    """
    print("\n---- 手动计算详细误差统计 ----")
    for name, result_arr in trt_results.items():
        if name in golden_results:
            golden_arr = golden_results[name]
            
            try:
                golden_arr = golden_arr.reshape(result_arr.shape)
            except ValueError:
                print(f"  - 输出张量: {name} | [警告] 形状不匹配，跳过统计。 TRT: {result_arr.shape}, Golden (raw): {golden_arr.shape}")
                continue

            result_arr_f32 = result_arr.astype(np.float32)
            golden_arr_f32 = golden_arr.astype(np.float32)

            abs_err = np.abs(result_arr_f32 - golden_arr_f32)
            rel_err = abs_err / (np.abs(golden_arr_f32) + 1e-9)

            print(f"  - 输出张量: {name}")
            print(f"    - 平均绝对误差 (Mean Abs Err): {np.mean(abs_err):.6g}")
            print(f"    - 平均相对误差 (Mean Rel Err): {np.mean(rel_err):.6g}")
            print(f"    - 最大绝对误差 (Max Abs Err):  {np.max(abs_err):.6g}")
            print(f"    - 最大相对误差 (Max Rel Err):  {np.max(rel_err):.6g}")
        else:
            print(f"  - 输出张量: {name} | [警告] 在黄金标准数据中未找到，无法计算统计。")


def verify_module(module_name):
    """
    为单个模块进行验证，并返回其验证结果。
    """
    print("=" * 80)
    print(f"开始验证模块: {module_name}")
    print("=" * 80)

    module_dir = os.path.join(BASE_DIR, module_name)
    if not os.path.isdir(module_dir):
        print(f"[警告] 找不到模块目录: {module_dir}，已跳过。")
        return False

    sim_onnx_files = glob.glob(os.path.join(module_dir, "sim_*.onnx"))
    engine_files = glob.glob(os.path.join(module_dir, "*fp32.engine"))

    if not sim_onnx_files:
        print(f"[错误] 在目录 {module_dir} 中找不到 'sim_*.onnx' 文件 (需要用它来解析输入输出结构)。")
        return False
    if not engine_files:
        print(f"[错误] 在目录 {module_dir} 中找不到预编译的 '.engine' 文件。")
        return False
        
    onnx_path = sim_onnx_files[0]
    engine_path = engine_files[0]
        
    print(f"找到ONNX模型 (仅用于解析结构): {os.path.basename(onnx_path)}")
    print(f"找到TensorRT引擎 (待验证): {os.path.basename(engine_path)}")
    
    onnx_model = onnx.load(onnx_path)
    feed_dict, golden_outputs = load_data(onnx_model, module_dir)
    
    if not feed_dict:
        print("[错误] 未能加载任何有效的输入数据，无法继续。")
        return False
    if not golden_outputs:
        print("[错误] 未能加载任何黄金标准输出数据 (.bin)，无法进行对比。")
        return False
        
    print("\n成功加载所有必需的输入和黄金标准输出数据。")

    trt_runner = TrtRunner(EngineFromBytes(load_engine(engine_path)), name="Pre-built TRT Engine")
    
    passed = False
    try:
        print("\n正在激活TRT推理器...")
        trt_runner.activate()
        print("推理器已激活, 开始执行推理...")
        
        trt_outputs = trt_runner.infer(feed_dict=feed_dict)
        
        print("推理完成，正在计算和对比精度...")
        
        print_detailed_statistics(trt_outputs, golden_outputs)
        
        compare_func = CompareFunc.simple(rtol=RTOL, atol=ATOL)
        
        run_results_trt = RunResults()
        run_results_trt.add([trt_outputs], trt_runner.name)

        run_results_golden = RunResults()
        run_results_golden.add([golden_outputs], "PyTorch (Golden .bin)")
        
        compare_results = Comparator.compare_accuracy(run_results_trt, run_results_golden, compare_func=compare_func)
        
        print(f"\n---- Polygraphy 最终裁定 (容忍度 rtol={RTOL}, atol={ATOL}) ----")
        
        if compare_results:
            passed = False
            print(f"\n[失败] ❌ 模块 '{module_name}' TRT引擎与原始PyTorch结果不一致。")
            print("      以下是Polygraphy报告的超差项：")
            print(compare_results)
        else:
            passed = True
            print(f"\n[成功] ✅ 模块 '{module_name}' TRT引擎与原始PyTorch结果一致。")

    finally:
        print("\n正在停用推理器...")
        if 'trt_runner' in locals() and trt_runner.is_active:
            trt_runner.deactivate()
        
    return passed

def main():
    """
    主函数，实现独立的模块验证。
    """
    print("=" * 80)
    print("正在加载 TensorRT 自定义插件...")
    if not TRT_PLUGIN_PATH.exists():
        print(f"[致命错误] 找不到插件文件: {TRT_PLUGIN_PATH}")
        print("请检查脚本顶部的 TRT_PLUGIN_PATH 配置是否正确。")
        sys.exit(1)
        
    try:
        ctypes.CDLL(str(TRT_PLUGIN_PATH))
        print(f"成功加载插件库: {TRT_PLUGIN_PATH}")
    except Exception as e:
        print(f"[致命错误] 加载TRT插件 {TRT_PLUGIN_PATH} 失败: {e}")
        sys.exit(1)
        
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    print("=" * 80)
    
    # --- 最终修复：对每个模块进行独立验证 ---
    results_summary = {}
    for module in MODULES_TO_VERIFY:
        passed = verify_module(module)
        results_summary[module] = passed

    # 打印最终的总结报告
    print("\n" + "#" * 80)
    print("所有模块验证完成。最终总结报告：")
    print("#" * 80 + "\n")
    
    all_passed = True
    for module, passed in results_summary.items():
        status = "✅ 成功" if passed else "❌ 失败"
        print(f"- {module}: {status}")
        if not passed:
            all_passed = False

    if not all_passed:
        print("\n[结论] 发现一个或多个模块存在精度问题。请首先关注标记为 [❌ 失败] 的模块。")
    else:
        print("\n[结论] 🎉 恭喜！所有模块都独立地通过了精度验证！")
        print("如果您的端到端应用仍然存在问题，那么根源很可能不在于单个模块的精度，而在于模块间的交互或未被验证的后处理代码。")


if __name__ == "__main__":
    main()
