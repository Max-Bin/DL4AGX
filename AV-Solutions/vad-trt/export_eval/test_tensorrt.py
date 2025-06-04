# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified from https://github.com/hustvl/VAD/blob/main/tools/test.py
# Licensed by https://github.com/hustvl/VAD/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import sys
sys.path.append('')
import numpy as np
import argparse
import mmcv
import os
import copy
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings

from mmcv.utils import get_dist_info, init_dist, wrap_fp16_model, set_random_seed, Config, DictAction, load_checkpoint
from mmcv.models import build_model, fuse_conv_bn
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from mmcv.datasets import build_dataset, build_dataloader, replace_ImageToTensor
import time
import os.path as osp
from mmcv.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test, single_gpu_test # 确保 single_gpu_test 也被导入了
from mmcv.fileio.io import dump, load # mmcv.dump 被使用，确保 mmcv 是来源
import json
from collections import OrderedDict
from pathlib import Path # 导入 Path

import warnings
warnings.filterwarnings("ignore")

# --- 辅助函数用于打印Tensor信息 ---
def print_tensor_info(tensor, name="Tensor", indent=0):
    """打印PyTorch张量的详细调试信息"""
    prefix = " " * indent
    if not isinstance(tensor, torch.Tensor):
        print(f"{prefix}{name}: 不是一个PyTorch张量, 类型为 {type(tensor)}")
        return
    print(f"{prefix}{name}:")
    print(f"{prefix}  形状 (Shape): {tensor.shape}")
    print(f"{prefix}  数据类型 (dtype): {tensor.dtype}")
    print(f"{prefix}  设备 (Device): {tensor.device}")
    if tensor.numel() > 0: # 仅在张量非空时计算统计值
        # 确保张量在CPU上并且是浮点类型才能计算统计值，避免不必要的GPU同步或类型错误
        tensor_for_stats = tensor
        if tensor.is_cuda:
            tensor_for_stats = tensor.cpu()
        
        if tensor_for_stats.is_floating_point():
            try:
                # 使用 .item() 获取标量值
                print(f"{prefix}  最小值 (Min): {torch.min(tensor_for_stats).item():.6f}")
                print(f"{prefix}  最大值 (Max): {torch.max(tensor_for_stats).item():.6f}")
                print(f"{prefix}  均值 (Mean): {torch.mean(tensor_for_stats).item():.6f}")
                print(f"{prefix}  方差 (Std): {torch.std(tensor_for_stats).item():.6f}")
                # 检查是否存在NaN或Inf
                print(f"{prefix}  包含NaN (Has NaN): {torch.isnan(tensor_for_stats).any().item()}")
                print(f"{prefix}  包含Inf (Has Inf): {torch.isinf(tensor_for_stats).any().item()}")
            except Exception as e:
                print(f"{prefix}  计算统计值时出错: {e}")
        else:
            print(f"{prefix}  非浮点型张量，跳过统计值计算。")
            # 对于非浮点型，可以打印一些唯一值或计数
            unique_vals, counts = torch.unique(tensor_for_stats, return_counts=True)
            print(f"{prefix}  唯一值 (Unique values): {unique_vals[:10].tolist()} ... (最多显示10个)") # 显示前10个唯一值
            print(f"{prefix}  对应数量 (Counts): {counts[:10].tolist()} ...")

    else:
        print(f"{prefix}  张量为空。")
    print("-" * (indent + 20))


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--json_dir', help='json parent dir name file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is useful when you want to format the result to a specific format and submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox", "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy format will be kwargs for dataset.evaluate() function (deprecate), change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE

    if not distributed:
        model_pytorch = copy.deepcopy(model) # 创建一个PyTorch模型的深拷贝用于对比
        model_pytorch = DataParallel(model_pytorch, device_ids=[0])
        model_pytorch.eval()

        model = DataParallel(model, device_ids=[0]) # TRT测试用的模型
        model.eval()

        me_file_path = Path(__file__) # 重命名以避免与helper类中的me冲突
        
        # --- TRT 自定义插件加载 ---
        # 确保这个路径是正确的，并且 libplugins.so 与你的TRT版本和CUDA版本兼容
        trt_plugin_path = me_file_path.parent / "../plugins/build/libplugins.so"
        try:
            print(f"尝试加载TRT插件: {trt_plugin_path}")
            ctypes.CDLL(str(trt_plugin_path)) # 转换为字符串
            print(f"TRT插件 {trt_plugin_path} 加载成功 (通过ctypes)。")
        except Exception as e:
            print(f"加载TRT插件 {trt_plugin_path} 失败: {e}")
            print("警告: TRT自定义插件加载失败可能会导致TRT引擎无法正确运行或加载。")


        from bev_deploy.hook import HookHelper, Hook
        from bev_deploy.trt.inference import InferTrt # 确保这个类能被正确导入
        from bev_deploy.patch.bevformer.patch import patch_spatial_cross_attn_forward, patch_point_sampling, \
            patch_bevformer_encoder_forward
        from bev_deploy.patch.bevformer.patch import fn_lidar2img, fn_canbus
        
        ch = HookHelper()
        # Hook的是原始的 model.module (DataParallel包装前的)
        ch.attach_hook(model.module, "vadv1") 
        
        for i in range(3): # 假设有3层，根据你的模型调整
            if hasattr(model.module.pts_bbox_head.transformer.encoder, 'layers') and \
               len(model.module.pts_bbox_head.transformer.encoder.layers) > i and \
               hasattr(model.module.pts_bbox_head.transformer.encoder.layers[i], 'attentions') and \
               len(model.module.pts_bbox_head.transformer.encoder.layers[i].attentions) > 1:
                ch.hooks[f"vadv1.pts_bbox_head.transformer.encoder.layers.{i}.attentions.1.forward"]._patch(patch_spatial_cross_attn_forward)
            else:
                print(f"警告: 无法为encoder layer {i} attention 1 设置patch，模型结构可能不匹配。")

        if hasattr(model.module.pts_bbox_head.transformer.encoder, 'point_sampling'):
            ch.hooks["vadv1.pts_bbox_head.transformer.encoder.point_sampling"]._patch(patch_point_sampling)
        else:
            print("警告: 无法为encoder point_sampling 设置patch。")
        
        if hasattr(model.module.pts_bbox_head.transformer, 'encoder'):
             ch.hooks["vadv1.pts_bbox_head.transformer.encoder.forward"]._patch(patch_bevformer_encoder_forward)
        else:
            print("警告: 无法为transformer.encoder.forward 设置patch。")


        from patch.patch_head import patch_VADHead_select_and_pad_query, patch_VADPerceptionTransformer_get_bev_features, patch_VADHead_select_and_pad_pred_map
        
        # 确保这些属性存在于 model.module.pts_bbox_head
        if hasattr(model.module.pts_bbox_head, 'select_and_pad_query'):
            ch.hooks["vadv1.pts_bbox_head.select_and_pad_query"]._patch(patch_VADHead_select_and_pad_query)
        else:
            print("警告: 无法为pts_bbox_head.select_and_pad_query 设置patch。")

        if hasattr(model.module.pts_bbox_head, 'select_and_pad_pred_map'):
            ch.hooks["vadv1.pts_bbox_head.select_and_pad_pred_map"]._patch(patch_VADHead_select_and_pad_pred_map)
        else:
            print("警告: 无法为pts_bbox_head.select_and_pad_pred_map 设置patch。")
        
        if hasattr(model.module.pts_bbox_head.transformer, 'get_bev_features'):
            ch.hooks["vadv1.pts_bbox_head.transformer.get_bev_features"]._patch(patch_VADPerceptionTransformer_get_bev_features)
        else:
            print("警告: 无法为transformer.get_bev_features 设置patch。")


        class TrtExtractImgFeatHelper(object):
            def __init__(self) -> None:
                self.infer_engine = InferTrt() # 修改变量名以避免与外部的 infer 冲突
                engine_path = "scratch/vadv1.extract_img_feat/vadv1.extract_img_feat.fp16.engine"
                print(f"尝试加载TRT引擎: {engine_path}")
                try:
                    self.infer_engine.read(engine_path)
                    print(f"TRT引擎 {engine_path} 加载成功。引擎信息:")
                    print(self.infer_engine) # InferTrt类应该有一个好的 __str__ 或 __repr__ 实现
                except Exception as e:
                    print(f"加载TRT引擎 {engine_path} 失败: {e}")
                    self.infer_engine = None


            def get_func(self):
                helper_self = self # 区分 TrtExtractImgFeatHelper 的 self 和被patch方法的 self
                def trt_extract_img_feat(original_self, img, img_metas, len_queue=None):
                    if not helper_self.infer_engine:
                        print("错误: TrtExtractImgFeatHelper 的 TRT引擎未加载，将尝试使用原始PyTorch方法。")
                        # 调用原始的PyTorch方法 (如果需要回退)
                        # 这需要你存储原始函数，或者让hook系统支持调用原始函数
                        # 为简单起见，这里我们先假设引擎必须加载成功
                        raise RuntimeError("TrtExtractImgFeatHelper TRT引擎未加载！")

                    print("\n--- TrtExtractImgFeatHelper: trt_extract_img_feat 执行开始 ---")
                    torch.cuda.synchronize() 
                    dev = img.device
                    
                    print_tensor_info(img, "输入 img (原始)", indent=2)
                    # 确保img是连续的
                    img_contiguous = img.contiguous()
                    if not img.is_contiguous():
                        print("  注意: 输入 img 不是内存连续的，已转换为连续。")
                    print_tensor_info(img_contiguous, "输入 img (连续的, 送入TRT)", indent=2)


                    # vadv1.extract_img_feat
                    # feat_shapes 需要与你的TRT引擎输出绑定严格对应
                    # 你需要从TRT引擎的定义或构建输出来确定这些形状
                    # 示例形状，请务必修改为你的实际输出形状
                    feat_shapes = [[1, 6, 256, 12, 20]] # 假设只有一个输出，形状为 [B, N_cam, C, H, W]
                                                       # 注意：这里的 B 应该是 1 (samples_per_gpu)
                                                       # 而 N_cam 嵌入在第二个维度。
                                                       # TRT引擎的输出形状可能是 [B*N_cam, C, H, W] -> [6, 256, 12, 20]
                                                       # 或者直接是 [1, 6, 256, 12, 20]
                                                       # 这取决于你的TRT引擎是如何构建的。
                                                       # 假设TRT输出的是 [B*N_cam, C, H, W]
                    # 调整 feat_shapes 以匹配 TRT 引擎的实际输出绑定
                    # 假设引擎输出的是 (B*N_cam, C, H, W)
                    # B=1, N_cam=6 -> (6, 256, 12, 20)
                    # 如果你的引擎输出是 (B, N_cam, C, H, W) -> (1, 6, 256, 12, 20)
                    # 这里我们假设是后者，因为PyTorch原始函数可能返回这种形式
                    
                    # 让我们假设 TRT 引擎的输出是 (num_cameras, channels, height, width)
                    # 并且我们只有一个输出绑定。
                    # **你需要根据你的TRT引擎的实际输出绑定来调整这里的 feat_shapes**
                    # 例如，如果你的TRT引擎的输出是 (6, 256, 12, 20)
                    # 那么 feat_shapes 应该是 [[6, 256, 12, 20]]
                    # 如果你的TRT引擎输出是 (1, 6, 256, 12, 20)
                    # 那么 feat_shapes 应该是 [[1, 6, 256, 12, 20]]
                    # 这里的示例假设是后者，因为原始PyTorch的img_feats通常是 (B, N, C, H, W)
                    # 但更可能的情况是，为了TRT优化，输入和输出都是 (B*N, ...)
                    # 让我们采用 (B*N, C, H, W) 的假设，因为这在 TRT 中更常见
                    # B=1, N_cam=6 => (6, C, H, W)
                    # 假设只有一个输出，通道数为256，H=12, W=20
                    feat_shapes = [[6, 256, 12, 20]] # 示例，你需要确认！

                    img_feats_trt = [torch.zeros(size=s, dtype=torch.float32, device=dev).contiguous() for s in feat_shapes]
                    
                    # 准备TRT的输入输出参数列表
                    # 顺序必须严格匹配TRT引擎的绑定顺序！
                    # 假设第一个绑定是输入img，后续绑定是输出
                    trt_bindings = [img_contiguous.data_ptr()] + [f.data_ptr() for f in img_feats_trt]
                    
                    print("  调用 TRT infer_engine.forward()...")
                    helper_self.infer_engine.forward(trt_bindings)
                    torch.cuda.synchronize()
                    print("  TRT infer_engine.forward() 调用完成。")

                    for i, feat_out in enumerate(img_feats_trt):
                        print_tensor_info(feat_out, f"TRT 输出 img_feats_trt[{i}]", indent=2)
                    
                    # 如果需要，这里可以将 TRT 的输出形状调整回 PyTorch 期望的形状
                    # 例如，如果 PyTorch 期望 List[Tensor(B, N_cam, C, H, W)]
                    # 而 TRT 输出的是 Tensor(B*N_cam, C, H, W)
                    # 你需要在这里进行 reshape
                    # final_img_feats = []
                    # for trt_feat in img_feats_trt:
                    #    if trt_feat.shape[0] == img.shape[0] * img.shape[1]: # B * N_cam
                    #        B_dim, N_cam_dim = img.shape[0], img.shape[1]
                    #        C_dim, H_dim, W_dim = trt_feat.shape[1], trt_feat.shape[2], trt_feat.shape[3]
                    #        final_img_feats.append(trt_feat.view(B_dim, N_cam_dim, C_dim, H_dim, W_dim))
                    #    else:
                    #        final_img_feats.append(trt_feat) # 或者抛出错误，如果形状不匹配预期
                    # print("--- TrtExtractImgFeatHelper: trt_extract_img_feat 执行结束 ---")
                    # return final_img_feats
                    
                    print("--- TrtExtractImgFeatHelper: trt_extract_img_feat 执行结束 ---")
                    return img_feats_trt # 直接返回TRT的输出列表，后续模块需要处理这个格式

                return trt_extract_img_feat
        import tensorrt as trt # 确保导入了 tensorrt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # 或者 trt.Logger.INFO, trt.Logger.VERBOSE 获取更详细日志

        print("尝试初始化TensorRT插件库 (trt.init_libnvinfer_plugins)...")
        if trt.init_libnvinfer_plugins(TRT_LOGGER, ""): # 使用你创建的logger实例
            print("trt.init_libnvinfer_plugins 调用成功。")
        else:
            print("警告: trt.init_libnvinfer_plugins 返回 false。这可能意味着没有找到额外的插件库去加载，或者发生了错误。")
        # 注意：即使返回false，也不一定意味着你的特定 libplugins.so 失败，
        # 因为这个函数是通用初始化，它会尝试加载所有标准TRT插件库。
        # 关键还是要看后续加载引擎时是否还报找不到creator的错误。


        h1 = TrtExtractImgFeatHelper()  
        if h1.infer_engine: # 仅在引擎成功加载时patch
            ch.hooks["vadv1.extract_img_feat"]._patch(h1.get_func())
        else:
            print("警告: TrtExtractImgFeatHelper 的引擎加载失败，vadv1.extract_img_feat 将不会被TRT替换。")


        from patch.patch_rotate import rotate # 确保这个patch文件存在且正确

        # 尝试获取 patch_head 模块，如果失败则打印警告
        try:
            mod_patch_head = sys.modules["patch.patch_head"] # 确保 "patch.patch_head" 模块已被导入
            ch.hooks["vadv1.func.rotate"] = Hook(mod_patch_head, "rotate", "vadv1.func")._patch(rotate)
            Hook.cache._capture["vadv1.func.rotate"] = []
        except KeyError:
            print("警告: 模块 'patch.patch_head' 未找到或未导入，无法patch vadv1.func.rotate。")
        except Exception as e:
            print(f"Patch vadv1.func.rotate 时发生错误: {e}")


        class TrtPtsBboxHeadHelper(object):
            def __init__(self) -> None:
                self.infer_fwd = InferTrt()
                self.infer_prev = InferTrt()
                self.original_pytorch_func = None # 用于存储原始的PyTorch函数

                engine_fwd_path = "scratch/vadv1.pts_bbox_head.forward/vadv1.pts_bbox_head.forward.engine"
                engine_prev_path = "scratch/vadv1_prev.pts_bbox_head.forward/vadv1_prev.pts_bbox_head.forward.engine"

                print(f"尝试加载TRT引擎: {engine_fwd_path}")
                try:
                    self.infer_fwd.read(engine_fwd_path)
                    print(f"TRT引擎 {engine_fwd_path} 加载成功。引擎信息:")
                    print(self.infer_fwd)
                except Exception as e:
                    print(f"加载TRT引擎 {engine_fwd_path} 失败: {e}")
                    self.infer_fwd = None

                print(f"尝试加载TRT引擎: {engine_prev_path}")
                try:
                    self.infer_prev.read(engine_prev_path)
                    print(f"TRT引擎 {engine_prev_path} 加载成功。引擎信息:")
                    print(self.infer_prev)
                except Exception as e:
                    print(f"加载TRT引擎 {engine_prev_path} 失败: {e}")
                    self.infer_prev = None


            def get_func(self):
                helper_self = self
                def trt_pts_bbox_head_fwd(original_self_module, mlvl_feats, img_metas, prev_bev=None, only_bev=False, ego_his_trajs=None, ego_lcf_feat=None):
                    current_infer_engine = None
                    engine_name_for_debug = ""

                    if prev_bev is not None:
                        current_infer_engine = helper_self.infer_prev
                        engine_name_for_debug = "infer_prev (vadv1_prev.pts_bbox_head.forward.engine)"
                    else:
                        current_infer_engine = helper_self.infer_fwd
                        engine_name_for_debug = "infer_fwd (vadv1.pts_bbox_head.forward.engine)"

                    if not current_infer_engine or not hasattr(current_infer_engine, 'forward'): # 检查 infer_fwd/infer_prev 是否成功加载
                        print(f"错误: TrtPtsBboxHeadHelper 的 TRT引擎 '{engine_name_for_debug}' 未加载或无效，将回退到原始PyTorch方法。")
                        if helper_self.original_pytorch_func:
                            print("  调用原始PyTorch函数...")
                            return helper_self.original_pytorch_func(original_self_module, mlvl_feats, img_metas, prev_bev, only_bev, ego_his_trajs, ego_lcf_feat)
                        else:
                            raise RuntimeError(f"TrtPtsBboxHeadHelper TRT引擎 '{engine_name_for_debug}' 未加载且无法回退到原始PyTorch函数！")
                    
                    print(f"\n--- TrtPtsBboxHeadHelper: trt_pts_bbox_head_fwd ({engine_name_for_debug}) 执行开始 ---")
                    
                    # 输入数据准备和校验
                    if not isinstance(mlvl_feats, list) or not mlvl_feats:
                        print_tensor_info(mlvl_feats, "错误: mlvl_feats 不是列表或为空", indent=2)
                        raise ValueError("mlvl_feats 必须是一个非空列表。")
                    
                    dev = mlvl_feats[0].device # 假设所有mlvl_feats在同一设备
                    print_tensor_info(mlvl_feats[0], "输入 mlvl_feats[0]", indent=2) # 假设主要使用第一个特征图

                    m_transformed = fn_lidar2img(img_metas) # fn_lidar2img 返回的是修改后的 img_metas
                    # fn_canbus 也返回修改后的 img_metas，包含 can_bus 和 shift
                    m_transformed_canbus = fn_canbus(mlvl_feats[0].contiguous(), copy.deepcopy(m_transformed), 100, 100, [0.6, 0.3]) 
                                                # 使用 mlvl_feats[0] 的连续版本, 深拷贝m_transformed避免副作用

                    can_bus_tensor = m_transformed_canbus[0]["can_bus"].contiguous()
                    shift_tensor = m_transformed_canbus[0]["shift"].contiguous()
                    lidar2img_tensor = m_transformed_canbus[0]["lidar2img"].to(dev).contiguous() # 确保在正确设备上且连续

                    print_tensor_info(can_bus_tensor, "输入 can_bus", indent=2)
                    print_tensor_info(shift_tensor, "输入 shift", indent=2)
                    print_tensor_info(lidar2img_tensor, "输入 lidar2img", indent=2)
                    if prev_bev is not None:
                        print_tensor_info(prev_bev.contiguous(), "输入 prev_bev", indent=2)

                    torch.cuda.synchronize()

                    # 定义输出张量的形状和查找表 (lut)
                    lut = [
                        ("bev_embed", [10000, 1, 256]),("all_cls_scores", [3, 1, 300, 10]),
                        ("all_bbox_preds", [3, 1, 300, 10]),("all_traj_preds", [3, 1, 300, 6, 12]),
                        ("all_traj_cls_scores", [3, 1, 300, 6]),("map_all_cls_scores", [3, 1, 100, 3]),
                        ("map_all_bbox_preds", [3, 1, 100, 4]),("map_all_pts_preds", [3, 1, 100, 20, 2]),
                        ("enc_cls_scores", None),("enc_bbox_preds", None),
                        ("map_enc_cls_scores", None),("map_enc_bbox_preds", None),
                        ("map_enc_pts_preds", None),("ego_fut_preds", [1, 3, 6, 2])]
                    
                    trt_outputs_dict = {} # 用于存储TRT输出张量
                    
                    # 准备TRT绑定参数列表
                    # !!! 关键: 这里的顺序必须严格匹配你的TRT引擎的绑定顺序 !!!
                    # 你需要查阅TRT引擎构建时的输入输出顺序，或者 InferTrt 类是否有方法获取绑定信息
                    trt_bindings = []
                    
                    # 假设的绑定顺序 (你需要根据实际情况调整)
                    # 1. mlvl_feats[0]
                    # 2. can_bus
                    # 3. shift
                    # 4. lidar2img
                    # 5. prev_bev (如果存在)
                    # 6. output_bev_embed
                    # 7. output_all_cls_scores
                    # ... 等等
                    
                    print("  准备TRT绑定参数...")
                    # 输入绑定
                    trt_bindings.append(mlvl_feats[0].contiguous().data_ptr()) # 确保连续
                    
                    if prev_bev is not None: # infer_prev 的绑定顺序可能不同
                        print("    使用 infer_prev 的绑定顺序 (假设): mlvl, can_bus, shift, lidar2img, prev_bev, outputs...")
                        trt_bindings.append(can_bus_tensor.data_ptr())
                        trt_bindings.append(shift_tensor.data_ptr())
                        trt_bindings.append(lidar2img_tensor.data_ptr())
                        trt_bindings.append(prev_bev.contiguous().data_ptr()) # 确保连续
                    else: # infer_fwd 的绑定顺序
                        print("    使用 infer_fwd 的绑定顺序 (假设): mlvl, shift, lidar2img, can_bus, outputs...")
                        # 这个顺序是基于原始ONNX脚本中的一个例子，你需要确认是否适用于你的TRT引擎
                        trt_bindings.append(shift_tensor.data_ptr())
                        trt_bindings.append(lidar2img_tensor.data_ptr())
                        trt_bindings.append(can_bus_tensor.data_ptr())

                    # 输出绑定
                    for key, shape_info in lut:
                        if shape_info is not None:
                            # 创建用于接收TRT输出的张量，确保是连续的
                            out_tensor = torch.zeros(size=shape_info, dtype=torch.float32, device=dev).contiguous()
                            trt_outputs_dict[key] = out_tensor
                            trt_bindings.append(out_tensor.data_ptr())
                            print(f"    添加输出绑定: {key} (形状: {shape_info})")
                        else:
                            # 对于None的输出，TRT引擎中不应该有对应的绑定，或者你需要处理这种情况
                            trt_outputs_dict[key] = None 
                            print(f"    跳过输出绑定 (lut值为None): {key}")
                    
                    print(f"  总绑定数量: {len(trt_bindings)}")
                    # 你可以打印 current_infer_engine.num_bindings 来比较数量是否匹配

                    print(f"  调用 TRT {engine_name_for_debug}.forward()...")
                    current_infer_engine.forward(trt_bindings)
                    torch.cuda.synchronize()
                    print(f"  TRT {engine_name_for_debug}.forward() 调用完成。")

                    # 打印TRT输出信息
                    for key, out_tensor_val in trt_outputs_dict.items():
                        if out_tensor_val is not None:
                            print_tensor_info(out_tensor_val, f"TRT 输出 {key}", indent=2)
                    
                    print(f"--- TrtPtsBboxHeadHelper: trt_pts_bbox_head_fwd ({engine_name_for_debug}) 执行结束 ---")
                    return trt_outputs_dict # 返回包含TRT输出张量的字典
                return trt_pts_bbox_head_fwd
            
        h2 = TrtPtsBboxHeadHelper()
        # 存储原始的PyTorch函数以便回退
        if hasattr(model.module.pts_bbox_head, 'forward'):
             h2.original_pytorch_func = ch.hooks["vadv1.pts_bbox_head.forward"].func # 假设 func 存储了原始函数
        
        # 仅当至少一个TRT引擎成功加载时才进行patch
        if h2.infer_fwd or h2.infer_prev:
            if h1.infer_engine: # 确保h1也成功了
                ch.hooks["vadv1.pts_bbox_head.forward"]._patch(h2.get_func())
                print("已尝试patch vadv1.pts_bbox_head.forward 使用TRT版本。")
            else:
                print("警告: TrtExtractImgFeatHelper (h1) 的引擎加载失败，因此跳过对 PtsBboxHead (h2) 的TRT patch。")
        else:
            print("警告: TrtPtsBboxHeadHelper 的两个引擎 (fwd 和 prev) 都加载失败，vadv1.pts_bbox_head.forward 将不会被TRT替换。")


        outputs = []
        prog_bar = mmcv.ProgressBar(len(data_loader.dataset)) # 使用 data_loader.dataset

        # --- PyTorch 对比运行 (可选，但对调试很有用) ---
        # 你可以在这里或循环内部运行一次PyTorch模型以获取参考输出
        # pytorch_outputs_for_comparison = []
        # print("\n运行PyTorch模型以获取参考输出...")
        # with torch.no_grad():
        #     for i_pytorch, data_pytorch in enumerate(data_loader):
        #         if i_pytorch >= 1: break # 只运行一个样本进行对比
        #         # 确保数据在GPU上
        #         # data_pytorch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v_list in data_pytorch.items() for v in (v_list if isinstance(v_list, list) else [v_list])}
        #         # 这里的数据传递给 DataParallel 包装的模型
        #         # result_pytorch = model_pytorch(data_pytorch, return_loss=False, rescale=True)
        #         # 你可能需要更细致地提取中间特征进行对比
        #         print(f"  PyTorch参考运行样本 {i_pytorch} 完成。")
        # print("PyTorch参考运行结束。\n")
        # --- 结束 PyTorch 对比运行 ---


        for i, data in enumerate(data_loader):
            print(f"\n===== 处理数据批次 {i} =====")
            # 如果只想测试少量样本，可以取消下面的注释
            if i >= 2:
                print(f"已达到最大测试批次数量 ({i})，停止。")
                break
            with torch.no_grad():           
                # result = model(return_loss=False, rescale=True, **data)
                # 确保数据在GPU上 (如果dataloader没有自动处理)
                # data_on_gpu = {}
                # for key, value in data.items():
                #     if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                #         data_on_gpu[key] = [v.cuda() for v in value]
                #     elif isinstance(value, torch.Tensor):
                #         data_on_gpu[key] = value.cuda()
                #     else: # img_metas 等
                #         data_on_gpu[key] = value
                
                print(f"--- TRT模型 ({type(model)}) 推理开始 (批次 {i}) ---")
                result = model(data, return_loss=False, rescale=True) # 使用原始的 data 字典
                print(f"--- TRT模型 推理结束 (批次 {i}) ---")
                outputs.extend(result)
            
            # 更新进度条的逻辑 (从原始代码复制)
            current_batch_size = 0
            if 'img_metas' in data and isinstance(data['img_metas'], list) and len(data['img_metas']) > 0:
                 if isinstance(data['img_metas'][0], mmcv.parallel.DataContainer):
                    current_batch_size = len(data['img_metas'][0].data[0])
                 elif isinstance(data['img_metas'][0], list):
                    current_batch_size = len(data['img_metas'][0])
            elif 'img' in data and isinstance(data['img'], list) and len(data['img']) > 0:
                 if isinstance(data['img'][0], mmcv.parallel.DataContainer):
                    if torch.is_tensor(data['img'][0].data) and data['img'][0].data.ndim > 0:
                        current_batch_size = data['img'][0].data.size(0)
                    elif isinstance(data['img'][0].data, list):
                         current_batch_size = len(data['img'][0].data)

            if current_batch_size == 0 and hasattr(data_loader.sampler, 'batch_size'):
                 current_batch_size = data_loader.sampler.batch_size
            
            if current_batch_size == 0:
                current_batch_size = samples_per_gpu

            for _ in range(current_batch_size):
                prog_bar.update()
        prog_bar.file.write("\n")
    else: # 分布式测试 (保持不变)
        model = DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)

    tmp = {}
    tmp['bbox_results'] = outputs
    outputs = tmp
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\n将结果写入 {args.out}')
            if isinstance(outputs, list): # 应该不会是list，因为被包装在tmp字典里了
                dump(outputs, args.out)
            else:
                dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        
        # 确保目录存在
        json_prefix_dir_name = args.config.split('/')[-1].split('.')[-2] if '.' in args.config else args.config.split('/')[-1]
        json_prefix_path_dir = osp.join('test', json_prefix_dir_name)
        os.makedirs(json_prefix_path_dir, exist_ok=True)
        
        kwargs['jsonfile_prefix'] = osp.join(json_prefix_path_dir, time.ctime().replace(' ', '_').replace(':', '_'))
        
        if args.format_only:
            dataset.format_results(outputs['bbox_results'], **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            print(dataset.evaluate(outputs['bbox_results'], **eval_kwargs))

if __name__ == '__main__':
    main()