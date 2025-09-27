import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
from sklearn import metrics
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score, \
recall_score,f1_score,roc_auc_score,confusion_matrix
np.random.seed(0)
import scipy
from matplotlib import pyplot as plt
import os
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
import joblib

# 新增图像分割相关的导入
from ultralytics import YOLO
import cv2
from PIL import Image
import torch

project_dir = '*****'
#project_dir = '*****'

# 修改路径为图像分割相关
image_dir = os.path.join(project_dir, 'segmentation_images')  # 图像目录
mask_dir = os.path.join(project_dir, 'ground_truth_masks')    # 真实掩码目录（用于评估）
model_path = os.path.join(project_dir, 'result_model12_pre/0323/segment/model1/best.pt')  # 分割模型路径
output_dir = os.path.join(project_dir, 'segmentation_results')  # 输出目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

def calculate_segmentation_metrics(true_mask, pred_mask):
    """
    计算分割任务的评估指标
    true_mask: 真实分割掩码（二值或多类别）
    pred_mask: 预测分割掩码（二值或多类别）
    """
    # 将掩码展平为一维数组
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()
    
    # 确保掩码是二值的（对于多类别分割，需要按类别处理）
    if len(np.unique(true_flat)) > 2:
        # 多类别分割，计算每个类别的指标
        unique_classes = np.unique(true_flat)
        metrics_dict = {}
        
        for class_id in unique_classes:
            if class_id == 0:  # 通常0是背景
                continue
                
            # 创建二值掩码 for 当前类别
            true_binary = (true_flat == class_id).astype(np.uint8)
            pred_binary = (pred_flat == class_id).astype(np.uint8)
            
            if np.sum(true_binary) == 0:  # 如果真实掩码中没有该类
                continue
                
            # 计算交集、并集等
            intersection = np.logical_and(true_binary, pred_binary).sum()
            union = np.logical_or(true_binary, pred_binary).sum()
            total_pixels = true_binary.size
            
            # IoU (Jaccard指数)
            iou = intersection / (union + 1e-6)
            
            # Dice系数
            dice = 2 * intersection / (np.sum(true_binary) + np.sum(pred_binary) + 1e-6)
            
            # 像素精度
            accuracy = np.sum(true_binary == pred_binary) / total_pixels
            
            # 精确率、召回率
            precision = intersection / (np.sum(pred_binary) + 1e-6)
            recall = intersection / (np.sum(true_binary) + 1e-6)
            
            metrics_dict[class_id] = {
                'iou': iou,
                'dice': dice,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
        
        return metrics_dict
    else:
        # 二值分割
        true_binary = true_flat
        pred_binary = pred_flat
        
        intersection = np.logical_and(true_binary, pred_binary).sum()
        union = np.logical_or(true_binary, pred_binary).sum()
        total_pixels = true_binary.size
        
        iou = intersection / (union + 1e-6)
        dice = 2 * intersection / (np.sum(true_binary) + np.sum(pred_binary) + 1e-6)
        accuracy = np.sum(true_binary == pred_binary) / total_pixels
        precision = intersection / (np.sum(pred_binary) + 1e-6)
        recall = intersection / (np.sum(true_binary) + 1e-6)
        
        return {
            'iou': iou,
            'dice': dice,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

def visualize_segmentation_results(image, true_mask, pred_mask, output_path):
    """可视化分割结果"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 真实掩码
    axes[1].imshow(true_mask, cmap='jet')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # 预测掩码
    axes[2].imshow(pred_mask, cmap='jet')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    # 重叠显示
    axes[3].imshow(image)
    axes[3].imshow(pred_mask, alpha=0.5, cmap='jet')
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 加载分割模型[4](@ref)
try:
    model = YOLO(model_path)  # 加载YOLOv8分割模型
    print(f"模型加载成功: {model_path}")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)

# 获取测试图像列表
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"找到 {len(image_files)} 张测试图像")

# 存储所有图像的指标
all_metrics = []

for image_file in image_files:
    print(f"处理图像: {image_file}")
    
    # 加载图像[6](@ref)
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 进行分割推理[1,4](@ref)
    results = model(image_path)
    
    if len(results) == 0 or results[0].masks is None:
        print(f"未检测到分割目标: {image_file}")
        continue
    
    # 获取第一个结果（假设单张图像）
    result = results[0]
    
    # 创建预测掩码[1](@ref)
    if result.masks.data is not None:
        # 获取掩码数据并转换为numpy数组
        masks = result.masks.data.cpu().numpy()
        
        # 合并所有实例的掩码（对于语义分割）
        pred_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for i, mask in enumerate(masks):
            # 调整掩码尺寸到原图大小[3](@ref)
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # 应用阈值创建二值掩码
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            
            # 为每个实例分配不同的标签值
            pred_mask[mask_binary == 1] = i + 1  # 从1开始编号
        
        # 加载真实掩码（如果存在）
        mask_file = os.path.splitext(image_file)[0] + '_mask.png'
        mask_path = os.path.join(mask_dir, mask_file)
        
        if os.path.exists(mask_path):
            true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # 计算分割指标
            metrics_result = calculate_segmentation_metrics(true_mask, pred_mask)
            
            if isinstance(metrics_result, dict) and 'iou' in metrics_result:
                # 二值分割结果
                all_metrics.append({
                    'image': image_file,
                    'iou': metrics_result['iou'],
                    'dice': metrics_result['dice'],
                    'accuracy': metrics_result['accuracy'],
                    'precision': metrics_result['precision'],
                    'recall': metrics_result['recall']
                })
                
                print(f"图像 {image_file} 的指标 - IoU: {metrics_result['iou']:.4f}, "
                      f"Dice: {metrics_result['dice']:.4f}, 准确率: {metrics_result['accuracy']:.4f}")
            else:
                # 多类别分割结果
                print(f"图像 {image_file} - 多类别分割")
                for class_id, class_metrics in metrics_result.items():
                    print(f"  类别 {class_id}: IoU: {class_metrics['iou']:.4f}, "
                          f"Dice: {class_metrics['dice']:.4f}")
            
            # 可视化并保存结果
            output_path = os.path.join(output_dir, f"result_{os.path.splitext(image_file)[0]}.png")
            visualize_segmentation_results(image, true_mask, pred_mask, output_path)
        
        else:
            print(f"未找到真实掩码: {mask_path}，仅进行预测")
            # 保存预测结果
            output_path = os.path.join(output_dir, f"pred_{os.path.splitext(image_file)[0]}.png")
            cv2.imwrite(output_path, pred_mask * 255)  # 转换为0-255范围
        
        # 保存掩码数据
        np.save(os.path.join(output_dir, f"mask_{os.path.splitext(image_file)[0]}.npy"), pred_mask)

# 计算平均指标
if all_metrics:
    metrics_df = pd.DataFrame(all_metrics)
    avg_metrics = metrics_df.mean(numeric_only=True)
    
    print("\n=== 平均分割指标 ===")
    print(f"平均IoU: {avg_metrics['iou']:.4f}")
    print(f"平均Dice系数: {avg_metrics['dice']:.4f}")
    print(f"平均准确率: {avg_metrics['accuracy']:.4f}")
    print(f"平均精确率: {avg_metrics['precision']:.4f}")
    print(f"平均召回率: {avg_metrics['recall']:.4f}")
    
    # 保存指标到文件
    metrics_df.to_csv(os.path.join(output_dir, 'segmentation_metrics.csv'), index=False)
    print(f"\n详细指标已保存到: {os.path.join(output_dir, 'segmentation_metrics.csv')}")

print(f"\n分割完成！结果保存在: {output_dir}")
