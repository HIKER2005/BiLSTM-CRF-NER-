"""
项目配置文件
"""

import os
from pathlib import Path

# ==================== API配置 ====================
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-c9291c92289b4ed0a74c480ed929ca32')
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ==================== 数据生成配置 ====================
DATA_GENERATION_CONFIG = {
    'num_sentences': 300,           # 默认生成句子数
    'batch_size': 20,               # 每批生成数量
    'delay': 1.5,                   # API调用间隔（秒）
    'temperature': 0.7,             # 生成句子的温度
    'annotation_temperature': 0.3,  # 标注的温度（较低保证准确）
    'domains': [                    # 涵盖的领域
        '新闻', '科技', '娱乐', '体育',
        '商业', '历史', '文化', '教育'
    ]
}

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    'embedding_dim': 100,      # 词嵌入维度
    'hidden_dim': 256,         # LSTM隐藏层维度
    'num_layers': 1,           # LSTM层数
    'dropout': 0.5,            # Dropout比例
}

# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    'batch_size': 32,          # 批次大小
    'num_epochs': 50,          # 训练轮数
    'learning_rate': 0.001,    # 学习率
    'device': 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu',
    'save_dir': 'checkpoints', # 模型保存目录
    'log_interval': 10,        # 日志打印间隔
}

# ==================== 数据集配置 ====================
DATASET_CONFIG = {
    'generated_file': 'data/generated_data.txt',
    'train_file': 'data/train.txt',
    'test_file': 'data/test.txt',
    'test_ratio': 0.2,         # 测试集比例
}

# ==================== 实体类型定义 ====================
ENTITY_TYPES = {
    'PER': '人名',
    'LOC': '地名',
    'ORG': '机构名'
}

ENTITY_COLORS = {
    'PER': '\033[93m',  # 黄色
    'LOC': '\033[92m',  # 绿色
    'ORG': '\033[96m',  # 青色
}

# ==================== 项目路径 ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

# 创建必要的目录
DATA_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)