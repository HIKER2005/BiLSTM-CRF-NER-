"""
BiLSTM+CRF模型可视化 - 完整修复版
"""

import os
# 修复 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from models.bilstm_crf import BiLSTM_CRF

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_model(checkpoint_path='checkpoints/best_model.pt', device='cpu'):
    """加载训练好的模型"""
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    vocab = checkpoint['vocab']
    config = checkpoint['config']

    model = BiLSTM_CRF(
        vocab_size=vocab.vocab_size,
        tag_size=vocab.tag_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, vocab, config


def visualize_emissions(model, vocab, sentence, device='cpu', save_path='outputs/emissions_heatmap.png'):
    """可视化发射分数（BiLSTM输出）"""
    words = list(sentence.replace(' ', ''))
    word_indices = [vocab.get_word_idx(word) for word in words]
    words_tensor = torch.tensor([word_indices], dtype=torch.long).to(device)

    with torch.no_grad():
        emissions = model.get_emissions(words_tensor)
        emissions = emissions.squeeze(0).cpu().numpy()

    tag_names = [vocab.get_tag(i) for i in range(vocab.tag_size)]

    plt.figure(figsize=(12, max(6, len(words) * 0.4)))
    sns.heatmap(emissions,
                xticklabels=tag_names,
                yticklabels=words,
                cmap='RdYlGn',
                center=0,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': '发射分数'})

    plt.title(f'发射分数热力图\n句子: {sentence}', fontsize=14, pad=20)
    plt.xlabel('标签', fontsize=12)
    plt.ylabel('字符', fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 保存: {save_path}")
    plt.close()


def visualize_transitions(model, vocab, save_path='outputs/transitions_heatmap.png'):
    """可视化CRF转移矩阵"""
    transitions = model.crf.transitions.detach().cpu().numpy()
    tag_names = [vocab.get_tag(i) for i in range(vocab.tag_size)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(transitions,
                xticklabels=tag_names,
                yticklabels=tag_names,
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': '转移分数'})

    plt.title('CRF 转移矩阵', fontsize=14, pad=20)
    plt.xlabel('到标签', fontsize=12)
    plt.ylabel('从标签', fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 保存: {save_path}")
    plt.close()


def visualize_viterbi_path(model, vocab, sentence, device='cpu', save_path='outputs/viterbi_path.png'):
    """可视化Viterbi解码路径"""
    words = list(sentence.replace(' ', ''))
    word_indices = [vocab.get_word_idx(word) for word in words]
    words_tensor = torch.tensor([word_indices], dtype=torch.long).to(device)

    with torch.no_grad():
        emissions = model.get_emissions(words_tensor)
        mask = torch.ones_like(words_tensor, dtype=torch.bool)
        # 修复：使用 _viterbi_decode（带下划线）
        predictions = model.crf._viterbi_decode(emissions, mask)
        pred_tags = predictions[0]

    tag_names = [vocab.get_tag(tag_idx) for tag_idx in pred_tags]

    fig, ax = plt.subplots(figsize=(max(10, len(words) * 0.8), 6))

    x = range(len(words))
    y = pred_tags

    ax.plot(x, y, 'o-', linewidth=2, markersize=10, label='预测路径')

    ax.set_yticks(range(vocab.tag_size))
    ax.set_yticklabels([vocab.get_tag(i) for i in range(vocab.tag_size)])

    ax.set_xticks(x)
    ax.set_xticklabels(words, fontsize=12)

    for i, (word, tag_idx, tag_name) in enumerate(zip(words, pred_tags, tag_names)):
        ax.annotate(tag_name,
                   xy=(i, tag_idx),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('字符', fontsize=12)
    ax.set_ylabel('标签', fontsize=12)
    ax.set_title(f'Viterbi 解码路径\n句子: {sentence}', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 保存: {save_path}")
    plt.close()


def visualize_training_history(history_path='checkpoints/train_history.json',
                               save_path='outputs/training_history.png'):
    """可视化训练历史"""
    import json

    if not os.path.exists(history_path):
        print(f"  ⚠️  找不到训练历史文件: {history_path}")
        return

    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    precision = [h['precision'] for h in history]
    recall = [h['recall'] for h in history]
    f1 = [h['f1'] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss曲线
    axes[0, 0].plot(epochs, train_loss, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('训练损失', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)

    # F1曲线
    axes[0, 1].plot(epochs, f1, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('F1 Score', fontsize=12)
    axes[0, 1].set_title('F1分数', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    # 精确率和召回率
    axes[1, 0].plot(epochs, precision, 'r-', linewidth=2, label='Precision')
    axes[1, 0].plot(epochs, recall, 'b-', linewidth=2, label='Recall')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_title('精确率 & 召回率', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 综合对比
    axes[1, 1].plot(epochs, precision, 'r-', linewidth=2, label='Precision', alpha=0.7)
    axes[1, 1].plot(epochs, recall, 'b-', linewidth=2, label='Recall', alpha=0.7)
    axes[1, 1].plot(epochs, f1, 'g-', linewidth=2, label='F1', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('综合指标', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('训练历史', fontsize=16, y=1.00)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 保存: {save_path}")
    plt.close()


def main():
    print("="*70)
    print(" "*20 + "BiLSTM+CRF 可视化程序")
    print("="*70)

    model_path = 'checkpoints/best_model.pt'
    if not os.path.exists(model_path):
        print(f"\n❌ 错误: 找不到模型文件 {model_path}")
        print("请先运行 python train.py 训练模型")
        return

    print(f"\n📦 加载模型: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab, config = load_model(model_path, device)
    print("  ✅ 模型加载完成")

    default_sentence = "乔布斯创立了苹果公司"
    user_input = input(f"\n请输入要可视化的句子 (直接回车使用默认): ").strip()
    sentence = user_input if user_input else default_sentence

    print(f"\n🎨 可视化句子: {sentence}")
    print("="*70)

    print("\n📊 生成发射分数热力图...")
    visualize_emissions(model, vocab, sentence, device)

    print("\n📊 生成CRF转移矩阵...")
    visualize_transitions(model, vocab)

    print("\n📊 生成Viterbi解码路径...")
    visualize_viterbi_path(model, vocab, sentence, device)

    print("\n📊 生成训练历史曲线...")
    visualize_training_history()

    print("\n" + "="*70)
    print("✅ 可视化完成！")
    print("="*70)
    print("\n生成的图片保存在 outputs/ 目录:")
    print("  - emissions_heatmap.png    (发射分数热力图)")
    print("  - transitions_heatmap.png  (CRF转移矩阵)")
    print("  - viterbi_path.png         (Viterbi解码路径)")
    print("  - training_history.png     (训练历史曲线)")
    print("="*70)


if __name__ == '__main__':
    main()