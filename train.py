"""
训练BiLSTM+CRF模型
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import json
import time

from models.bilstm_crf import BiLSTM_CRF
from utils.data_loader import load_data, create_dataloader
from utils.vocab import Vocabulary
from utils.metrics import compute_metrics, compute_metrics_by_type


def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    batch_count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        words = batch['words'].to(device)
        tags = batch['tags'].to(device)
        lengths = batch['lengths']

        # 创建mask
        mask = torch.arange(words.size(1)).expand(len(lengths), -1).to(device) < lengths.unsqueeze(1)

        # 前向传播
        loss = model(words, tags, mask)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / batch_count


def evaluate(model, dataloader, vocab, device, epoch):
    """评估模型"""
    model.eval()

    all_true_tags = []
    all_pred_tags = []
    all_words = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Eval]")

    with torch.no_grad():
        for batch in pbar:
            words = batch['words'].to(device)
            tags = batch['tags']
            lengths = batch['lengths']

            # 创建mask
            mask = torch.arange(words.size(1)).expand(len(lengths), -1).to(device) < lengths.unsqueeze(1)

            # 预测
            predictions = model(words, mask=mask)

            # 转换回标签
            for i, (pred, length) in enumerate(zip(predictions, lengths)):
                true_tag_seq = [vocab.get_tag(tags[i][j].item()) for j in range(length)]
                pred_tag_seq = [vocab.get_tag(tag_idx) for tag_idx in pred[:length]]
                word_seq = [vocab.get_word(words[i][j].item()) for j in range(length)]

                all_true_tags.append(true_tag_seq)
                all_pred_tags.append(pred_tag_seq)
                all_words.append(word_seq)

    # 计算指标
    metrics = compute_metrics(all_true_tags, all_pred_tags, all_words)
    metrics_by_type = compute_metrics_by_type(all_true_tags, all_pred_tags, all_words)

    return metrics, metrics_by_type


def save_checkpoint(model, optimizer, vocab, config, epoch, metrics, filepath):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab': vocab,
        'config': config,
        'metrics': metrics
    }, filepath)


def main():
    # 超参数配置
    config = {
        'embedding_dim': 100,
        'hidden_dim': 256,
        'num_layers': 2,  # 增加到2层LSTM
        'dropout': 0.5,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,  # L2正则化
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints',
        'patience': 10,  # 早停耐心值
    }

    print("=" * 70)
    print(" " * 20 + "BiLSTM+CRF 训练程序")
    print("=" * 70)
    print("\n配置参数:")
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    print("=" * 70)

    # 检查数据文件
    if not os.path.exists('data/train.txt'):
        print("\n❌ 错误: 找不到训练数据文件 data/train.txt")
        print("请先运行 python generate_data_with_deepseek.py 生成数据")
        return

    if not os.path.exists('data/test.txt'):
        print("\n❌ 错误: 找不到测试数据文件 data/test.txt")
        return

    # 加载数据
    print("\n📁 加载数据...")
    train_sentences, train_tags = load_data('data/train.txt')
    test_sentences, test_tags = load_data('data/test.txt')

    print(f"  训练集: {len(train_sentences)} 个句子")
    print(f"  测试集: {len(test_sentences)} 个句子")

    # 构建词表
    print("\n📚 构建词表...")
    vocab = Vocabulary()
    vocab.build_vocab(train_sentences, train_tags)
    print(f"  {vocab}")
    print(f"  标签: {[vocab.get_tag(i) for i in range(vocab.tag_size)]}")

    # 创建DataLoader
    print("\n🔄 创建数据加载器...")
    train_loader = create_dataloader(train_sentences, train_tags, vocab,
                                     config['batch_size'], shuffle=True)
    test_loader = create_dataloader(test_sentences, test_tags, vocab,
                                    config['batch_size'], shuffle=False)
    print(f"  训练批次: {len(train_loader)}")
    print(f"  测试批次: {len(test_loader)}")

    # 创建模型
    print("\n🏗️  创建模型...")
    model = BiLSTM_CRF(
        vocab_size=vocab.vocab_size,
        tag_size=vocab.tag_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    device = torch.device(config['device'])
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {num_params:,}")
    print(f"  可训练参数: {num_trainable:,}")
    print(f"  设备: {device}")

    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(),
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                  patience=5)

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)

    # 训练
    print("\n🚀 开始训练...")
    print("=" * 70)

    best_f1 = 0
    patience_counter = 0
    train_history = []

    start_time = time.time()

    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'=' * 70}")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # 评估
        metrics, metrics_by_type = evaluate(model, test_loader, vocab, device, epoch)

        # 学习率调整
        scheduler.step(metrics['f1'])
        current_lr = optimizer.param_groups[0]['lr']

        # 打印结果
        print(f"\n📊 训练损失: {train_loss:.4f}")
        print(f"📊 学习率: {current_lr:.6f}")
        print(f"\n整体指标:")
        print(f"  精确率 (Precision): {metrics['precision']:.4f}")
        print(f"  召回率 (Recall):    {metrics['recall']:.4f}")
        print(f"  F1分数 (F1-Score):  {metrics['f1']:.4f}")
        print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")

        print(f"\n各类型指标:")
        print(f"  {'类型':<8} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'支持数':<8}")
        print(f"  {'-' * 50}")
        for entity_type, scores in sorted(metrics_by_type.items()):
            print(f"  {entity_type:<8} {scores['precision']:<10.4f} "
                  f"{scores['recall']:<10.4f} {scores['f1']:<10.4f} "
                  f"{scores['support']:<8}")

        # 记录历史
        train_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'lr': current_lr
        })

        # 保存最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0

            best_model_path = os.path.join(config['save_dir'], 'best_model.pt')
            save_checkpoint(model, optimizer, vocab, config, epoch, metrics, best_model_path)

            print(f"\n💾 保存最佳模型 (F1={best_f1:.4f}) -> {best_model_path}")
        else:
            patience_counter += 1
            print(f"\n⏳ 未提升 ({patience_counter}/{config['patience']})")

        # 早停
        if patience_counter >= config['patience']:
            print(f"\n⚠️  早停触发！已连续{config['patience']}轮未提升")
            break

        # 定期保存检查点
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, vocab, config, epoch, metrics, checkpoint_path)
            print(f"💾 保存检查点 -> {checkpoint_path}")

    # 训练完成
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    print(f"  总耗时: {total_time / 60:.2f} 分钟")
    print(f"  最佳F1: {best_f1:.4f}")
    print(f"  最终轮次: {epoch}")
    print(f"  模型保存在: {config['save_dir']}/best_model.pt")

    # 保存训练历史
    history_path = os.path.join(config['save_dir'], 'train_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(train_history, f, indent=2, ensure_ascii=False)
    print(f"  训练历史: {history_path}")

    print("\n下一步:")
    print("  1. 运行 python predict.py 进行预测")
    print("  2. 运行 python demo.py 启动交互式演示")
    print("  3. 运行 python visualize.py 查看可视化")
    print("=" * 70)


if __name__ == '__main__':
    main()