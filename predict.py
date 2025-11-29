"""
使用训练好的模型进行预测
"""

import torch
from models.bilstm_crf import BiLSTM_CRF
from utils.metrics import extract_entities


def load_model(checkpoint_path='checkpoints/best_model.pt', device='cpu'):
    """加载训练好的模型"""
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'

    # 修改这里：添加 weights_only=False
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


def predict_sentence(model, vocab, sentence, device='cpu'):
    """
    预测单个句子

    Args:
        model: 训练好的模型
        vocab: 词表
        sentence: str或List[str] - 输入句子
        device: 设备

    Returns:
        words: List[str] - 分词结果
        tags: List[str] - 标签序列
        entities: List[tuple] - 提取的实体
    """
    # 如果输入是字符串，进行分词（简单按字符分）
    if isinstance(sentence, str):
        words = list(sentence.replace(' ', ''))
    else:
        words = sentence

    # 转换为索引
    word_indices = [vocab.get_word_idx(word) for word in words]
    words_tensor = torch.tensor([word_indices], dtype=torch.long).to(device)

    # 预测
    with torch.no_grad():
        predictions = model(words_tensor)

    # 转换回标签
    tags = [vocab.get_tag(tag_idx) for tag_idx in predictions[0]]

    # 提取实体
    entities = extract_entities(tags, words)

    return words, tags, entities


def predict_batch(model, vocab, sentences, device='cpu'):
    """
    批量预测

    Args:
        model: 训练好的模型
        vocab: 词表
        sentences: List[str] - 句子列表
        device: 设备

    Returns:
        results: List[dict] - 预测结果列表
    """
    results = []

    for sentence in sentences:
        words, tags, entities = predict_sentence(model, vocab, sentence, device)
        results.append({
            'sentence': sentence,
            'words': words,
            'tags': tags,
            'entities': entities
        })

    return results


def main():
    """主函数 - 测试预测功能"""
    import os

    print("=" * 70)
    print(" " * 20 + "BiLSTM+CRF 预测程序")
    print("=" * 70)

    # 检查模型文件
    model_path = 'checkpoints/best_model.pt'
    if not os.path.exists(model_path):
        print(f"\n❌ 错误: 找不到模型文件 {model_path}")
        print("请先运行 python train.py 训练模型")
        return

    # 加载模型
    print(f"\n📦 加载模型: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab, config = load_model(model_path, device)

    print(f"  设备: {device}")
    print(f"  词表大小: {vocab.vocab_size}")
    print(f"  标签数量: {vocab.tag_size}")
    print(f"  标签: {[vocab.get_tag(i) for i in range(vocab.tag_size)]}")

    # 测试句子
    test_sentences = [
        "我爱北京天安门",
        "乔布斯创立了苹果公司",
        "今天天气很好",
        "刘德华来自香港",
        "马云在杭州创办了阿里巴巴集团",
        "清华大学位于北京海淀区",
        "姚明是著名的篮球运动员",
        "华为公司总部在深圳",
    ]

    print("\n" + "=" * 70)
    print("🔍 开始预测...")
    print("=" * 70)

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n[{i}] 句子: {sentence}")
        print("-" * 70)

        words, tags, entities = predict_sentence(model, vocab, sentence, device)

        # 显示标注结果（表格形式）
        print("\n标注结果:")
        print(f"  {'字符':<5} {'标签':<10}")
        print(f"  {'-' * 20}")
        for word, tag in zip(words, tags):
            print(f"  {word:<5} {tag:<10}")

        # 显示提取的实体
        if entities:
            print("\n✅ 提取的实体:")
            for start, end, entity_type, entity_text in entities:
                print(f"  [{entity_type}] {entity_text} (位置: {start}:{end})")
        else:
            print("\n❌ 未识别到实体")

        print("=" * 70)

    print("\n✅ 预测完成！")
    print("\n提示: 运行 python demo.py 启动交互式预测界面")


if __name__ == '__main__':
    main()