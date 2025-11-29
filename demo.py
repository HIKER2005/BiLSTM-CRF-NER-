"""
交互式演示程序
"""

import torch
from predict import load_model, predict_sentence

try:
    from colorama import init, Fore, Style

    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("提示: 安装colorama可获得彩色输出 (pip install colorama)")


# 实体类型颜色映射
def get_entity_color(entity_type):
    """获取实体类型对应的颜色"""
    if not COLORAMA_AVAILABLE:
        return ''

    colors = {
        'PER': Fore.YELLOW,
        'LOC': Fore.GREEN,
        'ORG': Fore.CYAN,
    }
    return colors.get(entity_type, Fore.WHITE)


def colorize_entity(text, entity_type):
    """给实体添加颜色"""
    if not COLORAMA_AVAILABLE:
        return f"[{text}]"

    color = get_entity_color(entity_type)
    return f"{color}{Style.BRIGHT}{text}{Style.RESET_ALL}"


def display_result(words, tags, entities):
    """美化显示结果"""
    # 创建实体位置映射
    entity_map = {}
    for start, end, entity_type, entity_text in entities:
        for i in range(start, end):
            entity_map[i] = entity_type

    # 显示带颜色的句子
    print("\n" + "=" * 70)
    print("📋 标注结果:")
    print("-" * 70)

    # 显示原句（带实体高亮）
    print("\n原句（实体高亮）:")
    colored_words = []
    for i, word in enumerate(words):
        if i in entity_map:
            colored_word = colorize_entity(word, entity_map[i])
        else:
            colored_word = word
        colored_words.append(colored_word)
    print("  " + "".join(colored_words))

    # 显示详细标注
    print("\n详细标注:")
    print(f"  {'字符':<5} {'标签':<10}")
    print(f"  {'-' * 20}")
    for i, (word, tag) in enumerate(zip(words, tags)):
        if i in entity_map:
            if COLORAMA_AVAILABLE:
                color = get_entity_color(entity_map[i])
                print(f"  {color}{word:<5} {tag:<10}{Style.RESET_ALL}")
            else:
                print(f"  {word:<5} {tag:<10} *")
        else:
            print(f"  {word:<5} {tag:<10}")

    # 显示提取的实体
    print("\n" + "=" * 70)
    if entities:
        print("✅ 识别到的实体:")
        print("-" * 70)
        for start, end, entity_type, entity_text in entities:
            colored_text = colorize_entity(entity_text, entity_type)
            type_name = {'PER': '人名', 'LOC': '地名', 'ORG': '机构名'}.get(entity_type, entity_type)
            print(f"  [{type_name}] {colored_text} (位置: {start}:{end})")
    else:
        print("❌ 未识别到实体")

    print("=" * 70)


def display_help():
    """显示帮助信息"""
    print("\n" + "=" * 70)
    print("📖 使用帮助")
    print("=" * 70)
    print("\n支持的实体类型:")
    if COLORAMA_AVAILABLE:
        print(f"  {Fore.YELLOW}[PER]{Style.RESET_ALL} 人名 - 如：马云、姚明、刘德华")
        print(f"  {Fore.GREEN}[LOC]{Style.RESET_ALL} 地名 - 如：北京、上海、杭州")
        print(f"  {Fore.CYAN}[ORG]{Style.RESET_ALL} 机构名 - 如：阿里巴巴、清华大学、华为")
    else:
        print("  [PER] 人名 - 如：马云、姚明、刘德华")
        print("  [LOC] 地名 - 如：北京、上海、杭州")
        print("  [ORG] 机构名 - 如：阿里巴巴、清华大学、华为")

    print("\n命令:")
    print("  输入句子 - 进行实体识别")
    print("  help    - 显示此帮助")
    print("  example - 查看示例")
    print("  quit    - 退出程序")
    print("=" * 70)


def display_examples():
    """显示示例"""
    print("\n" + "=" * 70)
    print("💡 示例句子")
    print("=" * 70)
    examples = [
        "马云创立了阿里巴巴集团",
        "清华大学位于北京海淀区",
        "姚明是中国著名的篮球运动员",
        "华为公司总部在深圳",
        "刘德华来自香港",
        "故宫是北京的著名景点",
    ]
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    print("=" * 70)


def main():
    """主函数"""
    print("=" * 70)
    print(" " * 15 + "BiLSTM+CRF 命名实体识别演示")
    print("=" * 70)

    # 加载模型
    print("\n📦 正在加载模型...")

    import os
    model_path = 'checkpoints/best_model.pt'
    if not os.path.exists(model_path):
        print(f"\n❌ 错误: 找不到模型文件 {model_path}")
        print("请先运行 python train.py 训练模型")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab, config = load_model(model_path, device)

    print(f"✅ 模型加载完成！")
    print(f"  设备: {device}")
    print(f"  词表大小: {vocab.vocab_size}")

    display_help()

    print("\n🚀 开始识别 (输入 'help' 查看帮助):")
    print("-" * 70)

    while True:
        try:
            sentence = input("\n请输入句子: ").strip()

            if not sentence:
                continue

            # 处理命令
            if sentence.lower() == 'quit':
                print("\n👋 再见！")
                break

            elif sentence.lower() == 'help':
                display_help()
                continue

            elif sentence.lower() == 'example':
                display_examples()
                continue

            # 进行预测
            try:
                words, tags, entities = predict_sentence(model, vocab, sentence, device)
                display_result(words, tags, entities)
            except Exception as e:
                print(f"\n❌ 预测错误: {e}")
                continue

        except KeyboardInterrupt:
            print("\n\n👋 程序已中断，再见！")
            break
        except EOFError:
            print("\n\n👋 再见！")
            break


if __name__ == '__main__':
    main()