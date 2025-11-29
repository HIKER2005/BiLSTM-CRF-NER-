"""
修复训练数据的标注格式问题
"""

import re
from collections import defaultdict


def fix_tag_format(tag):
    """统一标注格式为 B-TYPE / I-TYPE / O"""
    if tag == 'O':
        return 'O'

    # 处理 PER-B, LOC-I 这种格式 (颠倒格式)
    if re.match(r'^(PER|LOC|ORG)-([BI])$', tag):
        entity_type, prefix = tag.split('-')
        return f'{prefix}-{entity_type}'

    # 处理 PER, LOC, ORG 这种无前缀格式
    if tag in ['PER', 'LOC', 'ORG']:
        return f'B-{tag}'  # 假设是开始

    # 处理 BMES 格式
    if re.match(r'^[BME]-(PER|LOC|ORG)$', tag):
        prefix, entity_type = tag.split('-')
        if prefix in ['B']:
            return f'B-{entity_type}'
        else:  # M, E 都当作 I
            return f'I-{entity_type}'

    # 已经是正确格式
    if re.match(r'^[BI]-(PER|LOC|ORG)$', tag):
        return tag

    # 其他情况返回 O
    return 'O'


def fix_entity_boundaries(words, tags):
    """修复实体边界错误"""
    fixed_tags = []
    i = 0

    while i < len(tags):
        tag = tags[i]

        # 处理 B- 标签后面应该是 I- 的情况
        if tag.startswith('B-'):
            entity_type = tag[2:]
            fixed_tags.append(tag)
            i += 1

            # 检查后续标签
            while i < len(tags):
                next_tag = tags[i]

                # 如果是同类型的 B-，说明上一个实体结束了
                if next_tag == f'B-{entity_type}':
                    break

                # 如果是其他 B- 或 O，说明实体结束
                if next_tag.startswith('B-') or next_tag == 'O':
                    break

                # 修正为 I-TYPE
                if next_tag in [f'I-{entity_type}', entity_type, f'{entity_type}-I']:
                    fixed_tags.append(f'I-{entity_type}')
                else:
                    break

                i += 1
        else:
            fixed_tags.append(tag if tag == 'O' else 'O')
            i += 1

    return fixed_tags


def clean_data_file(input_file, output_file):
    """清洗数据文件"""
    print(f"\n🔄 处理文件: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 统计问题
    format_errors = defaultdict(int)
    boundary_errors = 0
    total_sentences = 0
    total_tokens = 0

    # 解析数据
    sentences = []
    current_words = []
    current_tags = []

    for line in lines:
        line = line.strip()
        if not line:
            if current_words:
                sentences.append((current_words, current_tags))
                current_words = []
                current_tags = []
            continue

        parts = line.split()
        if len(parts) == 2:
            word, tag = parts
            current_words.append(word)
            current_tags.append(tag)

    if current_words:
        sentences.append((current_words, current_tags))

    print(f"  原始句子数: {len(sentences)}")

    # 清洗数据
    cleaned_sentences = []

    for words, tags in sentences:
        total_sentences += 1
        total_tokens += len(words)

        # 1. 修复标注格式
        fixed_format_tags = []
        for tag in tags:
            original_tag = tag
            fixed_tag = fix_tag_format(tag)
            if original_tag != fixed_tag:
                format_errors[original_tag] += 1
            fixed_format_tags.append(fixed_tag)

        # 2. 修复实体边界
        fixed_tags = fix_entity_boundaries(words, fixed_format_tags)

        # 3. 验证标注合法性
        valid = True
        for i, tag in enumerate(fixed_tags):
            # I- 标签必须跟在 B- 或 I- 后面
            if tag.startswith('I-'):
                entity_type = tag[2:]
                if i == 0 or (not fixed_tags[i - 1].endswith(f'-{entity_type}')):
                    fixed_tags[i] = f'B-{entity_type}'  # 修正为 B-
                    boundary_errors += 1

        cleaned_sentences.append((words, fixed_tags))

    # 写入清洗后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for words, tags in cleaned_sentences:
            for word, tag in zip(words, tags):
                f.write(f'{word} {tag}\n')
            f.write('\n')

    print(f"  ✅ 清洗后句子数: {len(cleaned_sentences)}")
    print(f"  📊 格式错误修复: {sum(format_errors.values())} 处")
    if format_errors:
        print(f"     错误类型分布:")
        for error_tag, count in sorted(format_errors.items(), key=lambda x: -x[1])[:10]:
            print(f"       {error_tag}: {count} 次")
    print(f"  📊 边界错误修复: {boundary_errors} 处")
    print(f"  💾 已保存到: {output_file}")


def main():
    print("=" * 70)
    print(" " * 20 + "数据清洗工具")
    print("=" * 70)

    # 清洗训练集
    clean_data_file('data/train.txt', 'data/train_cleaned.txt')

    # 清洗测试集
    clean_data_file('data/test.txt', 'data/test_cleaned.txt')

    print("\n" + "=" * 70)
    print("✅ 数据清洗完成！")
    print("=" * 70)
    print("\n下一步:")
    print("  1. 备份原始数据:")
    print("     mv data/train.txt data/train_original.txt")
    print("     mv data/test.txt data/test_original.txt")
    print("\n  2. 使用清洗后的数据:")
    print("     mv data/train_cleaned.txt data/train.txt")
    print("     mv data/test_cleaned.txt data/test.txt")
    print("\n  3. 重新训练模型:")
    print("     python train.py")
    print("=" * 70)


if __name__ == '__main__':
    main()