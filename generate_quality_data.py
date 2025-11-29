"""
生成高质量的NER训练数据 - 修复版
"""

import random

# 实体库
ENTITIES = {
    'PER': [
        '马云', '马化腾', '李彦宏', '刘强东', '雷军', '张一鸣',
        '任正非', '董明珠', '宗庆后', '柳传志',
        '袁隆平', '钟南山', '屠呦呦', '钱学森', '邓稼先',
        '鲁迅', '老舍', '莫言', '刘慈欣', '金庸',
        '姚明', '刘翔', '李娜', '孙杨', '苏炳添',
        '周杰伦', '王菲', '刘德华', '张学友', '邓丽君',
        '张艺谋', '陈凯歌', '冯小刚', '李安', '贾樟柯',
        '毛泽东', '邓小平', '周恩来', '习近平', '李克强',
        '牛欢', '张伟', '王芳', '李明', '刘洋', '陈晨', '赵丽', '孙悦'
    ],
    'LOC': [
        '北京', '上海', '广州', '深圳', '杭州', '南京', '成都', '武汉',
        '西安', '重庆', '天津', '苏州', '长沙', '青岛', '大连', '厦门',
        '长江', '黄河', '珠江', '淮河', '黑龙江', '松花江',
        '泰山', '黄山', '峨眉山', '华山', '衡山', '恒山', '嵩山',
        '长城', '故宫', '天安门', '颐和园', '圆明园', '天坛',
        '西湖', '漓江', '九寨沟', '黄果树瀑布', '桂林', '张家界',
        '中国', '美国', '日本', '英国', '法国', '德国', '俄罗斯',
        '中北大学', '清华园', '未名湖', '嘉陵江', '洪崖洞', '太原'
    ],
    'ORG': [
        '阿里巴巴', '腾讯', '百度', '华为', '小米', '字节跳动',
        '京东', '美团', '拼多多', '滴滴出行', '比亚迪',
        '中国银行', '工商银行', '建设银行', '农业银行', '招商银行',
        '清华大学', '北京大学', '复旦大学', '浙江大学', '南京大学',
        '中北大学', '山西大学', '太原理工大学',
        '中国科学院', '中国工程院', '科技部', '教育部',
        '国家博物馆', '故宫博物院', '中国美术馆',
        '中央电视台', '人民日报', '新华社', '光明日报',
        '中国航天局', '中国航空公司', '中国铁路总公司',
        '联合国', '世界卫生组织', '国际奥委会'
    ]
}

# 句子模板
TEMPLATES = [
    # 单实体
    ('{PER}是一位杰出的科学家', ['PER']),
    ('{PER}来自{LOC}', ['PER', 'LOC']),
    ('{PER}在{ORG}工作', ['PER', 'ORG']),
    ('{PER}在{ORG}学习', ['PER', 'ORG']),
    ('{LOC}是一个美丽的城市', ['LOC']),
    ('{ORG}是知名企业', ['ORG']),
    ('{ORG}发展迅速', ['ORG']),

    # 双实体
    ('{PER}创立了{ORG}', ['PER', 'ORG']),
    ('{PER}毕业于{ORG}', ['PER', 'ORG']),
    ('{PER}在{LOC}出生', ['PER', 'LOC']),
    ('{PER}生活在{LOC}', ['PER', 'LOC']),
    ('{ORG}位于{LOC}', ['ORG', 'LOC']),
    ('{ORG}总部在{LOC}', ['ORG', 'LOC']),
    ('{ORG}坐落在{LOC}', ['ORG', 'LOC']),
    ('{PER}就读于{ORG}', ['PER', 'ORG']),

    # 三实体
    ('{PER}在{LOC}创办了{ORG}', ['PER', 'LOC', 'ORG']),
    ('{PER}从{ORG}搬到了{LOC}', ['PER', 'ORG', 'LOC']),
    ('{ORG}的{PER}来自{LOC}', ['ORG', 'PER', 'LOC']),
    ('{PER}在{LOC}的{ORG}工作', ['PER', 'LOC', 'ORG']),

    # 复杂句式
    ('{PER}和{PER}一起创立了{ORG}', ['PER', 'PER', 'ORG']),
    ('{PER}从{LOC}来到{LOC}工作', ['PER', 'LOC', 'LOC']),
    ('{ORG}与{ORG}达成合作', ['ORG', 'ORG']),
    ('{PER}曾在{ORG}和{ORG}任职', ['PER', 'ORG', 'ORG']),

    # 真实场景
    ('{PER}在{ORG}担任首席执行官', ['PER', 'ORG']),
    ('{PER}教授在{ORG}进行研究', ['PER', 'ORG']),
    ('{LOC}的{ORG}非常有名', ['LOC', 'ORG']),
    ('{PER}访问了{LOC}的{ORG}', ['PER', 'LOC', 'ORG']),
    ('{PER}考上了{ORG}', ['PER', 'ORG']),
    ('{PER}去{LOC}旅游', ['PER', 'LOC']),
]


def generate_sentence():
    """生成一个训练样本"""
    template, entity_types = random.choice(TEMPLATES)

    # 为每个占位符选择实体（处理重复类型）
    entity_counter = {}
    entity_values = []

    for etype in entity_types:
        # 统计每种类型出现的次数
        if etype not in entity_counter:
            entity_counter[etype] = 0
        entity_counter[etype] += 1

        # 选择实体
        entity = random.choice(ENTITIES[etype])
        entity_values.append((etype, entity))

    # 替换模板中的占位符
    sentence = template
    entity_positions = []  # 存储 (起始位置, 实体文本, 实体类型)

    for etype, entity in entity_values:
        placeholder = f'{{{etype}}}'
        if placeholder in sentence:
            pos = sentence.find(placeholder)
            entity_positions.append((pos, entity, etype))
            sentence = sentence.replace(placeholder, entity, 1)

    # 生成BIO标注
    words = list(sentence)
    tags = ['O'] * len(words)

    # 按位置排序（因为替换后位置会变化）
    # 需要重新计算实际位置
    current_sentence = template
    current_pos = 0

    for etype, entity in entity_values:
        placeholder = f'{{{etype}}}'
        if placeholder in current_sentence:
            # 找到占位符位置
            placeholder_pos = current_sentence.find(placeholder)

            # 计算实体在最终句子中的实际位置
            actual_pos = placeholder_pos + (current_pos - placeholder_pos)

            # 标注 B-TYPE 和 I-TYPE
            for i, char in enumerate(entity):
                if i == 0:
                    tags[actual_pos + i] = f'B-{etype}'
                else:
                    tags[actual_pos + i] = f'I-{etype}'

            # 更新句子和位置
            current_sentence = current_sentence.replace(placeholder, entity, 1)
            current_pos = actual_pos + len(entity)

    return words, tags


def generate_dataset(num_samples=2000, train_ratio=0.8):
    """生成数据集"""
    print(f"🔄 生成 {num_samples} 个样本...")

    samples = []
    seen_sentences = set()  # 去重

    attempts = 0
    max_attempts = num_samples * 3  # 最多尝试3倍次数

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            words, tags = generate_sentence()
            sentence_str = ''.join(words)

            # 去重
            if sentence_str not in seen_sentences:
                samples.append((words, tags))
                seen_sentences.add(sentence_str)

                # 进度提示
                if len(samples) % 500 == 0:
                    print(f"  已生成 {len(samples)}/{num_samples} 个样本...")
        except Exception as e:
            print(f"  ⚠️  生成失败: {e}")
            continue

    print(f"  ✅ 成功生成 {len(samples)} 个样本")

    # 划分训练集和测试集
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    return train_samples, test_samples


def save_dataset(samples, filename):
    """保存数据集"""
    with open(filename, 'w', encoding='utf-8') as f:
        for words, tags in samples:
            for word, tag in zip(words, tags):
                f.write(f'{word} {tag}\n')
            f.write('\n')

    print(f"  ✅ 保存 {len(samples)} 个样本到 {filename}")


def show_samples(samples, num=5):
    """显示样本示例"""
    print(f"\n📖 样本示例 (前{num}个):")
    print("-"*70)

    for i, (words, tags) in enumerate(samples[:num], 1):
        sentence = ''.join(words)
        print(f"\n[{i}] {sentence}")
        print(f"    ", end="")

        j = 0
        while j < len(words):
            if tags[j].startswith('B-'):
                entity_type = tags[j][2:]
                entity_chars = [words[j]]
                k = j + 1
                while k < len(tags) and tags[k] == f'I-{entity_type}':
                    entity_chars.append(words[k])
                    k += 1
                entity_text = ''.join(entity_chars)
                print(f"[{entity_type}:{entity_text}]", end=" ")
                j = k
            else:
                print(words[j], end="")
                j += 1
        print()


def main():
    print("="*70)
    print(" "*20 + "生成高质量NER数据")
    print("="*70)

    # 生成数据
    train_samples, test_samples = generate_dataset(num_samples=2000)

    # 显示样本
    show_samples(train_samples, num=10)

    # 保存
    print("\n" + "="*70)
    print("💾 保存数据...")
    print("="*70)
    save_dataset(train_samples, 'data/train_new.txt')
    save_dataset(test_samples, 'data/test_new.txt')

    # 统计信息
    print("\n" + "="*70)
    print("📊 数据统计")
    print("="*70)
    print(f"  训练集: {len(train_samples)} 个样本")
    print(f"  测试集: {len(test_samples)} 个样本")

    # 实体统计
    from collections import Counter
    entity_counter = Counter()

    for _, tags in train_samples:
        for tag in tags:
            if tag != 'O':
                entity_type = tag.split('-')[1]
                entity_counter[entity_type] += 1

    print(f"\n  实体分布 (训练集):")
    total_entities = sum(entity_counter.values())
    for entity_type, count in sorted(entity_counter.items()):
        percentage = count / total_entities * 100
        print(f"    {entity_type}: {count} ({percentage:.1f}%)")

    print("\n" + "="*70)
    print("✅ 数据生成完成！")
    print("="*70)
    print("\n下一步:")
    print("  1. 检查生成的数据:")
    print("     cat data/train_new.txt | head -50")
    print("\n  2. 使用新数据:")
    print("     mv data/train.txt data/train_old.txt")
    print("     mv data/test.txt data/test_old.txt")
    print("     mv data/train_new.txt data/train.txt")
    print("     mv data/test_new.txt data/test.txt")
    print("\n  3. 重新训练:")
    print("     python train.py")
    print("="*70)


if __name__ == '__main__':
    main()