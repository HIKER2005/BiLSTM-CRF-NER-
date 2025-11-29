# """
# 使用DeepSeek API生成NER训练数据
# """
#
# import os
# import requests
# import json
# import time
# from tqdm import tqdm
# import re
#
#
# class DeepSeekDataGenerator:
#     def __init__(self, api_key, base_url="https://api.deepseek.com/v1"):
#         """
#         初始化DeepSeek数据生成器
#
#         Args:
#             api_key: DeepSeek API密钥
#             base_url: API基础URL
#         """
#         self.api_key = api_key
#         self.base_url = base_url
#         self.headers = {
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json"
#         }
#
#     def call_api(self, messages, temperature=0.7, max_tokens=2000):
#         """调用DeepSeek API"""
#         url = f"{self.base_url}/chat/completions"
#
#         payload = {
#             "model": "deepseek-chat",
#             "messages": messages,
#             "temperature": temperature,
#             "max_tokens": max_tokens
#         }
#
#         try:
#             response = requests.post(url, headers=self.headers, json=payload, timeout=30)
#             response.raise_for_status()
#             result = response.json()
#             return result['choices'][0]['message']['content']
#         except Exception as e:
#             print(f"API调用错误: {e}")
#             return None
#
#     def generate_sentences(self, num_sentences=50, domains=None):
#         """生成包含实体的句子"""
#         if domains is None:
#             domains = ['新闻', '科技', '娱乐', '体育', '商业', '历史', '文化']
#
#         domain_str = '、'.join(domains)
#
#         prompt = f"""请生成{num_sentences}个中文句子，要求：
#
# 1. 句子长度在10-30个字之间
# 2. 每个句子必须包含至少一个命名实体（人名、地名或机构名）
# 3. 涵盖以下领域：{domain_str}
# 4. 句子要自然、真实、多样化
# 5. 实体类型要均衡分布
#
# 每行一个句子，不要编号，直接输出句子。
#
# 示例：
# 马云创立了阿里巴巴集团
# 姚明是中国著名的篮球运动员
# 故宫位于北京市中心
# 苹果公司发布了新款iPhone
#
# 现在请生成{num_sentences}个句子："""
#
#
#         messages = [{"role": "user", "content": prompt}]
#
#         response = self.call_api(messages, temperature=0.8)
#
#         if response:
#             sentences = []
#             for line in response.strip().split('\n'):
#                 line = line.strip()
#                 line = re.sub(r'^\d+[\.\、\s]+', '', line)
#                 if line and len(line) >= 5:
#                     sentences.append(line)
#             return sentences
#
#         return []
#
#     def annotate_sentence(self, sentence):
#         """对单个句子进行实体标注"""
#         prompt = f"""请对下面的中文句子进行命名实体识别标注，使用BIO标注格式。
#
# 实体类型：
# - PER: 人名（如：马云、姚明、李白）
# - LOC: 地名（如：北京、上海、杭州、中国）
# - ORG: 机构名（如：阿里巴巴、清华大学、联合国）
#
# 标注格式：
# - B-XXX: 实体开始
# - I-XXX: 实体内部
# - O: 非实体
#
# 输出格式要求：
# 1. 每行一个字和对应的标签，用空格分隔
# 2. 按照句子顺序逐字标注
# 3. 不要有任何额外的解释或说明
# 4. 标注要准确、完整
#
# 示例输入：马云创立了阿里巴巴
# 示例输出：
# 马 B-PER
# 云 I-PER
# 创 O
# 立 O
# 了 O
# 阿 B-ORG
# 里 I-ORG
# 巴 I-ORG
# 巴 I-ORG
#
# 现在请标注以下句子：{sentence}
#
# 输出："""
#
#         messages = [{"role": "user", "content": prompt}]
#
#         response = self.call_api(messages, temperature=0.3)
#
#         if response:
#             lines = []
#             for line in response.strip().split('\n'):
#                 line = line.strip()
#                 if ' ' in line:
#                     parts = line.split()
#                     if len(parts) == 2 and len(parts[0]) == 1:
#                         lines.append(line)
#
#             annotated_chars = ''.join([line.split()[0] for line in lines])
#             if annotated_chars == sentence.replace(' ', ''):
#                 return '\n'.join(lines)
#             else:
#                 print(f"警告：标注不完整 - {sentence}")
#                 return None
#
#         return None
#
#     def generate_dataset(self, num_sentences=100, output_file='data/generated_data.txt',
#                          batch_size=10, delay=1.0):
#         """生成完整的数据集"""
#         os.makedirs(os.path.dirname(output_file), exist_ok=True)
#
#         all_data = []
#         num_batches = (num_sentences + batch_size - 1) // batch_size
#
#         print(f"开始生成数据集，总共{num_sentences}个句子，分{num_batches}批...")
#
#         for batch_idx in range(num_batches):
#             current_batch_size = min(batch_size, num_sentences - batch_idx * batch_size)
#
#             print(f"\n批次 {batch_idx + 1}/{num_batches}:")
#
#             # 生成句子
#             print(f"  生成{current_batch_size}个句子...")
#             sentences = self.generate_sentences(current_batch_size)
#
#             if not sentences:
#                 print("  句子生成失败，跳过此批次")
#                 continue
#
#             print(f"  成功生成{len(sentences)}个句子")
#
#             # 标注句子
#             print(f"  开始标注...")
#             for idx, sentence in enumerate(tqdm(sentences, desc="  标注进度")):
#                 annotated = self.annotate_sentence(sentence)
#
#                 if annotated:
#                     all_data.append(annotated)
#                 else:
#                     print(f"    句子标注失败: {sentence}")
#
#                 if idx < len(sentences) - 1:
#                     time.sleep(delay)
#
#             if batch_idx < num_batches - 1:
#                 time.sleep(delay)
#
#         print(f"\n成功生成{len(all_data)}个标注样本")
#
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write('\n\n'.join(all_data))
#
#         print(f"数据已保存到: {output_file}")
#
#         return len(all_data)
#
#     def validate_data(self, file_path):
#         """验证生成的数据质量"""
#         print(f"\n验证数据: {file_path}")
#
#         sentences = []
#         current_sentence = []
#
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     current_sentence.append(line)
#                 else:
#                     if current_sentence:
#                         sentences.append(current_sentence)
#                         current_sentence = []
#
#         if current_sentence:
#             sentences.append(current_sentence)
#
#         print(f"  总句子数: {len(sentences)}")
#
#         entity_counts = {'PER': 0, 'LOC': 0, 'ORG': 0}
#         total_tokens = 0
#         entity_tokens = 0
#
#         for sentence in sentences:
#             for line in sentence:
#                 parts = line.split()
#                 if len(parts) == 2:
#                     word, tag = parts
#                     total_tokens += 1
#
#                     if tag != 'O':
#                         entity_tokens += 1
#                         entity_type = tag.split('-')[-1]
#                         if entity_type in entity_counts:
#                             entity_counts[entity_type] += 1
#
#         print(f"  总字符数: {total_tokens}")
#         print(f"  实体字符数: {entity_tokens} ({entity_tokens / total_tokens * 100:.2f}%)")
#         print(f"  实体统计:")
#         for entity_type, count in entity_counts.items():
#             print(f"    {entity_type}: {count}")
#
#         return {
#             'num_sentences': len(sentences),
#             'total_tokens': total_tokens,
#             'entity_tokens': entity_tokens,
#             'entity_counts': entity_counts
#         }
#
#
# def split_train_test(input_file, train_file='data/train.txt',
#                      test_file='data/test.txt', test_ratio=0.2):
#     """将生成的数据分割为训练集和测试集"""
#     sentences = []
#     current_sentence = []
#
#     with open(input_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 current_sentence.append(line)
#             else:
#                 if current_sentence:
#                     sentences.append(current_sentence)
#                     current_sentence = []
#
#     if current_sentence:
#         sentences.append(current_sentence)
#
#     import random
#     random.shuffle(sentences)
#
#     split_idx = int(len(sentences) * (1 - test_ratio))
#     train_sentences = sentences[:split_idx]
#     test_sentences = sentences[split_idx:]
#
#     with open(train_file, 'w', encoding='utf-8') as f:
#         for sentence in train_sentences:
#             f.write('\n'.join(sentence) + '\n\n')
#
#     with open(test_file, 'w', encoding='utf-8') as f:
#         for sentence in test_sentences:
#             f.write('\n'.join(sentence) + '\n\n')
#
#     print(f"\n数据集分割完成:")
#     print(f"  训练集: {len(train_sentences)} 句 -> {train_file}")
#     print(f"  测试集: {len(test_sentences)} 句 -> {test_file}")
#
#
# def main():
#     """主函数"""
#     print("=" * 60)
#     print(" " * 15 + "DeepSeek NER数据生成器")
#     print("=" * 60)
#
#     api_key = os.getenv('DEEPSEEK_API_KEY')
#
#     if not api_key:
#         print("\n请输入您的DeepSeek API密钥:")
#         api_key = input("API Key: ").strip()
#
#         if not api_key:
#             print("错误：未提供API密钥")
#             return
#
#     generator = DeepSeekDataGenerator(api_key)
#
#     num_sentences = int(input("\n要生成多少个句子？(推荐200-500): ") or "300")
#     batch_size = 20
#     delay = 1.5
#
#     print(f"\n配置:")
#     print(f"  总句子数: {num_sentences}")
#     print(f"  批次大小: {batch_size}")
#     print(f"  调用间隔: {delay}秒")
#
#     output_file = 'data/generated_data.txt'
#     num_generated = generator.generate_dataset(
#         num_sentences=num_sentences,
#         output_file=output_file,
#         batch_size=batch_size,
#         delay=delay
#     )
#
#     if num_generated > 0:
#         stats = generator.validate_data(output_file)
#         split_train_test(output_file, test_ratio=0.2)
#
#         print("\n" + "=" * 60)
#         print("数据生成完成！")
#         print("=" * 60)
#         print("\n下一步：运行 python train.py 开始训练模型")
#     else:
#         print("\n数据生成失败，请检查API密钥和网络连接")
#
#
# if __name__ == '__main__':
#     main()
"""
使用 DeepSeek API 生成 NER 训练数据
带严格格式验证和自动修复
"""

import os
import json
import re
from openai import OpenAI
from collections import Counter

# DeepSeek API 配置
DEEPSEEK_API_KEY = "your_api_key_here"  # 替换为你的 API Key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 初始化客户端
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


def generate_prompt(num_samples=50):
    """生成严格的 prompt"""
    prompt = f"""你是一个专业的中文命名实体识别(NER)数据标注专家。

    ## 任务
    生成 {num_samples} 个高质量的中文 NER 训练样本。
    
    ## 严格要求
    
    ### 1. 标签体系（只能使用以下7种标签）
    - **B-PER**: 人名的第一个字（如：马云 → 马:B-PER）
    - **I-PER**: 人名的后续字（如：马云 → 云:I-PER）
    - **B-LOC**: 地名的第一个字
    - **I-LOC**: 地名的后续字
    - **B-ORG**: 机构名的第一个字
    - **I-ORG**: 机构名的后续字
    - **O**: 非实体字符
    
    ### 2. 绝对禁止的格式（会导致数据作废）
    ❌ 倒序格式：PER-B, LOC-I, ORG-B
    ❌ BMES格式：M-PER, E-ORG, S-LOC
    ❌ 无前缀：PER, LOC, ORG
    ❌ 其他变体：Person, Location, Organization
    
    ### 3. 标注规则
    - 每个实体的第一个字必须是 B- 标签
    - 每个实体的后续字必须是 I- 标签
    - 相同类型实体必须连续标注
    - 非实体字符必须标注为 O
    
    ### 4. 输出格式（严格遵循）
    字 标签
    字 标签
    ...
    <空行>
    字 标签
    ...

    ### 5. 实体类型示例
    - **人名(PER)**: 马云、马化腾、李彦宏、袁隆平、鲁迅、姚明、周杰伦
    - **地名(LOC)**: 北京、上海、长江、黄河、泰山、西湖、中国、美国
    - **机构(ORG)**: 阿里巴巴、清华大学、中国银行、联合国、中央电视台

    ## 标准示例

    示例1:
    马 B-PER
    云 I-PER
    创 O
    立 O
    了 O
    阿 B-ORG
    里 I-ORG
    巴 I-ORG
    巴 I-ORG

    示例2:
    袁 B-PER
    隆 I-PER
    平 I-PER
    在 O
    湖 B-LOC
    南 I-LOC
    工 O
    作 O

    示例3:
    清 B-ORG
    华 I-ORG
    大 I-ORG
    学 I-ORG
    位 O
    于 O
    北 B-LOC
    京 I-LOC

    ## 要求
    1. 每个句子长度 8-30 字
    2. 每个句子至少包含 1 个实体
    3. 实体类型要均衡分布
    4. 句子要符合中文语法，内容真实合理
    5. 严格使用上述7种标签，不得有任何偏差

    现在请生成 {num_samples} 个样本，严格按照上述格式输出。
    """
    return prompt


def call_deepseek_api(prompt, model="deepseek-chat", max_retries=3):
    """调用 DeepSeek API"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个严格遵循格式要求的NER数据标注专家。你只能使用 B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, O 这7种标签。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,  # 适度随机性
                max_tokens=4000,
                stream=False
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"  ⚠️  API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            continue


def fix_tag_format(tag):
    """修复标签格式"""
    tag = tag.strip().upper()

    # 已经是正确格式
    if tag in ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'O']:
        return tag

    # 处理倒序格式: PER-B → B-PER
    if re.match(r'^(PER|LOC|ORG)-([BI])$', tag):
        entity_type, prefix = tag.split('-')
        return f'{prefix}-{entity_type}'

    # 处理无前缀: PER → B-PER（默认为开始）
    if tag in ['PER', 'LOC', 'ORG']:
        return f'B-{tag}'

    # 处理 BMES 格式
    if re.match(r'^[BMES]-(PER|LOC|ORG)$', tag):
        prefix, entity_type = tag.split('-')
        if prefix == 'B':
            return f'B-{entity_type}'
        elif prefix in ['M', 'E']:  # M和E都当作I
            return f'I-{entity_type}'
        elif prefix == 'S':  # S(单字实体)当作B
            return f'B-{entity_type}'

    # 处理全称: PERSON → PER
    tag_mapping = {
        'PERSON': 'PER', 'PEOPLE': 'PER', '人名': 'PER',
        'LOCATION': 'LOC', 'PLACE': 'LOC', '地名': 'LOC',
        'ORGANIZATION': 'ORG', 'COMPANY': 'ORG', '机构': 'ORG'
    }

    for old, new in tag_mapping.items():
        if old in tag:
            if 'B' in tag or tag == old:
                return f'B-{new}'
            else:
                return f'I-{new}'

    # 无法识别，返回 O
    return 'O'


def fix_entity_boundaries(words, tags):
    """修复实体边界问题"""
    fixed_tags = []
    i = 0

    while i < len(tags):
        tag = tags[i]

        # 处理 I- 标签出现在开头或跟在 O 后面的情况
        if tag.startswith('I-'):
            entity_type = tag[2:]
            # 检查前一个标签
            if i == 0 or not fixed_tags[-1].endswith(f'-{entity_type}'):
                # 修正为 B-
                fixed_tags.append(f'B-{entity_type}')
            else:
                fixed_tags.append(tag)

        # 处理 B- 标签
        elif tag.startswith('B-'):
            fixed_tags.append(tag)

        # 处理 O 标签
        else:
            fixed_tags.append('O')

        i += 1

    return fixed_tags


def validate_sample(words, tags):
    """验证样本有效性"""
    if len(words) != len(tags):
        return False, "字符数和标签数不匹配"

    if len(words) < 5:
        return False, "句子太短"

    if len(words) > 100:
        return False, "句子太长"

    # 检查是否至少有一个实体
    has_entity = any(tag.startswith('B-') for tag in tags)
    if not has_entity:
        return False, "没有实体"

    # 检查标签合法性
    valid_tags = {'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'O'}
    for tag in tags:
        if tag not in valid_tags:
            return False, f"非法标签: {tag}"

    return True, "OK"


def parse_api_response(response_text):
    """解析 API 响应"""
    lines = response_text.strip().split('\n')

    samples = []
    current_words = []
    current_tags = []

    for line in lines:
        line = line.strip()

        # 跳过空行和markdown代码块标记
        if not line or line.startswith('```'):
            if current_words:
                samples.append((current_words, current_tags))
                current_words = []
                current_tags = []
            continue

            # 跳过说明性文字
        if '示例' in line or '要求' in line or line.startswith('#'):
            continue

            # 解析 "字 标签" 格式
        parts = line.split()
        if len(parts) == 2:
            word, tag = parts

            # 只保留单个字符
            if len(word) == 1:
                current_words.append(word)
                current_tags.append(tag)
        elif len(parts) == 1 and len(parts[0]) == 1:
            # 可能只有字符，没有标签
            current_words.append(parts[0])
            current_tags.append('O')

            # 添加最后一个样本
    if current_words:
        samples.append((current_words, current_tags))

    return samples


def clean_sample(words, tags):
    """清洗单个样本"""
    # 1. 修复标签格式
    fixed_tags = [fix_tag_format(tag) for tag in tags]

    # 2. 修复实体边界
    fixed_tags = fix_entity_boundaries(words, fixed_tags)

    return words, fixed_tags


def generate_data_with_deepseek(
        num_samples=200,
        batch_size=50,
        output_train='data/train_deepseek.txt',
        output_test='data/test_deepseek.txt',
        train_ratio=0.8
):
    """使用 DeepSeek 生成数据"""

    print("=" * 70)
    print(" " * 15 + "DeepSeek NER 数据生成器")
    print("=" * 70)

    if DEEPSEEK_API_KEY == "your_api_key_here":
        print("\n❌ 错误: 请先配置 DeepSeek API Key")
        print("请在脚本开头设置: DEEPSEEK_API_KEY = 'sk-xxx'")
        return

    all_samples = []
    error_count = 0
    fixed_count = 0

    # 分批生成
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - len(all_samples))

        print(f"\n📦 批次 {batch_idx + 1}/{num_batches} (目标: {current_batch_size} 个样本)")
        print("-" * 70)

        # 生成 prompt
        prompt = generate_prompt(current_batch_size)

        # 调用 API
        print("  🔄 调用 DeepSeek API...")
        try:
            response = call_deepseek_api(prompt)
            print("  ✅ API 调用成功")
        except Exception as e:
            print(f"  ❌ API 调用失败: {e}")
            error_count += 1
            continue

            # 解析响应
        print("  🔍 解析响应...")
        samples = parse_api_response(response)
        print(f"  📊 解析得到 {len(samples)} 个原始样本")

        # 清洗和验证
        print("  🧹 清洗和验证样本...")
        valid_samples = []

        for words, tags in samples:
            # 清洗
            words, tags = clean_sample(words, tags)

            # 验证
            is_valid, msg = validate_sample(words, tags)

            if is_valid:
                valid_samples.append((words, tags))
            else:
                print(f"    ⚠️  样本无效: {msg} - {''.join(words[:10])}...")
                error_count += 1

        print(f"  ✅ 有效样本: {len(valid_samples)} 个")

        all_samples.extend(valid_samples)

        print(f"  📈 累计有效样本: {len(all_samples)}/{num_samples}")

        # 如果已经足够了就停止
        if len(all_samples) >= num_samples:
            break

            # 去重
    print("\n🔄 去重...")
    unique_samples = []
    seen_sentences = set()

    for words, tags in all_samples:
        sentence = ''.join(words)
        if sentence not in seen_sentences:
            unique_samples.append((words, tags))
            seen_sentences.add(sentence)

    print(f"  去重前: {len(all_samples)} 个")
    print(f"  去重后: {len(unique_samples)} 个")

    all_samples = unique_samples[:num_samples]

    # 统计信息
    print("\n📊 数据统计:")
    print("-" * 70)

    tag_counter = Counter()
    entity_counter = Counter()

    for words, tags in all_samples:
        for tag in tags:
            tag_counter[tag] += 1
            if tag.startswith('B-'):
                entity_counter[tag[2:]] += 1

    print(f"  总样本数: {len(all_samples)}")
    print(f"  总字符数: {sum(tag_counter.values())}")
    print(f"  总实体数: {sum(entity_counter.values())}")

    print(f"\n  标签分布:")
    for tag, count in sorted(tag_counter.items()):
        percentage = count / sum(tag_counter.values()) * 100
        print(f"    {tag:10s}: {count:5d} ({percentage:5.2f}%)")

    print(f"\n  实体类型分布:")
    for entity_type, count in sorted(entity_counter.items()):
        percentage = count / sum(entity_counter.values()) * 100
        print(f"    {entity_type:5s}: {count:4d} ({percentage:5.2f}%)")

        # 显示样本示例
    print(f"\n📖 样本示例 (前5个):")
    print("-" * 70)

    for i, (words, tags) in enumerate(all_samples[:5], 1):
        sentence = ''.join(words)
        print(f"\n  [{i}] {sentence}")
        print(f"      ", end="")

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

        # 划分训练集和测试集
    import random
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]

    # 保存数据
    print(f"\n💾 保存数据...")
    print("-" * 70)

    os.makedirs(os.path.dirname(output_train) or '.', exist_ok=True)

    # 保存训练集
    with open(output_train, 'w', encoding='utf-8') as f:
        for words, tags in train_samples:
            for word, tag in zip(words, tags):
                f.write(f'{word} {tag}\n')
            f.write('\n')

    print(f"  ✅ 训练集: {len(train_samples)} 个样本 → {output_train}")

    # 保存测试集
    with open(output_test, 'w', encoding='utf-8') as f:
        for words, tags in test_samples:
            for word, tag in zip(words, tags):
                f.write(f'{word} {tag}\n')
            f.write('\n')

    print(f"  ✅ 测试集: {len(test_samples)} 个样本 → {output_test}")

    # 总结
    print("\n" + "=" * 70)
    print("✅ 数据生成完成！")
    print("=" * 70)
    print(f"\n  生成样本: {len(all_samples)}")
    print(f"  训练集: {len(train_samples)}")
    print(f"  测试集: {len(test_samples)}")
    print(f"  错误/跳过: {error_count}")

    print("\n下一步:")
    print(f"  1. 查看数据:")
    print(f"     head -50 {output_train}")
    print(f"\n  2. 合并到现有数据:")
    print(f"     cat data/train.txt {output_train} > data/train_merged.txt")
    print(f"\n  3. 训练模型:")
    print(f"     python train.py")
    print("=" * 70)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='使用 DeepSeek 生成 NER 数据')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='生成样本数量 (默认: 200)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='每批生成数量 (默认: 50)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例 (默认: 0.8)')
    parser.add_argument('--output_train', type=str, default='data/train_deepseek.txt',
                        help='训练集输出路径')
    parser.add_argument('--output_test', type=str, default='data/test_deepseek.txt',
                        help='测试集输出路径')

    args = parser.parse_args()

    generate_data_with_deepseek(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_train=args.output_train,
        output_test=args.output_test,
        train_ratio=args.train_ratio
    )


if __name__ == '__main__':
    main()