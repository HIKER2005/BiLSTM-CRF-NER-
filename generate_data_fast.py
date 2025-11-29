"""
DeepSeek NER 快速数据生成器 - 严格约束版本
带完整验证、自动修复和并行加速
"""

import os
import requests
import json
import time
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import random


class StrictDeepSeekDataGenerator:
    def __init__(self, api_key, base_url="https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 统计信息
        self.stats = {
            'total_generated': 0,
            'format_fixed': 0,
            'boundary_fixed': 0,
            'invalid_samples': 0,
            'api_errors': 0
        }

    def call_api(self, messages, temperature=0.7, max_retries=3):
        """调用API with重试"""
        url = f"{self.base_url}/chat/completions"

        for attempt in range(max_retries):
            try:
                payload = {
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 2000
                }

                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']

            except Exception as e:
                if attempt == max_retries - 1:
                    self.stats['api_errors'] += 1
                    print(f"\n  ❌ API调用失败: {e}")
                    return None
                time.sleep(1)

        return None

    def get_strict_system_prompt(self):
        """获取严格的系统提示"""
        return """你是一个严格的NER数据标注专家。

    **硬性规则**：
    1. 只能使用这7种标签：B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, O
    2. 绝对禁止：PER-B, M-ORG, E-ORG, S-PER, PER, LOC, ORG
    3. 每个实体第一个字必须是B-，后续字必须是I-
    4. 输出格式：字<空格>标签，每行一个
    
    违反任何规则都是错误。"""

    def generate_sentences(self, num_sentences=50):
        """生成句子"""
        prompt = f"""生成{num_sentences}个中文句子用于NER标注。

        要求：
        1. 句子长度：8-30字
        2. 必须包含至少1个命名实体
        3. 实体类型要均衡：人名(PER)、地名(LOC)、机构名(ORG)
        4. 句子自然、真实、语法正确
        
        实体示例：
        - 人名(PER): 马云、姚明、鲁迅、周杰伦、钟南山
        - 地名(LOC): 北京、上海、长江、黄山、中国
        - 机构(ORG): 阿里巴巴、清华大学、中国银行、联合国
        
        只输出句子，每行一个，不要编号。
        
        生成{num_sentences}个句子："""

        messages = [
            {"role": "system", "content": self.get_strict_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        response = self.call_api(messages, temperature=0.9)

        if response:
            sentences = []
            for line in response.strip().split('\n'):
                # 清理编号
                line = re.sub(r'^\d+[\.\、\s]+', '', line.strip())
                # 清理引号
                line = line.strip('"\'""''')

                if line and 5 <= len(line) <= 100:
                    sentences.append(line)

            return sentences
        return []

    def annotate_sentence(self, sentence):
        """标注单个句子 - 带严格约束"""
        prompt = f"""对句子进行BIO标注。

        **严格要求**：
        1. 只能使用这7种标签：B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, O
        2. 实体第一个字用B-，后续字用I-
        3. 格式：字<空格>标签，每行一个
        
        **禁止的标签**：
        ❌ PER-B, LOC-I (倒序)
        ❌ M-ORG, E-ORG (BMES)
        ❌ PER, LOC, ORG (无前缀)
        
        句子：{sentence}
        
        严格按照BIO格式标注："""

        messages = [
            {"role": "system", "content": self.get_strict_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        response = self.call_api(messages, temperature=0.3)

        if response:
            try:
                # 解析标注
                words, tags = self.parse_annotation(response)

                # 验证长度
                if len(words) != len(sentence.replace(' ', '')):
                    return None

                # 清洗和验证
                words, tags = self.clean_sample(words, tags)

                # 最终验证
                is_valid, msg = self.validate_sample(words, tags)

                if is_valid:
                    self.stats['total_generated'] += 1
                    return words, tags
                else:
                    self.stats['invalid_samples'] += 1

            except Exception as e:
                self.stats['invalid_samples'] += 1

        return None

    def parse_annotation(self, response):
        """解析标注响应"""
        words = []
        tags = []

        for line in response.strip().split('\n'):
            line = line.strip()

            # 跳过空行和markdown
            if not line or line.startswith('```') or line.startswith('#'):
                continue

            # 解析 "字 标签"
            parts = line.split()
            if len(parts) == 2:
                word, tag = parts
                if len(word) == 1:  # 只要单个字符
                    words.append(word)
                    tags.append(tag)

        return words, tags

    def fix_tag_format(self, tag):
        """修复标签格式"""
        tag = tag.strip().upper()

        # 已经正确
        if tag in ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'O']:
            return tag

        # 倒序: PER-B → B-PER
        if re.match(r'^(PER|LOC|ORG)-([BI])$', tag):
            entity_type, prefix = tag.split('-')
            self.stats['format_fixed'] += 1
            return f'{prefix}-{entity_type}'

        # 无前缀: PER → B-PER
        if tag in ['PER', 'LOC', 'ORG']:
            self.stats['format_fixed'] += 1
            return f'B-{tag}'

        # BMES格式: M-ORG → I-ORG
        if re.match(r'^[BMES]-(PER|LOC|ORG)$', tag):
            prefix, entity_type = tag.split('-')
            self.stats['format_fixed'] += 1
            if prefix == 'B':
                return f'B-{entity_type}'
            elif prefix in ['M', 'E', 'I']:
                return f'I-{entity_type}'
            elif prefix == 'S':
                return f'B-{entity_type}'

        # 英文全称
        tag_mapping = {
            'PERSON': 'PER', 'PEOPLE': 'PER',
            'LOCATION': 'LOC', 'PLACE': 'LOC',
            'ORGANIZATION': 'ORG', 'COMPANY': 'ORG'
        }

        for old, new in tag_mapping.items():
            if old in tag:
                self.stats['format_fixed'] += 1
                if 'B' in tag or tag == old:
                    return f'B-{new}'
                else:
                    return f'I-{new}'

        # 无法识别
        return 'O'

    def fix_entity_boundaries(self, words, tags):
        """修复实体边界"""
        fixed_tags = []

        for i, tag in enumerate(tags):
            # I- 标签检查
            if tag.startswith('I-'):
                entity_type = tag[2:]
                # 前面必须是同类型的 B- 或 I-
                if i == 0 or not fixed_tags[-1].endswith(f'-{entity_type}'):
                    fixed_tags.append(f'B-{entity_type}')
                    self.stats['boundary_fixed'] += 1
                else:
                    fixed_tags.append(tag)
            else:
                fixed_tags.append(tag)

        return fixed_tags

    def clean_sample(self, words, tags):
        """清洗样本"""
        # 1. 修复标签格式
        fixed_tags = [self.fix_tag_format(tag) for tag in tags]

        # 2. 修复实体边界
        fixed_tags = self.fix_entity_boundaries(words, fixed_tags)

        return words, fixed_tags

    def validate_sample(self, words, tags):
        """验证样本有效性"""
        # 长度检查
        if len(words) != len(tags):
            return False, "长度不匹配"

        if len(words) < 5:
            return False, "句子太短"

        if len(words) > 100:
            return False, "句子太长"

        # 标签合法性
        valid_tags = {'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'O'}
        for tag in tags:
            if tag not in valid_tags:
                return False, f"非法标签: {tag}"

        # 至少有一个实体
        has_entity = any(tag.startswith('B-') for tag in tags)
        if not has_entity:
            return False, "无实体"

        # 边界一致性
        for i, tag in enumerate(tags):
            if tag.startswith('I-'):
                entity_type = tag[2:]
                if i == 0 or not tags[i-1].endswith(f'-{entity_type}'):
                    return False, "边界错误"

        return True, "OK"

    def annotate_batch_parallel(self, sentences, max_workers=5):
        """并行标注"""
        results = [None] * len(sentences)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.annotate_sentence, sent): idx
                for idx, sent in enumerate(sentences)
            }

            with tqdm(total=len(sentences), desc="  标注进度", ncols=80) as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        pass
                    pbar.update(1)

        return results

    def format_sample(self, words, tags):
        """格式化样本为字符串"""
        lines = []
        for word, tag in zip(words, tags):
            lines.append(f'{word} {tag}')
        return '\n'.join(lines)

    def generate_dataset(self, num_sentences=200, output_file='data/generated_data.txt',
                         batch_size=50, max_workers=5):
        """生成数据集"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        all_samples = []
        seen_sentences = set()
        num_batches = (num_sentences + batch_size - 1) // batch_size

        print(f"\n{'='*70}")
        print(f"{'DeepSeek NER 严格数据生成器':^70}")
        print(f"{'='*70}")
        print(f"\n配置:")
        print(f"  目标样本数: {num_sentences}")
        print(f"  批次大小: {batch_size}")
        print(f"  并行度: {max_workers}")
        print(f"  预计时间: {num_sentences * 3 / max_workers / 60:.1f} 分钟")
        print(f"\n{'='*70}\n")

        start_time = time.time()

        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_sentences - len(all_samples))

            print(f"📦 批次 {batch_idx + 1}/{num_batches}")
            print(f"{'-'*70}")

            # 生成句子
            print(f"  🔄 生成 {current_batch_size} 个句子...")
            sentences = self.generate_sentences(current_batch_size)

            if not sentences:
                print("  ❌ 句子生成失败")
                continue

            print(f"  ✅ 成功生成 {len(sentences)} 个句子")

            # 并行标注
            print(f"  🏷️  开始并行标注 (并行度={max_workers})...")
            results = self.annotate_batch_parallel(sentences, max_workers)

            # 收集有效样本
            valid_count = 0
            for i, result in enumerate(results):
                if result:
                    words, tags = result
                    sentence = ''.join(words)

                    # 去重
                    if sentence not in seen_sentences:
                        all_samples.append((words, tags))
                        seen_sentences.add(sentence)
                        valid_count += 1

            print(f"  ✅ 有效样本: {valid_count}/{len(results)}")
            print(f"  📈 累计样本: {len(all_samples)}/{num_sentences}")
            print()

            # 达到目标数量
            if len(all_samples) >= num_sentences:
                break

        elapsed = time.time() - start_time

        # 截取到目标数量
        all_samples = all_samples[:num_sentences]

        # 保存数据
        print(f"{'='*70}")
        print(f"💾 保存数据...")
        print(f"{'-'*70}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for words, tags in all_samples:
                f.write(self.format_sample(words, tags))
                f.write('\n\n')

        print(f"  ✅ 已保存 {len(all_samples)} 个样本到: {output_file}")

        # 统计信息
        self.print_statistics(all_samples, elapsed)

        return len(all_samples)

    def print_statistics(self, samples, elapsed):
        """打印统计信息"""
        print(f"\n{'='*70}")
        print(f"📊 数据统计")
        print(f"{'='*70}")

        # 基础统计
        print(f"\n  ⏱️  耗时: {elapsed/60:.1f} 分钟")
        print(f"  ⚡ 速度: {len(samples)/(elapsed/60):.1f} 句/分钟")
        print(f"  ✅ 成功样本: {len(samples)}")

        # 修复统计
        print(f"\n  🔧 修复统计:")
        print(f"    格式修复: {self.stats['format_fixed']} 次")
        print(f"    边界修复: {self.stats['boundary_fixed']} 次")
        print(f"    无效样本: {self.stats['invalid_samples']} 个")
        print(f"    API错误: {self.stats['api_errors']} 次")

        # 标签分布
        tag_counter = Counter()
        entity_counter = Counter()

        for words, tags in samples:
            for tag in tags:
                tag_counter[tag] += 1
                if tag.startswith('B-'):
                    entity_counter[tag[2:]] += 1

        print(f"\n  🏷️  标签分布:")
        total_tags = sum(tag_counter.values())
        for tag in ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'O']:
            count = tag_counter[tag]
            percentage = count / total_tags * 100
            print(f"    {tag:10s}: {count:5d} ({percentage:5.2f}%)")

        print(f"\n  📌 实体分布:")
        total_entities = sum(entity_counter.values())
        for entity_type in ['PER', 'LOC', 'ORG']:
            count = entity_counter[entity_type]
            percentage = count / total_entities * 100 if total_entities > 0 else 0
            print(f"    {entity_type:5s}: {count:4d} ({percentage:5.2f}%)")

        # 样本示例
        print(f"\n  📖 样本示例 (前3个):")
        print(f"  {'-'*66}")

        for i, (words, tags) in enumerate(samples[:3], 1):
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


def split_train_test(input_file, train_file='data/train.txt',
                     test_file='data/test.txt', test_ratio=0.2):
    """分割训练测试集"""
    sentences = []
    current = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                current.append(line)
            elif current:
                sentences.append(current)
                current = []

    if current:
        sentences.append(current)

    random.shuffle(sentences)

    split_idx = int(len(sentences) * (1 - test_ratio))
    train_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]

    with open(train_file, 'w', encoding='utf-8') as f:
        for sent in train_sentences:
            f.write('\n'.join(sent) + '\n\n')

    with open(test_file, 'w', encoding='utf-8') as f:
        for sent in test_sentences:
            f.write('\n'.join(sent) + '\n\n')

    print(f"\n{'='*70}")
    print(f"📂 数据集分割")
    print(f"{'-'*70}")
    print(f"  训练集: {len(train_sentences)} 个样本 → {train_file}")
    print(f"  测试集: {len(test_sentences)} 个样本 → {test_file}")
    print(f"{'='*70}")


def main():
    """主函数 - 交互式版本"""
    print("=" * 70)
    print(" " * 15 + "DeepSeek NER 严格数据生成器")
    print("=" * 70)

    # 获取API Key
    api_key = os.getenv('DEEPSEEK_API_KEY')

    if not api_key:
        print("\n🔑 请输入 DeepSeek API Key:")
        print("   (或设置环境变量: export DEEPSEEK_API_KEY=sk-xxx)")
        api_key = input("\nAPI Key: ").strip()

    if not api_key:
        print("\n❌ 错误: 未提供API密钥")
        return

    print(f"\n✅ API Key: {api_key[:10]}...{api_key[-4:]}")

    # 询问生成数量
    print("\n" + "=" * 70)
    print("⚙️  配置参数")
    print("=" * 70)

    while True:
        num_input = input("\n📝 生成多少个句子? (推荐200-500): ").strip()

        if not num_input:
            num_sentences = 200
            print(f"   使用默认值: {num_sentences}")
            break

        try:
            num_sentences = int(num_input)
            if num_sentences < 10:
                print("   ⚠️  至少需要10个句子")
                continue
            elif num_sentences > 2000:
                confirm = input(f"   ⚠️  {num_sentences}个句子较多，确认? (y/n): ").strip().lower()
                if confirm == 'y':
                    break
                else:
                    continue
            else:
                break
        except ValueError:
            print("   ❌ 请输入有效的数字")

    # 询问并行度
    while True:
        workers_input = input("\n⚡ 并行度? (推荐5-10，越大越快但API压力越大): ").strip()

        if not workers_input:
            max_workers = 5
            print(f"   使用默认值: {max_workers}")
            break

        try:
            max_workers = int(workers_input)
            if max_workers < 1:
                print("   ⚠️  至少需要1")
                continue
            elif max_workers > 20:
                print("   ⚠️  并行度过高可能导致API限流")
                confirm = input(f"   确认使用 {max_workers}? (y/n): ").strip().lower()
                if confirm == 'y':
                    break
                else:
                    continue
            else:
                break
        except ValueError:
            print("   ❌ 请输入有效的数字")

    # 询问批次大小
    batch_size = 50
    advanced = input("\n🔧 需要高级配置吗? (y/n, 默认n): ").strip().lower()

    if advanced == 'y':
        while True:
            batch_input = input(f"\n📦 批次大小? (默认50): ").strip()

            if not batch_input:
                batch_size = 50
                break

            try:
                batch_size = int(batch_input)
                if 10 <= batch_size <= 100:
                    break
                else:
                    print("   ⚠️  建议范围: 10-100")
            except ValueError:
                print("   ❌ 请输入有效的数字")

    # 确认配置
    print("\n" + "=" * 70)
    print("📋 配置摘要")
    print("=" * 70)
    print(f"  生成句子数: {num_sentences}")
    print(f"  并行度: {max_workers}")
    print(f"  批次大小: {batch_size}")
    print(f"  预计时间: {num_sentences * 3 / max_workers / 60:.1f} 分钟")
    print(f"  预计API调用: ~{num_sentences * 2} 次")
    print("=" * 70)

    confirm = input("\n✅ 确认开始生成? (y/n): ").strip().lower()

    if confirm != 'y':
        print("\n❌ 已取消")
        return

    # 创建生成器
    generator = StrictDeepSeekDataGenerator(api_key)

    # 生成数据
    print("\n" + "=" * 70)
    print("🚀 开始生成...")
    print("=" * 70)

    try:
        num_generated = generator.generate_dataset(
            num_sentences=num_sentences,
            output_file='data/generated_data.txt',
            batch_size=batch_size,
            max_workers=max_workers
        )

        # 分割数据集
        if num_generated > 0:
            split = input("\n📂 是否自动分割训练/测试集? (y/n, 默认y): ").strip().lower()

            if split != 'n':
                test_ratio_input = input("   测试集比例? (0.1-0.3, 默认0.2): ").strip()

                try:
                    test_ratio = float(test_ratio_input) if test_ratio_input else 0.2
                    test_ratio = max(0.1, min(0.3, test_ratio))
                except:
                    test_ratio = 0.2

                split_train_test('data/generated_data.txt', test_ratio=test_ratio)

        print(f"\n{'=' * 70}")
        print(f"✅ 全部完成！")
        print(f"{'=' * 70}")
        print(f"\n生成的文件:")
        print(f"  📄 原始数据: data/generated_data.txt")

        if split != 'n':
            print(f"  📄 训练集: data/train.txt")
            print(f"  📄 测试集: data/test.txt")

        print(f"\n下一步:")
        print(f"  1. 查看数据:")
        print(f"     head -50 data/train.txt")
        print(f"\n  2. 检查数据质量:")
        print(f"     python analyze_data.py")
        print(f"\n  3. 训练模型:")
        print(f"     python train.py")
        print(f"{'=' * 70}\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，已停止生成")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")


if __name__ == '__main__':
    main()