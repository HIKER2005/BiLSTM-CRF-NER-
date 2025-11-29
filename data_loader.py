"""
数据加载和处理
支持CoNLL格式数据
"""

import torch
from torch.utils.data import Dataset, DataLoader


class NERDataset(Dataset):
    def __init__(self, sentences, tags, vocab):
        """
        Args:
            sentences: List[List[str]] - 句子列表
            tags: List[List[str]] - 标签列表
            vocab: Vocabulary - 词表对象
        """
        self.sentences = sentences
        self.tags = tags
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """返回一个样本的词索引和标签索引"""
        sentence = self.sentences[idx]
        tags = self.tags[idx]

        # 转换为索引
        word_indices = [self.vocab.get_word_idx(word) for word in sentence]
        tag_indices = [self.vocab.get_tag_idx(tag) for tag in tags]

        return {
            'words': torch.tensor(word_indices, dtype=torch.long),
            'tags': torch.tensor(tag_indices, dtype=torch.long),
            'length': len(sentence)
        }


def collate_fn(batch):
    """
    自定义collate函数，处理变长序列
    填充到batch中最长的序列长度
    """
    # 获取batch中最大长度
    max_len = max([item['length'] for item in batch])

    # 填充
    words_padded = []
    tags_padded = []
    lengths = []

    for item in batch:
        length = item['length']
        lengths.append(length)

        # 填充words
        words = item['words']
        pad_len = max_len - length
        words_padded.append(torch.cat([words, torch.zeros(pad_len, dtype=torch.long)]))

        # 填充tags
        tags = item['tags']
        tags_padded.append(torch.cat([tags, torch.zeros(pad_len, dtype=torch.long)]))

    return {
        'words': torch.stack(words_padded),
        'tags': torch.stack(tags_padded),
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }


def load_data(file_path):
    """
    加载CoNLL格式的数据

    格式：每行一个词和标签，用空格分隔，句子之间用空行分隔
    """
    sentences = []
    tags = []

    current_sentence = []
    current_tags = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == '':  # 空行表示句子结束
                if current_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
            else:
                parts = line.split()
                if len(parts) == 2:
                    word, tag = parts
                    current_sentence.append(word)
                    current_tags.append(tag)

        # 处理最后一个句子
        if current_sentence:
            sentences.append(current_sentence)
            tags.append(current_tags)

    print(f"加载了 {len(sentences)} 个句子")
    return sentences, tags


def create_dataloader(sentences, tags, vocab, batch_size=32, shuffle=True):
    """创建DataLoader"""
    dataset = NERDataset(sentences, tags, vocab)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return dataloader