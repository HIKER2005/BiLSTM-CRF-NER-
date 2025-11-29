"""
词表管理类
负责词和标签的索引映射
"""


class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.tag2idx = {'<PAD>': 0}
        self.idx2tag = {0: '<PAD>'}

    def add_word(self, word):
        """添加词到词表"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def add_tag(self, tag):
        """添加标签到标签表"""
        if tag not in self.tag2idx:
            idx = len(self.tag2idx)
            self.tag2idx[tag] = idx
            self.idx2tag[idx] = tag

    def build_vocab(self, sentences, tags):
        """从训练数据构建词表"""
        for sentence in sentences:
            for word in sentence:
                self.add_word(word)

        for tag_seq in tags:
            for tag in tag_seq:
                self.add_tag(tag)

    def get_word_idx(self, word):
        """获取词的索引，未登录词返回UNK"""
        return self.word2idx.get(word, self.word2idx['<UNK>'])

    def get_tag_idx(self, tag):
        """获取标签的索引"""
        return self.tag2idx.get(tag, 0)

    def get_word(self, idx):
        """根据索引获取词"""
        return self.idx2word.get(idx, '<UNK>')

    def get_tag(self, idx):
        """根据索引获取标签"""
        return self.idx2tag.get(idx, '<PAD>')

    @property
    def vocab_size(self):
        return len(self.word2idx)

    @property
    def tag_size(self):
        return len(self.tag2idx)

    def __repr__(self):
        return f"Vocabulary(vocab_size={self.vocab_size}, tag_size={self.tag_size})"