"""
评估指标计算
"""

from collections import defaultdict


def extract_entities(tags, words=None):
    """
    从标签序列中提取实体

    Args:
        tags: List[str] - 标签序列，如 ['B-PER', 'I-PER', 'O', 'B-LOC']
        words: List[str] - 词序列（可选）

    Returns:
        entities: List[tuple] - [(start, end, entity_type, entity_text)]
    """
    entities = []
    start = None
    entity_type = None

    for i, tag in enumerate(tags):
        if tag.startswith('B-'):
            # 如果前面有未结束的实体，先保存
            if start is not None:
                entity_text = ' '.join(words[start:i]) if words else ''
                entities.append((start, i, entity_type, entity_text))

            # 开始新实体
            start = i
            entity_type = tag[2:]

        elif tag.startswith('I-'):
            # 继续当前实体
            if start is None:
                # 没有B标签就出现I标签，视为新实体开始
                start = i
                entity_type = tag[2:]
        else:
            # O标签，结束当前实体
            if start is not None:
                entity_text = ' '.join(words[start:i]) if words else ''
                entities.append((start, i, entity_type, entity_text))
                start = None
                entity_type = None

    # 处理最后一个实体
    if start is not None:
        entity_text = ' '.join(words[start:]) if words else ''
        entities.append((start, len(tags), entity_type, entity_text))

    return entities


def compute_metrics(true_tags_list, pred_tags_list, words_list=None):
    """
    计算精确率、召回率、F1分数

    Args:
        true_tags_list: List[List[str]] - 真实标签序列列表
        pred_tags_list: List[List[str]] - 预测标签序列列表
        words_list: List[List[str]] - 词序列列表（可选）

    Returns:
        metrics: dict - 包含precision, recall, f1的字典
    """
    true_entities = set()
    pred_entities = set()

    for i, (true_tags, pred_tags) in enumerate(zip(true_tags_list, pred_tags_list)):
        words = words_list[i] if words_list else None

        # 提取真实实体
        for start, end, entity_type, text in extract_entities(true_tags, words):
            true_entities.add((i, start, end, entity_type))

        # 提取预测实体
        for start, end, entity_type, text in extract_entities(pred_tags, words):
            pred_entities.add((i, start, end, entity_type))

    # 计算指标
    tp = len(true_entities & pred_entities)  # 真阳性
    fp = len(pred_entities - true_entities)  # 假阳性
    fn = len(true_entities - pred_entities)  # 假阴性

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def compute_metrics_by_type(true_tags_list, pred_tags_list, words_list=None):
    """计算每个实体类型的指标"""
    entity_types = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for i, (true_tags, pred_tags) in enumerate(zip(true_tags_list, pred_tags_list)):
        words = words_list[i] if words_list else None

        # 提取真实实体
        true_entities_with_type = {}
        for start, end, entity_type, text in extract_entities(true_tags, words):
            key = (i, start, end, entity_type)
            true_entities_with_type[key] = entity_type

        # 提取预测实体
        pred_entities_with_type = {}
        for start, end, entity_type, text in extract_entities(pred_tags, words):
            key = (i, start, end, entity_type)
            pred_entities_with_type[key] = entity_type

        # 计算每个类型的tp, fp, fn
        all_entities = set(true_entities_with_type.keys()) | set(pred_entities_with_type.keys())

        for key in all_entities:
            if key in true_entities_with_type and key in pred_entities_with_type:
                # TP
                entity_type = true_entities_with_type[key]
                entity_types[entity_type]['tp'] += 1
            elif key in pred_entities_with_type:
                # FP
                entity_type = pred_entities_with_type[key]
                entity_types[entity_type]['fp'] += 1
            else:
                # FN
                entity_type = true_entities_with_type[key]
                entity_types[entity_type]['fn'] += 1

    # 计算每个类型的指标
    results = {}
    for entity_type, counts in entity_types.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn
        }

    return results