#!/usr/bin/env python3
# scripts/data_augmentation.py

import json
import random
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    question: str
    answer: str
    q_type: str
    difficulty: str
    explanation: str = ""
    source: str = ""
    supporting_facts: List[str] = None

    def __post_init__(self):
        if self.supporting_facts is None:
            self.supporting_facts = []


class DataAugmentor:
    def __init__(self):
        self.question_templates = {
            "人物识别": [
                "{}的主要人物是谁？",
                "{}是谁策划的？",
                "{}的核心人物是哪位？",
                "在{}中发挥关键作用的人物是谁？",
                "{}这件事与哪位人物关系最密切？"
            ],
            "事件描述": [
                "{}的详细经过是怎样的？",
                "请描述{}的具体过程",
                "{}发生了什么？",
                "{}事件的具体情况如何？",
                "能详细讲讲{}吗？"
            ],
            "时间顺序": [
                "{}发生在什么时候？",
                "{}的具体时间是？",
                "{}发生在哪一年？",
                "{}的时间背景是什么？",
                "{}发生在哪个历史时期？"
            ],
            "人物关系": [
                "{}和{}是什么关系？",
                "{}与{}之间有怎样的关联？",
                "{}和{}在三国中是什么关系？",
                "请说明{}和{}的关系",
                "{}与{}在血缘或政治上是何关系？"
            ],
            "地理知识": [
                "{}发生在哪里？",
                "{}的具体地点是？",
                "{}发生在什么地点？",
                "{}的地理位置在哪里？",
                "{}的发生地是何处？"
            ],
            "因果关系": [
                "{}的主要原因是什么？",
                "为什么会发生{}？",
                "{}的起因是什么？",
                "导致{}的因素有哪些？",
                "{}发生的背景是什么？"
            ],
            "对比分析": [
                "{}和{}有什么异同？",
                "比较{}和{}的异同点",
                "{}与{}相比有什么特点？",
                "分析{}和{}的相似与不同",
                "对比{}和{}的主要区别"
            ]
        }

    def load_base_data(self, data_path: str = "data/raw/sanguo_base_data.json") -> Dict:
        """加载基础数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_qa_pairs(self, base_data: Dict) -> List[QAPair]:
        """生成多样化的问题-答案对"""
        qa_pairs = []

        # 生成各类问题
        qa_pairs.extend(self._generate_character_qa(base_data["characters"]))
        qa_pairs.extend(self._generate_event_qa(base_data["events"]))
        qa_pairs.extend(self._generate_relation_qa(base_data["characters"]))
        qa_pairs.extend(self._generate_timeline_qa(base_data["events"]))
        qa_pairs.extend(self._generate_location_qa(base_data))
        qa_pairs.extend(self._generate_battle_qa(base_data["battles"]))
        qa_pairs.extend(self._generate_comparison_qa(base_data))

        # 打乱顺序
        random.shuffle(qa_pairs)

        logger.info(f"生成了 {len(qa_pairs)} 个问答对")
        return qa_pairs

    def _generate_character_qa(self, characters: List[Dict]) -> List[QAPair]:
        """生成人物相关问答"""
        qa_pairs = []

        for char in characters:
            # 基本信息问答
            qa_pairs.append(QAPair(
                question=f"{char['name']}的字是什么？",
                answer=char['style_name'],
                q_type="人物识别",
                difficulty="简单",
                explanation=f"{char['name']}的字是{char['style_name']}",
                supporting_facts=[char['name'], char['style_name']]
            ))

            qa_pairs.append(QAPair(
                question=f"{char['name']}属于哪个势力？",
                answer=char['faction'],
                q_type="人物识别",
                difficulty="简单",
                explanation=f"{char['name']}属于{char['faction']}势力",
                supporting_facts=[char['name'], char['faction']]
            ))

            # 事件关联问答
            for event in char['key_events'][:3]:  # 取前3个关键事件
                qa_pairs.append(QAPair(
                    question=f"{event}的主要人物是谁？",
                    answer=char['name'],
                    q_type="人物识别",
                    difficulty="中等",
                    explanation=f"{event}的主要人物是{char['name']}",
                    supporting_facts=[char['name'], event]
                ))

            # 关系问答
            for relation in char['relations'][:2]:
                qa_pairs.append(QAPair(
                    question=f"{char['name']}和{relation}是什么关系？",
                    answer=self._generate_relation_answer(char['name'], relation),
                    q_type="人物关系",
                    difficulty="中等",
                    explanation=f"{char['name']}和{relation}在三国中有密切关系",
                    supporting_facts=[char['name'], relation]
                ))

        return qa_pairs

    def _generate_event_qa(self, events: List[Dict]) -> List[QAPair]:
        """生成事件相关问答"""
        qa_pairs = []

        for event in events:
            # 时间问答
            qa_pairs.append(QAPair(
                question=f"{event['name']}发生在什么时候？",
                answer=event['time'],
                q_type="时间顺序",
                difficulty="简单",
                explanation=f"{event['name']}发生在{event['time']}",
                supporting_facts=[event['name'], event['time']]
            ))

            # 地点问答
            qa_pairs.append(QAPair(
                question=f"{event['name']}发生在哪里？",
                answer=event['location'],
                q_type="地理知识",
                difficulty="简单",
                explanation=f"{event['name']}发生在{event['location']}",
                supporting_facts=[event['name'], event['location']]
            ))

            # 参与者问答
            participants_str = "、".join(event['participants'])
            qa_pairs.append(QAPair(
                question=f"{event['name']}的主要参与者有哪些？",
                answer=participants_str,
                q_type="人物识别",
                difficulty="中等",
                explanation=f"{event['name']}的主要参与者包括{participants_str}",
                supporting_facts=[event['name']] + event['participants']
            ))

            # 描述性问答
            qa_pairs.append(QAPair(
                question=f"请描述{event['name']}的具体过程",
                answer=event['description'],
                q_type="事件描述",
                difficulty="困难",
                explanation=event['description'],
                supporting_facts=[event['name'], event['description']]
            ))

        return qa_pairs

    def _generate_relation_qa(self, characters: List[Dict]) -> List[QAPair]:
        """生成人物关系问答"""
        qa_pairs = []

        # 预定义一些特定关系
        known_relations = {
            ("刘备", "关羽"): "结义兄弟关系，刘备是大哥，关羽是二弟",
            ("刘备", "张飞"): "结义兄弟关系，刘备是大哥，张飞是三弟",
            ("诸葛亮", "诸葛瑾"): "兄弟关系，诸葛亮是弟弟，诸葛瑾是哥哥",
            ("曹操", "夏侯惇"): "堂兄弟关系，曹操的父亲曹嵩是夏侯惇的叔父",
            ("周瑜", "小乔"): "夫妻关系，周瑜是丈夫，小乔是妻子",
            ("孙权", "孙策"): "兄弟关系，孙策是哥哥，孙权是弟弟",
            ("司马懿", "司马师"): "父子关系，司马懿是父亲，司马师是儿子",
        }

        for (person1, person2), relation_desc in known_relations.items():
            qa_pairs.append(QAPair(
                question=f"{person1}和{person2}是什么关系？",
                answer=relation_desc,
                q_type="人物关系",
                difficulty="中等",
                explanation=relation_desc,
                supporting_facts=[person1, person2]
            ))

        return qa_pairs

    def _generate_timeline_qa(self, events: List[Dict]) -> List[QAPair]:
        """生成时间顺序问答"""
        qa_pairs = []

        # 按时间排序事件
        sorted_events = sorted(events, key=lambda x: self._extract_year(x['time']))

        # 生成时间顺序问题
        for i in range(len(sorted_events) - 1):
            event1 = sorted_events[i]
            event2 = sorted_events[i + 1]

            qa_pairs.append(QAPair(
                question=f"{event1['name']}和{event2['name']}哪个发生得更早？",
                answer=event1['name'],
                q_type="时间顺序",
                difficulty="中等",
                explanation=f"{event1['name']}({event1['time']})比{event2['name']}({event2['time']})发生得更早",
                supporting_facts=[event1['name'], event2['name'], event1['time'], event2['time']]
            ))

        return qa_pairs

    def _generate_location_qa(self, base_data: Dict) -> List[QAPair]:
        """生成地理知识问答"""
        qa_pairs = []

        for location in base_data["locations"]:
            qa_pairs.append(QAPair(
                question=f"{location['name']}在三国时期属于哪个势力？",
                answer=location['faction'],
                q_type="地理知识",
                difficulty="简单",
                explanation=f"{location['name']}在三国时期属于{location['faction']}",
                supporting_facts=[location['name'], location['faction']]
            ))

        return qa_pairs

    def _generate_battle_qa(self, battles: List[Dict]) -> List[QAPair]:
        """生成战役相关问答"""
        qa_pairs = []

        for battle in battles:
            qa_pairs.append(QAPair(
                question=f"{battle['name']}的结果如何？",
                answer=battle['结果'],
                q_type="事件描述",
                difficulty="中等",
                explanation=f"{battle['name']}的结果是{battle['结果']}",
                supporting_facts=[battle['name'], battle['结果']]
            ))

            qa_pairs.append(QAPair(
                question=f"{battle['name']}的主要参战方是谁？",
                answer=" vs ".join(battle['双方']),
                q_type="人物识别",
                difficulty="中等",
                explanation=f"{battle['name']}的主要参战方是{'和'.join(battle['双方'])}",
                supporting_facts=[battle['name']] + battle['双方']
            ))

        return qa_pairs

    def _generate_comparison_qa(self, base_data: Dict) -> List[QAPair]:
        """生成对比分析问答"""
        qa_pairs = []

        # 人物对比
        comparisons = [
            ("诸葛亮", "周瑜", "都是三国时期著名的军事家，诸葛亮擅长谋略和治国，周瑜擅长水战和音乐"),
            ("曹操", "刘备", "曹操是奸雄，善于用人但多疑；刘备是仁君，以仁义著称但能力稍逊"),
            ("关羽", "张飞", "都是蜀汉五虎上将，关羽重义气擅长马战，张飞勇猛擅长步战")
        ]

        for person1, person2, comparison in comparisons:
            qa_pairs.append(QAPair(
                question=f"{person1}和{person2}有什么不同？",
                answer=comparison,
                q_type="对比分析",
                difficulty="困难",
                explanation=comparison,
                supporting_facts=[person1, person2]
            ))

        return qa_pairs

    def _extract_year(self, time_str: str) -> int:
        """从时间字符串中提取年份"""
        match = re.search(r'(\d+)年', time_str)
        if match:
            return int(match.group(1))
        return 0

    def _generate_relation_answer(self, person1: str, person2: str) -> str:
        """生成人物关系答案"""
        # 这里可以根据具体人物生成更精确的关系描述
        return f"{person1}和{person2}在三国时期有密切的关联，具体关系需要根据历史背景分析"

    def export_qa_dataset(self, qa_pairs: List[QAPair], output_path: str = "data/processed/sanguo_qa_dataset.json"):
        """导出问答数据集"""
        # 统计信息
        type_count = defaultdict(int)
        difficulty_count = defaultdict(int)

        for qa in qa_pairs:
            type_count[qa.q_type] += 1
            difficulty_count[qa.difficulty] += 1

        dataset = {
            "metadata": {
                "total_samples": len(qa_pairs),
                "difficulty_distribution": dict(difficulty_count),
                "type_distribution": dict(type_count),
                "creation_time": self._get_current_time()
            },
            "samples": [qa.__dict__ for qa in qa_pairs]
        }

        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        logger.info(f"问答数据集已导出到: {output_path}")
        logger.info(f"数据集统计: {dataset['metadata']}")

    def _get_current_time(self):
        import time
        return time.strftime("%Y-%m-%d %H:%M:%S")


def main():
    """主函数"""
    augmentor = DataAugmentor()

    # 加载基础数据
    base_data = augmentor.load_base_data()

    # 生成问答对
    qa_pairs = augmentor.generate_qa_pairs(base_data)

    # 导出数据集
    augmentor.export_qa_dataset(qa_pairs)


if __name__ == "__main__":
    main()