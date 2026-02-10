"""
OCR文字识别处理器模块
用于处理OCR检测结果，提供文字过滤、匹配、坐标转换等功能
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
import Levenshtein


class OCRProcessor:
    """OCR文字识别处理器"""
    
    def __init__(self, confidence_threshold: float = 0.6, similarity_threshold: float = 0.8):
        """
        初始化OCR处理器
        
        Args:
            confidence_threshold: 置信度阈值，低于此值的检测结果将被过滤
            similarity_threshold: 相似度阈值，用于文字匹配
        """
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
        
        # 常用游戏文字关键词（可根据具体游戏扩展）
        self.game_keywords = {
            'start': ['开始', '启动', '进入', 'START', 'PLAY'],
            'confirm': ['确认', '确定', 'OK', 'CONFIRM'],
            'cancel': ['取消', '返回', 'CANCEL', 'BACK'],
            'level': ['等级', '关卡', 'LEVEL', 'STAGE'],
            'score': ['分数', '得分', 'SCORE', 'POINTS'],
            'time': ['时间', '计时', 'TIME', 'TIMER'],
            'hp': ['生命', '血量', 'HP', 'HEALTH'],
            'mp': ['魔法', '能量', 'MP', 'MANA'],
            'gold': ['金币', '金钱', 'GOLD', 'MONEY'],
            'attack': ['攻击', '进攻', 'ATTACK'],
            'defense': ['防御', '防守', 'DEFENSE'],
            'skill': ['技能', '招式', 'SKILL'],
            'item': ['物品', '道具', 'ITEM'],
            'equip': ['装备', 'EQUIP', 'EQUIPMENT'],
            'shop': ['商店', '商城', 'SHOP'],
            'mission': ['任务', '使命', 'MISSION'],
            'quest': ['任务', 'QUEST'],
            'battle': ['战斗', '对战', 'BATTLE'],
            'victory': ['胜利', '获胜', 'VICTORY'],
            'defeat': ['失败', '败北', 'DEFEAT'],
            'loading': ['加载', '载入', 'LOADING'],
            'menu': ['菜单', '选单', 'MENU'],
            'settings': ['设置', '设定', 'SETTINGS'],
            'exit': ['退出', '离开', 'EXIT'],
            'continue': ['继续', 'CONTINUE'],
            'save': ['保存', '存档', 'SAVE'],
            'load': ['读取', '载入存档', 'LOAD']
        }
    
    def filter_results(self, ocr_results: List[Dict], 
                      min_confidence: Optional[float] = None) -> List[Dict]:
        """
        过滤OCR检测结果
        
        Args:
            ocr_results: OCR检测结果列表
            min_confidence: 最小置信度阈值，如果为None则使用默认阈值
            
        Returns:
            过滤后的结果列表
        """
        if min_confidence is None:
            min_confidence = self.confidence_threshold
        
        filtered_results = []
        for result in ocr_results:
            if result.get('confidence', 0) >= min_confidence:
                # 过滤掉空文字或过短文字
                text = result.get('text', '').strip()
                if len(text) >= 1:  # 至少1个字符
                    filtered_results.append(result)
        
        self.logger.debug(f"过滤后剩余 {len(filtered_results)} 个结果")
        return filtered_results
    
    def find_keywords(self, ocr_results: List[Dict], 
                     keyword_categories: List[str] = None) -> Dict[str, List[Dict]]:
        """
        在OCR结果中查找关键词
        
        Args:
            ocr_results: OCR检测结果列表
            keyword_categories: 要查找的关键词类别列表，如果为None则查找所有类别
            
        Returns:
            按关键词类别分组的结果字典
        """
        if keyword_categories is None:
            keyword_categories = list(self.game_keywords.keys())
        
        keyword_results = {}
        
        for category in keyword_categories:
            if category not in self.game_keywords:
                continue
                
            keywords = self.game_keywords[category]
            matched_results = []
            
            for result in ocr_results:
                text = result.get('text', '').strip()
                
                # 检查是否匹配关键词
                for keyword in keywords:
                    if self._text_match(text, keyword):
                        matched_result = result.copy()
                        matched_result['keyword'] = keyword
                        matched_result['category'] = category
                        matched_results.append(matched_result)
                        break  # 匹配到一个关键词就停止
            
            keyword_results[category] = matched_results
        
        return keyword_results
    
    def find_specific_text(self, ocr_results: List[Dict], 
                          target_texts: List[str]) -> List[Dict]:
        """
        查找特定文字
        
        Args:
            ocr_results: OCR检测结果列表
            target_texts: 目标文字列表
            
        Returns:
            匹配到的结果列表
        """
        matched_results = []
        
        for result in ocr_results:
            text = result.get('text', '').strip()
            
            for target in target_texts:
                if self._text_match(text, target):
                    matched_result = result.copy()
                    matched_result['target_text'] = target
                    matched_result['match_type'] = 'exact' if text == target else 'partial'
                    matched_results.append(matched_result)
                    break
        
        return matched_results
    
    def extract_numbers(self, ocr_results: List[Dict]) -> List[Dict]:
        """
        从OCR结果中提取数字信息
        
        Args:
            ocr_results: OCR检测结果列表
            
        Returns:
            包含数字的结果列表
        """
        number_results = []
        
        for result in ocr_results:
            text = result.get('text', '').strip()
            
            # 使用正则表达式提取数字
            numbers = re.findall(r'\d+', text)
            
            if numbers:
                number_result = result.copy()
                number_result['extracted_numbers'] = numbers
                number_result['numeric_value'] = int(numbers[0]) if numbers else 0
                number_results.append(number_result)
        
        return number_results
    
    def group_by_proximity(self, ocr_results: List[Dict], 
                          distance_threshold: int = 50) -> List[List[Dict]]:
        """
        根据位置相近度对OCR结果进行分组
        
        Args:
            ocr_results: OCR检测结果列表
            distance_threshold: 距离阈值，小于此距离的文字将被分到同一组
            
        Returns:
            分组后的结果列表
        """
        if not ocr_results:
            return []
        
        # 计算每个文字的中心点
        centers = []
        for result in ocr_results:
            bbox = result.get('bbox', [])
            if len(bbox) >= 4:
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                centers.append((center_x, center_y))
            else:
                centers.append((0, 0))
        
        # 简单的距离分组算法
        groups = []
        used_indices = set()
        
        for i in range(len(ocr_results)):
            if i in used_indices:
                continue
                
            group = [ocr_results[i]]
            used_indices.add(i)
            
            for j in range(i + 1, len(ocr_results)):
                if j in used_indices:
                    continue
                    
                # 计算距离
                dist = self._calculate_distance(centers[i], centers[j])
                if dist <= distance_threshold:
                    group.append(ocr_results[j])
                    used_indices.add(j)
            
            groups.append(group)
        
        return groups
    
    def sort_by_position(self, ocr_results: List[Dict], 
                        sort_order: str = 'top-left') -> List[Dict]:
        """
        根据位置对OCR结果进行排序
        
        Args:
            ocr_results: OCR检测结果列表
            sort_order: 排序方式，可选 'top-left', 'left-top', 'reading-order'
            
        Returns:
            排序后的结果列表
        """
        if not ocr_results:
            return []
        
        # 计算每个文字的中心点
        centers = []
        for result in ocr_results:
            bbox = result.get('bbox', [])
            if len(bbox) >= 4:
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                centers.append((center_x, center_y))
            else:
                centers.append((0, 0))
        
        # 根据排序方式排序
        if sort_order == 'top-left':
            # 先按Y坐标（从上到下），再按X坐标（从左到右）
            sorted_indices = sorted(range(len(centers)), 
                                  key=lambda i: (centers[i][1], centers[i][0]))
        elif sort_order == 'left-top':
            # 先按X坐标（从左到右），再按Y坐标（从上到下）
            sorted_indices = sorted(range(len(centers)), 
                                  key=lambda i: (centers[i][0], centers[i][1]))
        else:  # reading-order
            # 阅读顺序：先按行（Y坐标相近），再按列（X坐标）
            # 简单的实现：先按Y坐标分组，然后每组内按X坐标排序
            y_values = [center[1] for center in centers]
            y_threshold = 20  # Y坐标差异阈值
            
            # 分组
            groups = {}
            for i, y in enumerate(y_values):
                found_group = False
                for group_y in groups:
                    if abs(y - group_y) <= y_threshold:
                        groups[group_y].append(i)
                        found_group = True
                        break
                if not found_group:
                    groups[y] = [i]
            
            # 排序
            sorted_indices = []
            for group_y in sorted(groups.keys()):
                group_indices = groups[group_y]
                sorted_group = sorted(group_indices, key=lambda i: centers[i][0])
                sorted_indices.extend(sorted_group)
        
        return [ocr_results[i] for i in sorted_indices]
    
    def _text_match(self, text1: str, text2: str) -> bool:
        """
        判断两个文字是否匹配
        
        Args:
            text1: 第一个文字
            text2: 第二个文字
            
        Returns:
            是否匹配
        """
        # 完全匹配
        if text1 == text2:
            return True
        
        # 包含关系
        if text2 in text1 or text1 in text2:
            return True
        
        # 相似度匹配（使用编辑距离）
        try:
            import Levenshtein
            similarity = 1 - Levenshtein.distance(text1, text2) / max(len(text1), len(text2))
            return similarity >= self.similarity_threshold
        except ImportError:
            # 如果Levenshtein不可用，使用简单的字符重叠方法
            if not text1 or not text2:
                return False
            
            # 计算字符重叠比例
            set1 = set(text1)
            set2 = set(text2)
            overlap = len(set1.intersection(set2))
            total_chars = len(set1.union(set2))
            
            if total_chars == 0:
                return False
            
            similarity = overlap / total_chars
            return similarity >= self.similarity_threshold
        except:
            # 其他异常情况
            return False
    
    def _calculate_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """计算两点之间的欧几里得距离"""
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
    
    def add_custom_keywords(self, category: str, keywords: List[str]):
        """添加自定义关键词"""
        if category not in self.game_keywords:
            self.game_keywords[category] = []
        
        self.game_keywords[category].extend(keywords)
        self.game_keywords[category] = list(set(self.game_keywords[category]))  # 去重
    
    def remove_custom_keywords(self, category: str, keywords: List[str]):
        """移除自定义关键词"""
        if category in self.game_keywords:
            for keyword in keywords:
                if keyword in self.game_keywords[category]:
                    self.game_keywords[category].remove(keyword)


# 测试函数
def test_ocr_processor():
    """测试OCR处理器"""
    # 创建测试数据
    test_results = [
        {
            'text': '开始游戏',
            'confidence': 0.85,
            'bbox': [[10, 10], [50, 10], [50, 30], [10, 30]]
        },
        {
            'text': '100',
            'confidence': 0.92,
            'bbox': [[200, 20], [230, 20], [230, 40], [200, 40]]
        },
        {
            'text': '设置',
            'confidence': 0.78,
            'bbox': [[300, 10], [340, 10], [340, 30], [300, 30]]
        },
        {
            'text': '取消',
            'confidence': 0.45,  # 低置信度，应该被过滤
            'bbox': [[400, 10], [440, 10], [440, 30], [400, 30]]
        }
    ]
    
    # 创建处理器
    processor = OCRProcessor(confidence_threshold=0.6)
    
    # 测试过滤功能
    filtered = processor.filter_results(test_results)
    print(f"过滤后结果数量: {len(filtered)}")
    
    # 测试关键词查找
    keywords = processor.find_keywords(filtered)
    for category, results in keywords.items():
        if results:
            print(f"{category}: {[r['text'] for r in results]}")
    
    # 测试数字提取
    numbers = processor.extract_numbers(filtered)
    print(f"提取到的数字: {[n['text'] for n in numbers]}")


if __name__ == "__main__":
    test_ocr_processor()