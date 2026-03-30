import os
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    set_seed
)


# ====================== 1. 核心配置 ======================
class ChartGemmaIntentConfig:
    # 模型配置
    MODEL_NAME = "ahmed-masry/chartgemma"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 生成参数（平衡流畅度与准确性）
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7  # 控制多样性，0.5-0.9 为宜
    NUM_BEAMS = 4     # 束搜索，提升生成质量
    REPETITION_PENALTY = 1.2  # 避免重复生成
    # 意图驱动提示词模板（适配 Intentable Recipe）
    INTENT_PROMPT_TEMPLATES = {
        "overview": "Based on the chart image, provide a concise overview of the key data trends and totals. Follow the Intentable overview intent: summarize the overall picture with core values and chart type.",
        "describe": "Based on the chart image, describe the specific data point for the category '{target_key}' (series: {target_series} if applicable). Follow the Intentable describe intent: state the exact value and label clearly.",
        "compare": "Based on the chart image, compare the two data points: {target1_key} (series: {target1_series}) vs {target2_key} (series: {target2_series} if applicable). Follow the Intentable compare intent: calculate and state the difference and percentage change.",
        "trend": "Based on the chart image, analyze the trend of the series '{target_series}' over the categories {target_keys}. Follow the Intentable trend intent: describe if the trend is increasing, decreasing, or fluctuating, and note key inflection points."
    }

# ====================== 2. ChartGemma 推理核心类 ======================
class ChartGemmaIntentEngine:
    def __init__(self):
        self.config = ChartGemmaIntentConfig()
        # 加载模型与处理器（PaliGemma 专用）
        self.processor = AutoProcessor.from_pretrained(self.config.MODEL_NAME)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.config.MODEL_NAME,
            torch_dtype=torch.bfloat16 if self.config.DEVICE == "cuda" else torch.float32
        ).to(self.config.DEVICE)
        self.model.eval()  # 推理模式，关闭 dropout

    def load_chart_image(self, image_path_or_url):
        """加载图表图像（支持本地路径/URL）"""
        if image_path_or_url.startswith("http"):
            from requests import get
            from io import BytesIO
            response = get(image_path_or_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # 本地图片路径
            image = Image.open(image_path_or_url).convert("RGB")
        return image

    def generate_intent_based_inference(self, chart_image, intent_recipe):
        """
        核心推理函数：适配 Intentable Recipe 生成结果
        :param chart_image: PIL.Image 图表图像
        :param intent_recipe: 结构化意图（字典），格式参考 Intentable Recipe
        :return: 生成的自然语言结果
        """
        # 1. 校验意图合法性
        intent_action = intent_recipe.get("action")
        if intent_action not in self.config.INTENT_PROMPT_TEMPLATES:
            raise ValueError(f"不支持的意图类型：{intent_action}，仅支持 {list(self.config.INTENT_PROMPT_TEMPLATES.keys())}")

        # 2. 填充意图模板，生成推理指令
        prompt_template = self.config.INTENT_PROMPT_TEMPLATES[intent_action]
        # 提取 Recipe 中的目标信息
        target_key = intent_recipe.get("targets", [{}])[0].get("key", "") if intent_recipe.get("targets") else ""
        target_series = intent_recipe.get("targets", [{}])[0].get("series", "") if intent_recipe.get("targets") else ""
        target_keys = ", ".join([t["key"] for t in intent_recipe.get("targets", [])]) if intent_recipe.get("targets") else ""

        # 填充模板（处理可选字段）
        prompt = prompt_template.format(
            target_key=target_key,
            target_series=target_series,
            target1_key=target_key,
            target1_series=target_series,
            target2_key=intent_recipe.get("targets", [{}])[1].get("key", "") if len(intent_recipe.get("targets", [])) > 1 else "",
            target2_series=intent_recipe.get("targets", [{}])[1].get("series", "") if len(intent_recipe.get("targets", [])) > 1 else "",
            target_keys=target_keys
        )

        # 3. 处理多模态输入（图像 + 文本指令）
        inputs = self.processor(
            images=chart_image,
            text=prompt,
            return_tensors="pt"
        ).to(self.config.DEVICE)

        # 4. 生成推理结果（禁用梯度计算，加速）
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                temperature=self.config.TEMPERATURE,
                num_beams=self.config.NUM_BEAMS,
                repetition_penalty=self.config.REPETITION_PENALTY,
                do_sample=True if self.config.TEMPERATURE > 0 else False  # 温度>0 时采样，否则束搜索
            )

        # 5. 解码生成结果（跳过特殊 token）
        result = self.processor.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return result

# ====================== 3. 适配 Intentable 的多意图生成接口 ======================
def intent_based_chart_inference(image_path, intent_recipes):
    """
    适配 Intentable 风格的批量意图生成
    :param image_path: 图表图像路径/URL
    :param intent_recipes: 列表，每个元素是 Intentable 风格的 Recipe 字典
    :return: 字典，key 为意图索引，value 为生成结果
    """
    # 初始化引擎
    engine = ChartGemmaIntentEngine()
    # 加载图表图像
    chart_image = engine.load_chart_image(image_path)
    # 批量执行意图推理
    results = {}
    for idx, recipe in enumerate(intent_recipes):
        print(f"\n🔍 执行意图 {idx+1}：{recipe['action']}")
        print(f"📋 意图详情：{recipe}")
        print("-" * 50)
        result = engine.generate_intent_based_inference(chart_image, recipe)
        results[f"intent_{idx+1}"] = result
        print(f"✅ 生成结果：{result}\n")

    return results

# ====================== 4. 示例运行（适配 Intentable 场景） ======================
if __name__ == "__main__":
    # -------------------------- 示例1：本地图表图像 --------------------------
    # 替换为你的本地图表路径（支持 bar/line/pie 等）
    CHART_IMAGE_PATH = "./sample_chart.png"  # 示例：柱状图/折线图

    # -------------------------- 示例2：Intentable 风格意图配置 --------------------------
    # 模拟 Intentable 的 Recipe 结构，覆盖四大核心意图
    intent_recipes = [
        # 意图1：概览（Overview）
        {
            "action": "overview",
            "chart_type": "bar",
            "unit": "million USD"
        },
        # 意图2：描述（Describe）- 提取2023年的数值
        {
            "action": "describe",
            "targets": [{"key": "2023", "series": "Product A"}]
        },
        # 意图3：对比（Compare）- 对比2022和2023年Product A
        {
            "action": "compare",
            "targets": [
                {"key": "2022", "series": "Product A"},
                {"key": "2023", "series": "Product A"}
            ]
        },
        # 意图4：趋势（Trend）- 分析2021-2023年Product A的趋势
        {
            "action": "trend",
            "targets": [
                {"key": "2021", "series": "Product A"},
                {"key": "2022", "series": "Product A"},
                {"key": "2023", "series": "Product A"}
            ]
        }
    ]

    # -------------------------- 执行推理 --------------------------
    if os.path.exists(CHART_IMAGE_PATH):
        inference_results = intent_based_chart_inference(CHART_IMAGE_PATH, intent_recipes)
        print("\n" + "="*80)
        print("📊 最终推理结果汇总（适配 Intentable 风格）")
        print("="*80)
        for intent_name, result in inference_results.items():
            print(f"\n🎯 {intent_name}：\n{result}")
    else:
        print(f"❌ 本地图表文件不存在：{CHART_IMAGE_PATH}")
        print("💡 请替换为真实的图表图像路径/URL，或使用在线示例（如：https://i.imgur.com/7Z8X7.png）")

    # -------------------------- 在线图表示例（无需本地文件） --------------------------
    # 取消注释以下代码，直接测试在线图表
    """
    ONLINE_CHART_URL = "https://i.imgur.com/7Z8X7.png"  # 示例：Statista 风格柱状图
    inference_results = intent_based_chart_inference(ONLINE_CHART_URL, intent_recipes)
    print("\n" + "="*80)
    print("📊 在线图表推理结果汇总")
    print("="*80)
    for intent_name, result in inference_results.items():
        print(f"\n🎯 {intent_name}：\n{result}")
    """