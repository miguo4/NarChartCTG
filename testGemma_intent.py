from PIL import Image
import requests
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
import os
# 强制CUDA同步，让错误精准显示
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
# torch.hub.download_url_to_file('https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png', 'chart_example_1.png')

# image_path = "./content/chart_example_1.png"
image_path = "./content/nar_chart.png"
# input_text ="program of thought: what is the sum of Faceboob Messnger and Whatsapp values in the 18-29 age group?"
input_text = "Describe this chart in detail.\n"  
# input_text ="Analyze this chart carefully.\n"  

# # Load Model
# model = PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma", torch_dtype=torch.float16)
# processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")




# ====================== 1. 核心配置 ======================
class ChartGemmaIntentConfig:
    # 意图驱动提示词模板（适配 Intentable Recipe）
    INTENT_PROMPT_TEMPLATES = {
        "overview": "Based on the chart image, provide a concise overview of the key data trends and totals. Follow the Intentable overview intent: summarize the overall picture with core values and chart type.",
        "describe": "Based on the chart image, describe the specific data point for the category '{target_key}' (series: {target_series} if applicable). Follow the Intentable describe intent: state the exact value and label clearly.",
        # "compare": "Based on the chart image, compare the two data points: {target1_key} (series: {target1_series}) vs {target2_key} (series: {target2_series} if applicable). Follow the Intentable compare intent: calculate and state the difference and percentage change.",
        # "trend": "Based on the chart image, analyze the trend of the series '{target_series}' over the categories {target_keys}. Follow the Intentable trend intent: describe if the trend is increasing, decreasing, or fluctuating, and note key inflection points."
    }
    # -------------------------- 示例1：本地图表图像 --------------------------
    CHART_IMAGE_PATH = "./content/nar_chart.png" 
    # 模拟 Intentable 的 Recipe 结构，覆盖四大核心意图
    intent_recipes = [
        # 意图1：概览（Overview）
        {
            "action": "overview",
            "chart_type": "bar",
            "unit": "survival time"
        },
        # 意图2：描述（Describe）- 提取2023年的数值
        {
            "action": "describe",
            "targets": [{"key": "Public", "series": "3.5 K"}]
        },
        # 意图3：对比（Compare）- 对比2022和2023年Product A
        # {
        #     "action": "compare",
        #     "targets": [
        #         {"key": "2022", "series": "Product A"},
        #         {"key": "2023", "series": "Product A"}
        #     ]
        # },
        # # 意图4：趋势（Trend）- 分析2021-2023年Product A的趋势
        # {
        #     "action": "trend",
        #     "targets": [
        #         {"key": "2021", "series": "Product A"},
        #         {"key": "2022", "series": "Product A"},
        #         {"key": "2023", "series": "Product A"}
        #     ]
        # }
    ]
    
class ChartGemmaIntentEngine:
    def __init__(self):
       self.config = ChartGemmaIntentConfig() 
       LOCAL_MODEL_DIR = "./chartgemma"
        # 加载本地模型 + 处理器
       self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            LOCAL_MODEL_DIR,
            torch_dtype=torch.float16,
            local_files_only=True,  # 强制只读本地，不联网
            trust_remote_code=True  # 自定义模型必加
        )

       self.processor = AutoProcessor.from_pretrained(
            LOCAL_MODEL_DIR,
            local_files_only=True,
            trust_remote_code=True
        )

       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       self.model = self.model.to(self.device)
    #    print("✅ Load Model成功：")
       
    def load_chart_image(self, image_path_or_url):
        """加载图表图像（支持本地路径/URL）"""
        # if image_path_or_url.startswith("http"):
        #     from requests import get
        #     from io import BytesIO
        #     response = get(image_path_or_url)
        #     image = Image.open(BytesIO(response.content)).convert("RGB")
        # else:
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
        
        print("-" * 50)
        print(f"核心推理函数\n")
        print("意图模板prompt_template",prompt_template,) 
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
        print("意图生成：",prompt) 
        print(f"\n")
        # Process Inputs
        inputs = self.processor(text= input_text, images=chart_image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)# # ✅ 关键修复：把输入转成和型一样的半精
        prompt_length = inputs['input_ids'].shape[1]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # print("✅ Process Inputs成功：")

        # Generate
        generate_ids = self.model.generate(**inputs, num_beams=4, max_new_tokens=512)
        output_text = self.processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print("✅ Generate")
        # print(output_text)

        return output_text    






  

def intent_based_chart_inference():
    # 初始化引擎
    engine = ChartGemmaIntentEngine()
    
    # 加载图表图像
    image_path = engine.config.CHART_IMAGE_PATH
    chart_image = engine.load_chart_image(image_path)
    intent_recipes = engine.config.intent_recipes
    # 批量执行意图推理
    results = {}
    for idx, recipe in enumerate(intent_recipes):
        result = engine.generate_intent_based_inference(chart_image, recipe)
        results[f"intent_{idx+1}"] = result
        print(f"\n🔍 执行意图 {idx+1}：{recipe['action']}")
        print(f"📋 意图详情：{recipe}")
        print(f"✅ 生成结果：{result}\n")

    return results




if __name__ == "__main__":
    # -------------------------- 执行推理 --------------------------
        inference_results = intent_based_chart_inference()
        print("\n" + "="*80)
        print("📊 最终推理结果汇总（适配 Intentable 风格）")
        print("="*80)
        for intent_name, result in inference_results.items():
            print(f"\n🎯 {intent_name}：\n{result}")