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

LOCAL_MODEL_DIR = "./chartgemma"
# 加载本地模型 + 处理器
model = PaliGemmaForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_DIR,
    torch_dtype=torch.float16,
    local_files_only=True,  # 强制只读本地，不联网
    trust_remote_code=True  # 自定义模型必加
)

processor = AutoProcessor.from_pretrained(
    LOCAL_MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("✅ Load Model成功：")


# Process Inputs
image = Image.open(image_path).convert('RGB')
inputs = processor(text=input_text, images=image, return_tensors="pt")
inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)# # ✅ 关键修复：把输入转成和型一样的半精
prompt_length = inputs['input_ids'].shape[1]
inputs = {k: v.to(device) for k, v in inputs.items()}
print("✅ Process Inputs成功：")

# Generate
generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512)
output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("✅ Generate")
print(output_text)
