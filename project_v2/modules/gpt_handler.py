import os
from PIL import Image
import pytesseract
from openai import OpenAI

# 配置 Tesseract OCR 路径（确保已安装 Tesseract）
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 初始化 OpenAI 客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='chi_sim')  # 使用中文 OCR
    return text.strip()

def text_wrap(text):
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def gpt_analyze_image(image_path, api_key=None):
    if api_key:
        client.api_key = api_key

    ocr_text = extract_text_from_image(image_path)
    prompt = f"""你是建筑图纸分析专家，请根据以下 OCR 文字判断图素内容：

{text_wrap(ocr_text)}

请输出格式如下：
- 图素类型：
- 编号（如有）：
- 结构组成/用途说明："""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个工程图纸分析专家，擅长识别图素结构。"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT 识别出错：{str(e)}"
