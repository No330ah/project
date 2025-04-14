import os
from PIL import Image
import pytesseract
from openai import OpenAI
import xml.etree.ElementTree as ET
import json

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

def save_result_to_xml(data, output_path="outputs/result.xml"):
    root = ET.Element("DiagramObjects")
    obj = ET.SubElement(root, "Object")
    obj.set("type", data.get("type", "unknown"))

    if data.get("id"):
        ET.SubElement(obj, "ID").text = str(data["id"])
    if data.get("scale"):
        ET.SubElement(obj, "Scale").text = data["scale"]
    if data.get("linked_floor"):
        ET.SubElement(obj, "LinkedFloor").text = data["linked_floor"]
    if data.get("position"):
        pos = ET.SubElement(obj, "Position")
        for i, v in enumerate(data["position"]):
            ET.SubElement(pos, f"P{i+1}").text = str(v)
    if data.get("description"):
        ET.SubElement(obj, "Description").text = data["description"]

    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

def gpt_analyze_image(image_path, api_key=None):
    if api_key:
        client.api_key = api_key

    ocr_text = extract_text_from_image(image_path)
    prompt = f"""
你是建筑图纸结构提取助手，请从以下 OCR 内容中提取图素结构信息，并以 JSON 格式输出，字段包括：
- type（图素类型）
- id（编号）
- scale（比例）
- description（图素用途说明）
- position（坐标区域，格式为[x1, y1, x2, y2]，如无可为空）
- linked_floor（关联楼层，可为空）

以下是内容：
{text_wrap(ocr_text)}

请严格输出为 JSON 格式。
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个建筑图纸分析专家，返回结构化 JSON"},
                {"role": "user", "content": prompt}
            ]
        )
        result = response.choices[0].message.content.strip()

        if result.startswith("```json"):
            result = result.replace("```json", "").strip()
        if result.endswith("```"):
            result = result[:-3].strip()

        try:
            parsed = json.loads(result)
            save_result_to_xml(parsed)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return "GPT 输出无法解析为 JSON:\n" + result

    except Exception as e:
        return f"GPT 识别出错：{str(e)}"