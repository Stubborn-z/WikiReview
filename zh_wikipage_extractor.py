import argparse
import json
import os
import pathlib
import re

import pandas as pd
import requests
import wikipediaapi
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

# 初始化中文NER模型
ner_tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese-ner")
ner_model = AutoModelForTokenClassification.from_pretrained("ckiplab/bert-base-chinese-ner")
ner_pipeline = pipeline(
    "ner",
    model=ner_model,
    tokenizer=ner_tokenizer,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)

def get_references(sentence, reference_dict):
    """处理参考文献标记"""
    refs = re.findall(r'\[\d+\]', sentence)
    sentence = re.sub(r'\[\d+\]', '', sentence).strip().replace("\n", "")
    return sentence, [reference_dict[ref.replace("[", "").replace("]", "")] for ref in refs]

# def extract_data(url, reference_dict):
#     """提取结构化内容"""
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     data = {}

#     for header in soup.find_all(['h1', 'h2', 'h3', "h4", "h5", "h6"]):
#         section_title = header.text.strip()
#         section_data = []

#         # 遍历该章节下的所有文本段落
#         for sibling in header.find_next_siblings():
#             if sibling.name in ['h1', 'h2', 'h3', "h4", "h5", "h6"]:
#                 break  # 进入下一个章节，停止收集当前章节内容
            
#             if sibling.name == 'p':
#                 sentences = re.split(r'(?<=[。！？；])', sibling.text.strip())  # 句子分割
#                 for sentence in sentences:
#                     if sentence:
#                         sentence, refs = get_references(sentence, reference_dict)
#                         section_data.append({"sentence": sentence, "refs": refs})

#         if section_data:
#             data[section_title] = section_data

#     return data

def extract_data(url, reference_dict):
    """提取结构化内容（改进版）"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    data = {}

    # 清理标题文本
    def clean_title(title):
        return re.sub(r'\[\d+\]|\s*\[编辑\]', '', title).strip()

    current_section = None
    current_level = 0

    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
        if element.name.startswith('h'):
            # 处理标题
            title = clean_title(element.get_text())
            current_level = int(element.name[1])
            current_section = title
            data[current_section] = []
        elif current_section and element.name == 'p':
            # 处理段落内容
            paragraph = element.get_text().strip()
            if not paragraph:
                continue
            
            # 分割句子并处理参考文献
            sentences = re.split(r'(?<=[。！？；])', paragraph)
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    # 处理引用标记
                    clean_sent, refs = get_references(sent, reference_dict)
                    if clean_sent:
                        data[current_section].append({
                            "sentence": clean_sent,
                            "refs": refs
                        })

    # 合并子章节到父章节
    final_data = {}
    current_parent = None
    for title, content in data.items():
        if content:  # 只保留有内容的章节
            final_data[title] = content

    return final_data

def extract_references(url):
    """提取参考文献"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    references = {}

    # 适配不同维基百科结构
    ref_sections = soup.find_all('ol', {'class': 'references'})
    if not ref_sections:
        ref_sections = soup.find_all('div', {'class': 'reflist'})

    for ref_section in ref_sections:
        for i, ref in enumerate(ref_section.find_all('li'), start=1):
            ext_link = ref.find('a', {'class': 'external'})
            if ext_link:
                references[str(i)] = ext_link['href']
            else:
                references[str(i)] = ref.text.strip()

    return references

def getSections(page, structured_data):
    """构建章节结构（改进版）"""
    def recursive_section(section):
        title = section.title.strip()
        return {
            "section_title": title,
            "section_content": structured_data.get(title, []),
            "subsections": [recursive_section(sub) for sub in section.sections]
        }
    
    return [recursive_section(section) for section in page.sections]

def get_wikipedia_json_output(username, url):
    """获取维基数据"""
    wiki_api = wikipediaapi.Wikipedia(username, 'zh')
    page_name = url.replace("https://zh.wikipedia.org/wiki/", "")
    page = wiki_api.page(page_name)

    reference_dict = extract_references(url)
    structured_data = extract_data(url, reference_dict)

    return {
        "title": page_name,
        "url": url,
        "summary": page.summary,
        "content": getSections(page, structured_data),
        "references": reference_dict
    }, page_name, reference_dict


def section_dict_to_text(data, inv_ref_dict, level=1):
    """生成文本内容"""
    title = data["section_title"]
    content = data["section_content"]
    subsections = data["subsections"]
    
    output = f"\n\n{'#' * level} {title}"
    if content:
        output += "\n\n"
        for sent_info in content:
            sentence = sent_info["sentence"]
            refs = [f"[{inv_ref_dict[ref]}]" for ref in sent_info["refs"] if ref != "[无法获取链接]"]
            output += f"{sentence}{' ' + ' '.join(refs) if refs else ''}。"
    
    for sub in subsections:
        output += section_dict_to_text(sub, inv_ref_dict, level+1)
    return output

def output_as_text(result, ref_dict):
    """生成最终文本"""
    inv_ref_dict = {v:k for k,v in ref_dict.items()}
    text = f"# {result['title']}\n\n{result['summary']}\n"
    
    for section in result["content"]:
        text += section_dict_to_text(section, inv_ref_dict)
    
    text += "\n\n## 参考文献\n"
    for idx, link in ref_dict.items():
        text += f"[{idx}] {link}\n"
    
    return text

def extract_entities(text):
    """使用BERT模型进行实体识别"""
    try:
        max_length = 510
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        entities = []
        for chunk in chunks:
            results = ner_pipeline(chunk)
            entities.extend([res for res in results if res['score'] > 0.8])
        
        merged = []
        for entity in entities:
            if merged and merged[-1]['entity_group'] == entity['entity_group'] \
                    and merged[-1]['end'] == entity['start']:
                merged[-1]['word'] += entity['word']
                merged[-1]['end'] = entity['end']
            else:
                merged.append(entity)
        
        valid_types = {'PER', 'ORG', 'LOC', 'GPE', 'DATE', 'EVENT', 'ORDINAL', 'CARDINAL', 'NORP'}
        return list({
            ent['word'].strip(): ent['entity_group']
            for ent in merged
            if ent['entity_group'] in valid_types
        }.items())
    
    except Exception as e:
        print(f"实体识别错误: {str(e)}")
        return []

def process_url(url, output_dir, username='Knowledge Curation Project'):
    """处理单个URL"""
    result, page_name, ref_dict = get_wikipedia_json_output(username, url)
    
    # 生成干净文本
    clean_text = re.sub(r'\s+', '', result['summary'])
    for section in result['content']:
        for content in section['section_content']:
            clean_text += re.sub(r'\s+', '', content['sentence'])
    
    # 提取实体
    entities = extract_entities(clean_text)
    result['entities'] = [{"name": name, "type": type_} for name, type_ in entities]
    
    # 保存文件
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", page_name)
    
    # 保存JSON
    json_path = os.path.join(output_dir, 'json', f"{safe_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 保存TXT
    txt = output_as_text(result, ref_dict)
    txt_path = os.path.join(output_dir, 'txt', f"{safe_name}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(txt)

def main(args):
    """主函数"""
    pathlib.Path(f'{args.output_dir}/json').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{args.output_dir}/txt').mkdir(parents=True, exist_ok=True)
    
    if args.batch:
        df = pd.read_csv(args.batch_file)
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                process_url(row['url'], args.output_dir)
            except Exception as e:
                print(f"处理失败: {row['url']} - {str(e)}")
    else:
        process_url(args.url, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='中文维基百科内容提取工具')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    parser.add_argument('--batch_file', help='批量处理文件路径（CSV格式）')
    parser.add_argument('-u', '--url', 
                       default='https://zh.wikipedia.org/wiki/2025%E5%B9%B4%E7%92%B0%E5%8F%B0%E8%BB%8D%E4%BA%8B%E6%BC%94%E7%B7%B4',
                       help='维基页面URL')
    parser.add_argument('-o', '--output_dir',
                       default='./',
                       help='输出目录')
    
    args = parser.parse_args()
    main(args)