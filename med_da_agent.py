import os
import json
import re
import time
import logging
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import xml.etree.ElementTree as ET

import arxiv
from openai import OpenAI
from dotenv import load_dotenv

# 1. 加载环境变量
load_dotenv()

# ==========================================
# 🚨 新增：强制让 Python 底层网络库走代理
# (请将 7897 替换为你实际的代理端口，你之前查到的是 7897)
# ==========================================
PROXY_URL = "http://127.0.0.1:7897" 
os.environ['http_proxy'] = PROXY_URL
os.environ['https_proxy'] = PROXY_URL
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL
# ==========================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 初始化 OpenAI 客户端（兼容 DeepSeek、智谱等）
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1")
)

# 模型名称从环境变量读取，默认 deepseek-chat
DEFAULT_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# 专属“灵魂 Prompt”（针对医学图像跨域自适应方向）
SYSTEM_PROMPT = """
你是一位计算机视觉领域的资深研究员，专精于【医学图像跨域自适应（Domain Adaptation/Generalization）】。
用户的研究背景：主要关注医学图像（MRI/CT/超声/病理等）的跨域问题，曾使用过协同训练（Co-training/Mean Teacher）、提示词引导（Prompt Tuning/Vision-Language Model）和对比学习（Contrastive Learning）等方法。

你的任务是阅读最新论文的标题和摘要，并输出结构化的分析报告。
请严格按照以下 JSON 格式输出，不要包含任何 Markdown 标记（如 ```json），不要包含任何其他多余字符：
{
    "relevance_score": 1到5的整数（1=完全不相关/纯自然图像，5=完美契合医学图像跨域且有重大创新）,
    "is_open_source": "是/否/未知",
    "one_sentence_summary": "用一句通俗的中文概括这篇论文解决了什么医学图像痛点",
    "core_innovation": "核心创新点（重点分析是否用到了对比学习、Prompt、协同训练，或提出了新的跨域特征对齐策略）",
    "inspiration_for_me": "这篇论文对我的研究有什么具体启发？能否迁移到我的课题中？"
}
"""


def retry(max_retries=3, delay=1, backoff=2):
    """
    简单的重试装饰器。在达到最大重试次数后返回 None，而不是抛出异常，
    以便在并行任务中优雅地跳过失败项，而不中断整个流程。
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"函数 {func.__name__} 重试 {max_retries} 次后仍失败，跳过此项: {e}")
                        return None  # 返回 None 而不是 raise
                    logger.warning(f"函数 {func.__name__} 第 {retries} 次失败，{current_delay} 秒后重试: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


@retry(max_retries=3, delay=2, backoff=2)
def search_arxiv(query, max_results=10, days=3):
    """
    使用 requests 直接请求 arXiv API，彻底绕过 arxiv 库的网络兼容性问题。
    """
    # 强制指定代理 (请确保这里的端口 7897 与你 test_proxy.py 中测试成功的一致)
    proxies = {
        "http": "http://127.0.0.1:7897",
        "https": "http://127.0.0.1:7897"
    }
    
    # 构建 arXiv API URL 和参数
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": max_results
    }
    
    papers = []
    cutoff_date = datetime.now() - timedelta(days=days)

    try:
        logger.info(f"正在通过代理请求 arXiv API...")
        # 发送请求，设置 15 秒超时
        response = requests.get(url, params=params, proxies=proxies, timeout=15)
        response.raise_for_status() # 如果返回 4xx 或 5xx，会抛出异常触发 retry
        
        # 解析 arXiv 返回的 Atom XML 格式数据
        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', namespace):
            title = entry.find('atom:title', namespace).text.strip()
            summary = entry.find('atom:summary', namespace).text.strip().replace('\n', ' ').replace('\r', ' ')
            url_link = entry.find('atom:id', namespace).text
            published_str = entry.find('atom:published', namespace).text
            
            # 解析时间并过滤 (处理带 'Z' 的 UTC 时间字符串)
            published_dt = datetime.fromisoformat(published_str.replace('Z', '+00:00')).replace(tzinfo=None)
            if published_dt < cutoff_date:
                continue
                
            papers.append({
                "title": title,
                "summary": summary,
                "url": url_link,
                "published": published_dt.strftime("%Y-%m-%d")
            })
            
    except Exception as e:
        logger.error(f"arXiv 检索失败: {e}")
        # 如果发生错误，返回已收集的部分结果（由 retry 装饰器决定是否重试）
        
    return papers


def safe_parse_json(content):
    """
    安全解析 JSON，兼容模型可能返回的多余文本或 Markdown 代码块，并确保关键字段类型正确。
    """
    if not content:
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None
            
    # 类型安全校验：确保 relevance_score 是整数
    if 'relevance_score' in parsed:
        try:
            parsed['relevance_score'] = int(float(parsed['relevance_score']))
        except (ValueError, TypeError):
            parsed['relevance_score'] = 1 # 解析失败给个保底低分
            
    return parsed


@retry(max_retries=3, delay=1, backoff=2)
def analyze_paper(title, summary, model=DEFAULT_MODEL):
    """
    调用大模型进行深度分析，返回解析后的 JSON 字典。
    """
    user_prompt = f"论文标题：{title}\n\n论文摘要：{summary}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    parsed = safe_parse_json(content)

    if parsed is None:
        logger.warning(f"模型未返回有效 JSON，原始内容: {content[:100]}...")
        return None
    return parsed


def analyze_papers_parallel(papers, model=DEFAULT_MODEL, max_workers=4):
    """
    并行分析多篇论文，使用线程池提高效率。
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_paper = {
            executor.submit(analyze_paper, paper['title'], paper['summary'], model): paper
            for paper in papers
        }
        # 处理完成的任务
        for future in as_completed(future_to_paper):
            paper = future_to_paper[future]
            try:
                analysis = future.result()
                if analysis:
                    results.append({"paper": paper, "analysis": analysis})
            except Exception as e:
                logger.error(f"分析论文失败: {paper['title'][:50]}... 错误: {e}")
    return results


def generate_report(papers_analysis):
    """
    生成 Markdown 格式的简报，按相关度排序并过滤低相关论文。
    """
    # 过滤掉分析失败 (analysis 为 None) 或缺失打分的数据
    valid_papers = [item for item in papers_analysis if item.get('analysis') and 'relevance_score' in item['analysis']]
    
    # 排序 (此时已确保 relevance_score 是 int)
    valid_papers.sort(key=lambda x: x['analysis'].get('relevance_score', 0), reverse=True)

    md_content = "# 🧠 MedDA-Agent 每日学术前沿简报\n\n"
    md_content += f"今日共检索并分析 {len(papers_analysis)} 篇潜在相关论文，其中 {len(valid_papers)} 篇解析成功。\n\n---\n\n"

    high_quality_count = 0
    for item in valid_papers:
        paper = item['paper']
        analysis = item['analysis']
        score = int(analysis.get('relevance_score', 0)) # 确保是整数进行比较
        
        if score < 3:  
            continue

        high_quality_count += 1
        md_content += f"### {high_quality_count}. {paper['title']}\n"
        md_content += (
            f"**🔗 链接**: [arXiv]({paper['url']}) | "
            f"**📅 日期**: {paper['published']} | "
            f"**⭐ 相关度**: {score}/5 | "
            f"**💻 开源**: {analysis.get('is_open_source', '未知')}\n\n"
        )
        md_content += f"**🎯 一句话总结**: {analysis.get('one_sentence_summary', '暂无总结')}\n\n"
        md_content += f"**💡 核心创新**: {analysis.get('core_innovation', '暂无')}\n\n"
        md_content += f"**🚀 对我的启发**: {analysis.get('inspiration_for_me', '暂无')}\n\n"
        md_content += "---\n\n"

    if high_quality_count == 0:
        md_content += "📭 今日暂无高度相关的优质论文，建议扩大检索范围或关注其他方向。\n"

    return md_content


def main():
    parser = argparse.ArgumentParser(description="MedDA-Agent: 医学图像跨域自适应论文智能分析助手")
    parser.add_argument('--max_results', type=int, default=15, help='检索论文的最大数量（默认15）')
    parser.add_argument('--days', type=int, default=7, help='只处理最近几天内提交的论文（默认7天）')
    parser.add_argument('--output', type=str, default='daily_paper_report.md', help='输出报告文件名')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='大模型名称（默认从环境变量读取）')
    parser.add_argument('--max_workers', type=int, default=1, help='并行分析的最大线程数（默认1）')
    args = parser.parse_args()

    logger.info("🚀 MedDA-Agent 启动，正在检索 arXiv...")
    query = (
        '("domain adaptation" OR "domain generalization" OR "unsupervised adaptation") '
        'AND ("medical image" OR "medical imaging" OR "MRI" OR "CT" OR "ultrasound" OR "pathology")'
    )

    papers = search_arxiv(query, max_results=args.max_results, days=args.days)

    if not papers:
        logger.warning("❌ 未检索到任何论文，请检查网络或 arXiv 查询条件。")
        return

    logger.info(f"✅ 检索到 {len(papers)} 篇论文，开始调用大模型深度分析...")
    papers_analysis = analyze_papers_parallel(papers, model=args.model, max_workers=args.max_workers)

    logger.info("📝 正在生成简报...")
    report = generate_report(papers_analysis)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"✅ 任务完成！简报已保存为: {args.output}")
    logger.info("💡 提示：在 VS Code 中打开该文件，点击右上角的 '打开预览' 图标即可查看美观的排版。")


if __name__ == "__main__":
    main()