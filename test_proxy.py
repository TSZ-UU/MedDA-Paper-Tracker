import requests

# 你的代理地址 (请确认你的代理软件端口确实是 7897)
proxies = {
    "http": "http://127.0.0.1:7897",
    "https": "http://127.0.0.1:7897"
}

print("正在尝试通过代理连接 arXiv...")
try:
    # 请求 arXiv API，设置 10 秒超时
    response = requests.get(
        "https://export.arxiv.org/api/query?search_query=all:electron&max_results=1", 
        proxies=proxies, 
        timeout=10
    )
    if response.status_code == 200:
        print("✅ 代理连接成功！你的代理配置完全没问题。")
    else:
        print(f"⚠️ 连接返回了异常状态码: {response.status_code}")
except Exception as e:
    print(f"❌ 代理连接失败！错误信息: {e}")