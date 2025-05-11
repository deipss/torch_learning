import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime

# 设置请求头，模拟浏览器访问
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

URLS = [
    'https://cn.chinadaily.com.cn/',
    'https://china.chinadaily.com.cn/',
    'https://world.chinadaily.com.cn/',
    'https://cn.chinadaily.com.cn/gtx/'

]

NEWS_JSON_FILE_NAME = "news_results.json"
CN_NEWS_FOLDER_NAME = "cn_news"


def is_sensitive_word(word):
    cnt = 0
    sensitive_words = ["平", "%%", "习", "&&&&&#", "近"]  # 去除了重复项
    for sensitive_word in sensitive_words:
        if sensitive_word in word:
            cnt += 1
    return cnt > 2


# 创建以当前日期命名的文件夹
def create_folder():
    today = datetime.now().strftime("%Y%m%d")
    folder_path = os.path.join(CN_NEWS_FOLDER_NAME, today)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def fetch_page(url):
    """获取页面内容"""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"[WARN] 请求失败: {url} - {e}")
        return None


def extract_links(html, visited_urls):
    """解析 HTML，提取所有链接"""
    soup = BeautifulSoup(html, "html.parser")
    today = datetime.now().strftime("%Y%m/%d")
    urls = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if today in href and href not in visited_urls:
            visited_urls.add(href)
            urls.add(href)
    return urls


def extract_news_content(url):
    """提取新闻页面的标题、图片和正文内容"""
    try:

        # 获取页面内容
        html = fetch_page(url)
        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")

        # 提取标题
        title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "无标题"

        # 提取正文图片
        images = []
        article_div = soup.find("div", class_="Artical_Content")
        if article_div:
            for img in article_div.select("img"):
                img_url = img.get("src")
                if img_url and not img_url.startswith("data:"):
                    images.append(urljoin(url, img_url))

        # 提取正文文本
        content = ""
        for p in soup.select("p"):
            text = p.get_text(strip=True)
            if text and len(text) > 10:  # 过滤短文本
                content += text + " "

        return {
            "title": title,
            "images": images,
            "content": content.strip()
        }

    except Exception as e:
        print(f"提取新闻内容出错: {e}")
        return None


def extract_all_not_visit_urls():
    # 初始爬取目标页面
    visited_urls = set()
    full_urls = []
    for base_url in URLS:

        html = fetch_page(base_url)
        if not html:
            print("无法获取初始页面内容，程序退出。")
            return

        # 提取所有链接
        urls = extract_links(html, visited_urls)
        print(f"{base_url} 共发现 {len(urls)} 个链接。")

    for url in visited_urls:
        if url.startswith("//"):
            full_urls.append("https:" + url)
        else:
            full_urls.append(url)
    print(f"去重共发现 {len(visited_urls)} 个链接。")
    return full_urls


import json  # Add this import at the top with other imports


def crawling_news_meta():
    folder_path = create_folder()
    urls = extract_all_not_visit_urls()
    results = []
    for id, url in enumerate(urls):
        context = extract_news_content(url)
        if not context:
            print(f"[ERROR] 无法获取新闻内容: {url}")
            continue
        if len(context['images']) == 0:
            print(f"[ERROR] 未找到图片: {url}")
            continue
        if is_sensitive_word(context['title']):
            print(f"[ERROR] 标题包含敏感词: {url}")
            continue
        if is_sensitive_word(context['content']):
            print(f"[ERROR] 标题包含敏感词: {url}")
            continue
        context['folder'] = "{:04d}".format(id)
        context['url'] = url
        results.append(context)
    for id, context in enumerate(results):
        context['folder'] = "{:04d}".format(id)
    json_path = os.path.join(folder_path, "%s" % NEWS_JSON_FILE_NAME)
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)


def download_images():
    today = datetime.now().strftime("%Y%m%d")
    today_path = os.path.join(CN_NEWS_FOLDER_NAME, today)
    json_folder_path = os.path.join(today_path, NEWS_JSON_FILE_NAME)
    if not os.path.exists(json_folder_path):
        crawling_news_meta()
    else:
        print("数据已存在，跳过爬取。")
    with open(json_folder_path, "r", encoding="utf-8") as json_file:
        results = json.load(json_file)
        for result in results:
            folder = result['folder']
            img_folder_path = os.path.join(today_path, folder)
            os.makedirs(img_folder_path, exist_ok=True)
            for id, image_url in enumerate(result['images']):
                image_name = os.path.basename(image_url)
                image_path = os.path.join(img_folder_path, image_name)
                if not os.path.exists(image_path):
                    try:
                        response = requests.get(image_url)
                        response.raise_for_status()
                        with open(image_path, "wb") as image_file:
                            image_file.write(response.content)
                    except requests.RequestException as e:
                        print(f"[ERROR] 下载图片失败: {image_url} - {e}")
    print("图片下载完成。")


if __name__ == "__main__":
    download_images()
