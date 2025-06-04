import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from abc import abstractmethod
from dataclasses import dataclass, asdict
from typing import List

CHINADAILY = 'chinadaily'
EASTDAY = 'eastday'
BBC = 'bbc'

import os
import requests
from datetime import datetime

# 设置请求头，模拟浏览器访问
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

NEWS_JSON_FILE_NAME = "news_results.json"
CN_NEWS_FOLDER_NAME = "cn_news"


@dataclass
class NewsArticle:
    """
    数据类，用于存储新闻文章的相关信息。

    Attributes:
        title (str): 新闻标题。
        images (List[str]): 新闻文章中的图片文件名列表。
        image_urls (List[str]): 新闻文章中图片的URL列表。
        content_cn (str): 新闻内容的中文版本。
        content_en (str): 新闻内容的英文版本。
        folder (str): 新闻所属的文件夹名称。
        index_inner (int): 新闻在内部索引中的位置。
        index_show (int): 新闻在展示索引中的位置。
        url (str): 新闻的原始URL。
        source (str): 新闻来源。
        news_type (str): 新闻类型（如：国内新闻、国际新闻等）。
        publish_time (str): 新闻发布时间。
        author (str): 新闻作者。
        tags (List[str]): 新闻标签列表。
        summary (str): 新闻摘要。
    """

    def __init__(self,
                 title: str = None,
                 images: List[str] = None,
                 image_urls: List[str] = None,
                 content_cn: str = None,
                 content_en: str = None,
                 folder: str = None,
                 index_inner: int = None,
                 index_show: int = None,
                 url: str = None,
                 source: str = None,
                 news_type: str = None,
                 publish_time: str = None,
                 author: str = None,
                 tags: List[str] = None,
                 summary: str = None):
        self.title = title
        self.images = images or []
        self.image_urls = image_urls or []
        self.content_cn = content_cn
        self.content_en = content_en
        self.folder = folder
        self.index_inner = index_inner
        self.index_show = index_show
        self.url = url
        self.source = source
        self.news_type = news_type
        self.publish_time = publish_time
        self.author = author
        self.tags = tags or []
        self.summary = summary

    def to_dict(self):
        return self.__dict__


class NewsScraper:

    def __init__(self, source_url: str, source: str, news_type: str):
        self.source_url = source_url
        self.source = source
        self.news_type = news_type

    @abstractmethod
    def origin_url(self):
        """返回新闻网站的原始URL"""
        pass

    def is_sensitive_word(self, word) -> bool:
        cnt = 0
        sensitive_words = ["平", "%%", "习", "&&&&&#", "近", "县"]  # 去除了重复项
        for sensitive_word in sensitive_words:
            if sensitive_word in word:
                cnt += 1
        return cnt > 1

    # 创建以当前日期命名的文件夹
    def create_folder(self, today=datetime.now().strftime("%Y%m%d")) -> str:
        folder_path = os.path.join(CN_NEWS_FOLDER_NAME, today, self.source)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def fetch_page(self, url):
        """获取页面内容"""
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"[WARN] 请求失败: {url} 错误信息： {e}")
            return None


class ChinaDailyScraper(NewsScraper):

    def origin_url(self) -> list[str]:
        return [
            'https://cn.chinadaily.com.cn/',
            'https://china.chinadaily.com.cn/',
            'https://world.chinadaily.com.cn/',
            'https://cn.chinadaily.com.cn/gtx/'
        ]

    def truncate_after_300_find_period(self, text: str) -> str:
        if len(text) <= 300:
            return text

        end_pos = 300  # 第300个字符的位置（索引从0开始，取前300个字符）

        # 从end_pos位置开始向后查找第一个句号
        last_period = text.find('。', end_pos)

        if last_period != -1:
            # 截取至句号位置（包含句号）
            return text[:last_period + 1]
        else:
            # 300字符后无句号，返回全文（或截断并添加省略号）
            return text  # 或返回 text[:end_pos] + "..."（按需选择）

    def extract_news_content(self, url) -> NewsArticle:
        """提取新闻页面的标题、图片和正文内容"""
        try:

            # 获取页面内容
            html = self.fetch_page(url)
            if not html:
                return None

            soup = BeautifulSoup(html, "html.parser")

            # 提取标题
            title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "无标题"
            title = '【CHINA DAILY】' + title
            # 提取正文图片
            image_urls = []
            article_div = soup.find("div", class_="Artical_Content")
            if article_div:
                for img in article_div.select("img"):
                    img_url = img.get("src")
                    if img_url and not img_url.startswith("data:"):
                        image_urls.append(urljoin(url, img_url))

            # 提取正文文本
            content = ""
            for p in soup.select("p"):
                text = p.get_text(strip=True)
                if text and len(text) > 10:  # 过滤短文本
                    content += text + " "
            content = self.truncate_after_300_find_period(content)
            article = NewsArticle(source=self.source, news_type=self.news_type)
            article.title = title
            article.content_cn = content.strip()
            article.url = url
            article.image_urls = image_urls
            article.images = [os.path.basename(i) for i in image_urls]
            return article

        except Exception as e:
            print(f"提取新闻内容出错: {e}")
            return None

    def extract_all_not_visit_urls(self, today):
        # 初始爬取目标页面
        visited_urls = set()
        full_urls = []
        for base_url in self.origin_url():

            html = self.fetch_page(base_url)
            if not html:
                print("无法获取初始页面内容，程序退出。")
                return

            # 提取所有链接
            urls = self.extract_links(html, visited_urls, today)
            print(f"{base_url} 共发现 {len(urls)} 个链接。")

        for url in visited_urls:
            if url.startswith("//"):
                full_urls.append("https:" + url)
            else:
                full_urls.append(url)
        print(f"去重共发现 {len(visited_urls)} 个链接。")
        return full_urls

    def extract_links(self, html, visited_urls, today) -> set[str]:
        """解析 HTML，提取所有链接"""
        soup = BeautifulSoup(html, "html.parser")
        if today is None:
            today = datetime.now().strftime("%Y%m/%d")
        else:
            today = datetime.strptime(today, "%Y%m%d").strftime("%Y%m/%d")
        urls = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if today in href and href not in visited_urls:
                visited_urls.add(href)
                urls.add(href)
        return urls

    def crawling_news_meta(self, today) -> List[NewsArticle]:
        folder_path = self.create_folder(today)
        urls = self.extract_all_not_visit_urls(today)
        results = []
        if urls is None:
            print("无法获取初始页面内容，程序退出。")
            return results
        for id, url in enumerate(urls):
            article = self.extract_news_content(url)
            if not article:
                print(f"[ERROR] 无法获取新闻内容: {url}")
                continue
            if len(article.images) == 0:
                print(f"[ERROR] 未找到图片: {url}")
                continue
            if self.is_sensitive_word(article.title):
                print(f"[ERROR] 标题包含敏感词: {url}")
                continue
            if self.is_sensitive_word(article.content_cn):
                print(f"[ERROR] 标题包含敏感词: {url}")
                continue
            if '/gtx/' in url:
                print(f"[ERROR] URL 包含敏感词gtx : {url}")
                continue
            if len(article.content_cn) < 10:
                print(f"[ERROR] 内容过短: {url}")
                continue
            article.folder = "{:04d}".format(id)
            results.append(article)
        for id, article in enumerate(results):
            article.folder = "{:04d}".format(id)
            article.index_inner = id
            article.index_show = id
        json_path = os.path.join(folder_path, "%s" % NEWS_JSON_FILE_NAME)
        json_results = [i.to_dict() for i in results]
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(json_results, json_file, ensure_ascii=False, indent=4)
        return results

    def download_images(self, today: datetime.now().strftime("%Y%m%d")):
        today_path = os.path.join(CN_NEWS_FOLDER_NAME, today, self.source)
        results = []
        if not os.path.exists(today_path):
            results = self.crawling_news_meta(today)
        else:
            print("数据已存在，跳过爬取。")
        for result in results:
            folder = result.folder
            img_folder_path = os.path.join(today_path, folder)
            os.makedirs(img_folder_path, exist_ok=True)
            for id, image_url in enumerate(result.image_urls):
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
    cs = ChinaDailyScraper(source_url='https://cn.chinadaily.com.cn/', source=CHINADAILY, news_type='国内新闻')
    cs.download_images(today='20250604')
