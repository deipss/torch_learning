from langchain.chains import TransformChain, SequentialChain
from langchain.agents import AgentExecutor, Tool
from langchain_core.runnables import RunnableParallel
import os


class NewsPPTAgent:
    def __init__(self):
        # 初始化子链
        self.process_chain = self._create_process_chain()
        self.audio_chain = self._create_audio_chain()
        self.image_chain = self._create_image_chain()
        self.ppt_chain = self._create_ppt_chain()

        # 构建完整流程
        self.full_chain = RunnableParallel(
            processed_news=self.process_chain,
            audio_files=self.audio_chain,
            images=self.image_chain
        ) | self.ppt_chain

    def _create_process_chain(self):
        def news_processor(inputs):
            # 新闻预处理逻辑
            return {"keywords": ["关键词1", "关键词2"], "summary": "新闻摘要"}

        return TransformChain(
            input_variables=["news"],
            output_variables=["processed_news"],
            transform=news_processor
        )

    def _create_audio_chain(self):
        def audio_generator(inputs):
            # 音频生成逻辑
            return {"audio_path": "audio.mp3"}

        return TransformChain(
            input_variables=["processed_news"],
            output_variables=["audio_files"],
            transform=audio_generator
        )

    def _create_image_chain(self):
        def image_generator(inputs):
            # 宫崎骏风格图片生成逻辑
            return {"image_path": "image.png"}

        return TransformChain(
            input_variables=["processed_news"],
            output_variables=["images"],
            transform=image_generator
        )

    def _create_ppt_chain(self):
        def ppt_assembler(inputs):
            # PPT组装逻辑
            return {"ppt_path": "news.pptx"}

        return TransformChain(
            input_variables=["processed_news", "audio_files", "images"],
            output_variables=["ppt_path"],
            transform=ppt_assembler
        )

    def run(self, news_list):
        return self.full_chain.batch([{"news": news} for news in news_list])


from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI


def create_miyazaki_image_chain():
    prompt = PromptTemplate(
        input_variables=["keywords", "summary"],
        template="""生成宫崎骏风格插画提示词：
        要求：吉卜力动画风格、柔和水彩质感、奇幻场景
        新闻关键词：{keywords}
        新闻摘要：{summary}
        输出格式：英文提示词，包含'miyazaki hayao style'"""
    )

    return LLMChain(
        llm=OpenAI(temperature=0.7),
        prompt=prompt,
        output_key="image_prompt"
    )


def generate_image(image_prompt):
    from openai import OpenAI
    client = OpenAI()

    response = client.images.generate(
        model="dall-e-3",
        prompt=image_prompt + ", Studio Ghibli style, watercolor texture, fantasy elements",
        size="1024x1024",
        quality="standard",
        n=1
    )
    return response.data[0].url


from langchain.tools import tool
from openai import OpenAI


@tool
def generate_news_audio(text: str, output_dir: str = "audio"):
    """生成新闻语音并保存为MP3"""
    client = OpenAI()

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{hash(text)}.mp3"
    path = os.path.join(output_dir, filename)

    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",  # 适合新闻播报的声音
        input=text
    )
    response.stream_to_file(path)
    return path



from pptx import Presentation
from pptx.util import Inches
import requests
from PIL import Image
import io


def create_ppt_page(slide, title, image_path, audio_path):
    # 添加标题
    title_shape = slide.shapes.title
    title_shape.text = title

    # 添加图片
    img = Image.open(requests.get(image_path, stream=True).raw)
    img_io = io.BytesIO()
    img.save(img_io, format='PNG')

    left = Inches(1)
    top = Inches(1.5)
    slide.shapes.add_picture(img_io, left, top, width=Inches(6))

    # 添加音频（Windows系统可用）
    if os.name == 'nt':
        slide.shapes.add_movie(
            audio_path,
            left, Inches(5),
            width=Inches(1),
            height=Inches(1))


def generate_ppt(pages_data, output_path="news_report.pptx"):
    prs = Presentation()
    for data in pages_data:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        create_ppt_page(
            slide,
            title=data['title'],
            image_path=data['image_url'],
            audio_path=data['audio_path']
        )
    prs.save(output_path)