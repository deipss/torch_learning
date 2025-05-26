from moviepy import *
from PIL import ImageDraw
import datetime
from zhdate import ZhDate
import os
import math
from PIL import Image
from pathlib import Path

BACKGROUND_IMAGE_PATH = "images/generated_background.png"
GLOBAL_WIDTH = 2560
GLOBAL_HEIGHT = 1440
GAP = int(GLOBAL_WIDTH * 0.02)
INNER_WIDTH = GLOBAL_WIDTH - GAP
INNER_HEIGHT = GLOBAL_HEIGHT - GAP
W_H_RADIO = GLOBAL_WIDTH / GLOBAL_HEIGHT
FPS = 60
NEWS_JSON_FILE_NAME = "news_results.json"
CN_NEWS_FOLDER_NAME = "cn_news"


def create_region_bg(width, height, color='#FFFFFF', duration=1):
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [(0, 0),
         (width, height)],
        fill=color
    )
    temp_path = f"temp/region_bg_{width}x{height}.png"
    img.save(temp_path)
    return ImageClip(temp_path).with_duration(duration)


def add_newline_every_n_chars(text, n):
    """
    每隔固定的字数在文本中添加换行符

    参数:
    text (str): 需要添加换行符的长文本
    n (int): 指定的固定字数

    返回:
    str: 添加换行符后的文本
    """
    if n <= 0:
        return text

    return '\n'.join([text[i:i + n] for i in range(0, len(text), n)])


def calculate_font_size_and_line_length(text, box_width, box_height, font_ratio=1.0, line_height_ratio=1.5,
                                        start_size=72):
    """
    计算适合文本框的字体大小和每行字数

    参数:
    text (str): 要显示的文本
    box_width (int): 文本框宽度（像素）
    box_height (int): 文本框高度（像素）
    font_ratio (float): 字体大小与平均字符宽度的比例系数
    line_height_ratio (float): 行高与字体大小的比例系数
    start_size (int): 开始尝试的最大字体大小

    返回:
    dict: 包含计算结果的字典，键为 'font_size' 和 'chars_per_line'
    """
    # 从最大字体开始尝试，逐步减小直到文本适应文本框
    for font_size in range(start_size, 0, -1):
        # 计算每个字符的平均宽度和行高
        char_width = font_size * font_ratio
        line_height = font_size * line_height_ratio

        # 计算每行可容纳的字符数
        chars_per_line = max(1, math.floor(box_width / char_width))

        # 计算所需的总行数
        total_lines = math.ceil(len(text) / chars_per_line)

        # 计算所需的总高度
        total_height = total_lines * line_height

        # 如果高度符合要求，返回当前字体大小和每行字符数
        if total_height <= box_height:
            return font_size, chars_per_line

    return 40, len(text)


def generate_quad_layout_video(audio_path, image_path_top, txt_cn, title, output_path):
    # 加载背景和音频
    bg_clip = ColorClip(size=(INNER_WIDTH, INNER_HEIGHT), color=(255, 255, 255))  # 白色背景
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    bg_clip = bg_clip.with_duration(duration).with_audio(audio_clip)
    bg_width, bg_height = bg_clip.size

    # 计算各区域尺寸
    top_height = int(bg_height * 0.7)
    bottom_height = bg_height - top_height
    left_width = int(bg_width * 0.4)
    right_width = bg_width - left_width
    bottom_left_width = int(bg_width * 0.7)
    bottom_right_width = bg_width - bottom_left_width

    # 左上图片处理
    top_left_img = ImageClip(image_path_top)
    if top_left_img.w > left_width or top_left_img.h > top_height:
        scale = min(left_width / top_left_img.w, top_height / top_left_img.h)
        top_left_img = top_left_img.resized(scale)
    offset_w, offest_h = (left_width - top_left_img.w) // 2, (top_height - top_left_img.h) // 2
    top_left_img = top_left_img.with_position((offset_w, offest_h)).with_duration(duration)

    font_size, chars_per_line = calculate_font_size_and_line_length(txt_cn, right_width * 95 / 100,
                                                                    top_height * 95 / 100)
    txt_cn = '\n'.join([txt_cn[i:i + chars_per_line] for i in range(0, len(txt_cn), chars_per_line)])
    top_right_txt = TextClip(
        interline=font_size // 2,
        text=txt_cn,
        font_size=font_size,
        color='white',
        font='./font/simhei.ttf',
        size=(right_width, top_height),
        bg_color='#FFCC99',
        method='label'
    ).with_duration(duration).with_position((left_width, 'top'))

    font_size, chars_per_line = calculate_font_size_and_line_length(title, bottom_left_width * 95 / 100,
                                                                    bottom_height * 95 / 100)
    title = '\n'.join([title[i:i + chars_per_line] for i in range(0, len(title), chars_per_line)])
    bottom_left_txt = TextClip(
        text=title,
        interline=font_size // 2,
        font_size=font_size,
        color='black',
        font='./font/simhei.ttf',
        size=(bottom_left_width, bottom_height),
        method='caption'
    ).with_duration(duration).with_position(('left', top_height))

    # 右下图片处理 todo
    bottom_right_img = ImageClip('images/head.png')
    if bottom_right_img.w > bottom_right_width or bottom_right_img.h > bottom_height:
        scale = min(bottom_right_width / bottom_right_img.w, bottom_height / bottom_right_img.h)
        bottom_right_img = bottom_right_img.resized(scale)
    bottom_right_img = bottom_right_img.with_position((bottom_left_width, top_height)).with_duration(duration)

    # 创建各区域背景框
    top_left_bg = create_region_bg(left_width, top_height, '#F5F5F5', duration=duration)
    top_right_bg = create_region_bg(right_width, top_height, '#F0F0F0', duration=duration)
    bottom_left_bg = create_region_bg(bottom_left_width, bottom_height, '#EEEEEE', duration=duration)
    bottom_right_bg = create_region_bg(bottom_right_width, bottom_height, '#F8F8F8', duration=duration)

    # 合成最终视频
    final_video = CompositeVideoClip([
        bg_clip,
        top_left_bg.with_position(('left', 'top')),
        top_right_bg.with_position((left_width, 'top')),
        bottom_left_bg.with_position(('left', top_height)),
        bottom_right_bg.with_position((bottom_left_width, top_height)),
        top_left_img,
        top_right_txt,
        bottom_left_txt,
        bottom_right_img
    ], size=(bg_width, bg_height))
    # final_video.preview()

    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=FPS)


def generate_background_image(width, height, color='#FF9900'):
    # 创建一个新的图像
    image = Image.new("RGB", (width, height), color)  # 橘色背景
    draw = ImageDraw.Draw(image)

    # 计算边框宽度(1%的宽度)
    border_width = GAP

    # 绘制圆角矩形(内部灰白色)
    draw.rounded_rectangle(
        [(border_width, border_width), (width - border_width, height - border_width)],
        radius=30,  # 圆角半径
        fill="#F0F0F0"  # 灰白色填充
    )

    image.save(BACKGROUND_IMAGE_PATH)
    return image


def get_full_date():
    """获取完整的日期信息：公历日期、农历日期和星期"""
    today = datetime.datetime.now()

    # 获取公历日期
    solar_date = today.strftime("%Y年%m月%d日")

    # 获取农历日期
    lunar_date = ZhDate.from_datetime(today).chinese()

    # 获取星期几
    weekday_map = ["一", "二", "三", "四", "五", "六", "日"]
    weekday = f"星期{weekday_map[today.weekday()]}"

    return "今天是{}, \n农历{}, \n{}".format(solar_date, lunar_date, weekday)


def generate_video_intro(bg_music_path, output_path="videos/intro.mp4"):
    """生成带日期文字和背景音乐的片头视频

    Args:
        bg_music_path: 背景音乐文件路径
        output_path: 输出视频路径
    """

    # 加载背景图片
    bg_clip = ImageClip(BACKGROUND_IMAGE_PATH)

    # 加载背景音乐
    audio_clip = AudioFileClip(bg_music_path)
    duration = audio_clip.duration

    # 设置背景视频时长
    bg_clip = bg_clip.with_duration(duration).with_audio(audio_clip)

    # 创建日期文字
    date_text = get_full_date()
    date_parts = date_text.split('\n')
    max_length = max(len(part) for part in date_parts) if date_parts else len(date_text)

    txt_clip = TextClip(
        text=date_text,
        font_size=int(GLOBAL_WIDTH / max_length * 0.9),
        color='white',
        font='./font/simhei.ttf',
        stroke_color='black',
        stroke_width=2
    ).with_duration(duration).with_position(('center', 0.7), relative=True)

    # 合成最终视频
    final_clip = CompositeVideoClip([bg_clip, txt_clip], size=bg_clip.size)
    final_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=24
    )


def combine_videos_with_transitions(video_paths):
    output_path = build_today_combine_video_path()
    bg_clip = ImageClip(BACKGROUND_IMAGE_PATH)

    # 加载视频和音频
    clips = []
    for i, video_path in enumerate(video_paths):
        # 加载视频
        video = VideoFileClip(video_path)
        video = video.with_position(('center', 'center'), relative=True)
        # 将视频放置在背景上
        video_with_bg = CompositeVideoClip([
            bg_clip,
            video
        ], use_bgclip=True)
        # 将视频放置在背景上
        clips.append(video_with_bg)

    # 合并视频，添加1秒的过渡效果
    final_clip = concatenate_videoclips(clips, method="compose")

    final_clip = final_clip.with_speed_scaled(1.5)
    # 导出最终视频
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=FPS)
    # final_clip.preview()


def temp_video_text_align():
    generate_quad_layout_video(
        output_path="videos/quad_layout_video_4.mp4",
        audio_path="audios/vallex_generation.wav",
        image_path_top="images/quantum.png",
        txt_cn=
        """
        美国总统特朗普4日宣布对所有进入美国、在外国制作的电影征收100%关税，这一决定持续引发业界强烈反对。 美国和加拿大电影电视行业从业者的工会组织——国际戏剧舞台从业者联盟近日发布声明表示，鉴于加拿大与美国的文化和经济伙伴关系，美国政府需要采取措施，恢复公平竞争环境，维护美加两国的电影和电视行业利益。 国际戏剧舞台从业者联盟主席 马修·勒布：我们希望创造公平的竞争环境，并正在寻求惠及所有成员的解决方案，尤其是电视剧、小成本电影和独立电影。我们期待美国政府就拟议的关税计划提供更多信息，但任何的贸易决策都不能损害我们加拿大成员和整个行业的利益。 国际戏剧舞台从业者联盟是一个有着超百年历史的美国和加拿大联合工会组织。联盟成立于1893年，1898年以来一直代表美国和加拿大的影视业幕后从业者，在美加两地有超17万名业内人员。勒布表示，成千上万的家庭、小企业和社区承受着行业萎缩带来的经济压力，关税将对该联盟造成严重影响。此外，鉴于加拿大与美国独特的文化和经济伙伴关系，联盟认为应特别考虑加拿大的电影和电视制作。
        """
        ,
        title="""美国总统特朗普4日宣布对所有进入美国、在外国制作的电影征收100%关税，"""
    )


def build_today_path():
    return os.path.join(CN_NEWS_FOLDER_NAME, datetime.datetime.now().strftime("%Y%m%d"))


def build_today_news_path():
    return os.path.join(CN_NEWS_FOLDER_NAME, datetime.datetime.now().strftime("%Y%m%d"), NEWS_JSON_FILE_NAME)


def build_today_intro_path():
    return os.path.join(CN_NEWS_FOLDER_NAME, datetime.datetime.now().strftime("%Y%m%d"), "intro.mp4")


def build_today_combine_video_path():
    return os.path.join(CN_NEWS_FOLDER_NAME, datetime.datetime.now().strftime("%Y%m%d"), "video.mp4")


def build_today_video_path(index: str):
    return os.path.join(CN_NEWS_FOLDER_NAME, datetime.datetime.now().strftime("%Y%m%d"), index, "video.mp4")


def build_today_bg_music_path(index: str):
    return os.path.join(CN_NEWS_FOLDER_NAME, datetime.datetime.now().strftime("%Y%m%d"), index, "bg_music.mp4")


def build_today_index_path(index: str):
    return os.path.join(CN_NEWS_FOLDER_NAME, datetime.datetime.now().strftime("%Y%m%d"), index)




def find_highest_resolution_image(directory: str) -> tuple[str, int, int] | None:
    """
    遍历指定目录（包括子目录）找出分辨率最高的图片

    参数:
    directory (str): 要搜索的目录路径

    返回:
    tuple: (图片路径, 宽度, 高度) 或 None(如果未找到图片)
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"指定的路径 '{directory}' 不是有效目录")

    # 支持的图片扩展名集合
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif'}

    highest_resolution = 0
    best_image = None

    # 使用 os.walk 遍历目录树
    for root, _, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为支持的图片格式
            ext = Path(file).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                file_path = os.path.join(root, file)

                try:
                    # 使用 PIL 打开图片并获取尺寸
                    with Image.open(file_path) as img:
                        width, height = img.size
                        resolution = width * height

                        # 如果当前图片分辨率更高，则更新最佳图片
                        if resolution > highest_resolution:
                            highest_resolution = resolution
                            best_image = (file_path, width, height)
                except Exception as e:
                    print(f"无法处理图片 {file_path}: {e}")

    return best_image



def combine_videos():
    import json
    generate_background_image(GLOBAL_WIDTH, GLOBAL_HEIGHT)
    video_paths = []
    intro_path = build_today_intro_path()
    video_paths.append(intro_path)
    # todo generate audio for introduction
    # todo generate audio for every news
    generate_video_intro("audios/1.mp3", intro_path)
    with open(build_today_news_path(), "r", encoding="utf-8") as f:
        news_list = json.load(f)
        for i, news in enumerate(news_list):
            index_path = build_today_index_path(news['folder'])
            video_path = build_today_video_path(news['folder'])
            img_path, w, h = find_highest_resolution_image(index_path)
            generate_quad_layout_video(
                output_path=video_path,
                audio_path="audios/1.mp3",
                image_path_top=img_path,
                txt_cn=news["content"],
                title=news["title"]
            )
            video_paths.append(video_path)
            if i > 2:
                break
    combine_videos_with_transitions(video_paths)


# 示例使用
if __name__ == "__main__":
    combine_videos_with_transitions(
        ['cn_news/20250512/intro.mp4', 'cn_news/20250512/0000/video.mp4', 'cn_news/20250512/0001/video.mp4'])
