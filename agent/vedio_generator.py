from moviepy import *
from PIL import Image, ImageDraw
import datetime
from zhdate import ZhDate

import textwrap

BACKGROUND_IMAGE_PATH = "images/generated_background.png"
GLOBAL_WIDTH = 2560
GLOBAL_HEIGHT = 1440
GAP = int(GLOBAL_WIDTH * 0.02)
INNER_WIDTH = GLOBAL_WIDTH - GAP
INNER_HEIGHT = GLOBAL_HEIGHT - GAP
W_H_RADIO = GLOBAL_WIDTH / GLOBAL_HEIGHT
FPS = 60

NEWS_MATE = {
    "title": "《我是歌手》 2024 年 1 月 1 日 00:00 发布",
    "content_cn": "《我是歌手》 2024 年 1 月 1 日 00:00 发布",
    "content_en": "《我是歌手》 2024 年 1 月 1 日 00:00 发布",
    "image": "images/news.png",
    "audio": "audios/news.mp3",
    "md5": "12345678901234567890123456789012",
    "id": ""
}


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


def generate_quad_layout_video(audio_path, image_path_top, txt_cn, txt_en, output_path):
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

    # 右上文字处理 todo

    top_right_txt = TextClip(
        text=txt_cn,
        color='white',
        font='./font/simhei.ttf',
        size=(right_width, top_height),
        bg_color='#FFCC99',
        method='label'
    ).with_duration(duration).with_position((left_width, 'top'))

    # 左下文字处理 todo
    wrapped_text = textwrap.fill(txt_en, width=bottom_left_width * 9 // 10)
    font_size = min(30, int(bottom_height / (wrapped_text.count('\n') + 1) * 0.8))
    bottom_left_txt = TextClip(
        text=wrapped_text,
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


def combine_videos_with_transitions():
    # todo 组装生成今天的信息
    # 设置三个视频的资源路径
    video_paths = [
        "videos/intro.mp4",
        "videos/quad_layout_video_1.mp4",
        "videos/quad_layout_video_2.mp4"
    ]

    output_path = "videos/combined_final.mp4"
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


def temp():
    generate_background_image(GLOBAL_WIDTH, GLOBAL_HEIGHT)
    generate_video_intro("audios/1.mp3", "videos/intro.mp4")
    generate_quad_layout_video(
        output_path="videos/quad_layout_video_2.mp4",
        audio_path="audios/vallex_generation.wav",
        image_path_top="images/quantum.png",
        txt_cn=
        """
        美国总统特朗普4日宣布对所有进入美国、在外国制作的电影征收100%关税，这一决定持续引发业界强烈反对。 美国和加拿大电影电视行业从业者的工会组织——国际戏剧舞台从业者联盟近日发布声明表示，鉴于加拿大与美国的文化和经济伙伴关系，美国政府需要采取措施，恢复公平竞争环境，维护美加两国的电影和电视行业利益。 国际戏剧舞台从业者联盟主席 马修·勒布：我们希望创造公平的竞争环境，并正在寻求惠及所有成员的解决方案，尤其是电视剧、小成本电影和独立电影。我们期待美国政府就拟议的关税计划提供更多信息，但任何的贸易决策都不能损害我们加拿大成员和整个行业的利益。 国际戏剧舞台从业者联盟是一个有着超百年历史的美国和加拿大联合工会组织。联盟成立于1893年，1898年以来一直代表美国和加拿大的影视业幕后从业者，在美加两地有超17万名业内人员。勒布表示，成千上万的家庭、小企业和社区承受着行业萎缩带来的经济压力，关税将对该联盟造成严重影响。此外，鉴于加拿大与美国独特的文化和经济伙伴关系，联盟认为应特别考虑加拿大的电影和电视制作。
        """
        ,
        txt_en="""
            This came as Israeli Prime Minister Benjamin Netanyahu said on Wednesday there is doubt over the survival of three hostages previously believed alive in Gaza. His statement came a day after US President Donald Trump said only 21 of 24 hostages believed alive had survived.The news sent families of remaining captives in Gaza into panic.The new bloodshed on Wednesday came days after Israel approved a plan to intensify its operations in the Palestinian enclave, which would include seizing Gaza, holding on to captured territories, forcibly displacing Palestinians to southern Gaza and taking control of aid distribution along with private security companies.
            """
    )

    generate_quad_layout_video(
        output_path="videos/quad_layout_video_1.mp4",
        audio_path="audios/vallex_generation.wav",
        image_path_top="images/quantum.png",
        txt_cn=
        """
        美国总统特朗普4日宣布对所有进入美国、在外国制作的电影征收100%关税，这一决定持续引发业界强烈反对。 美国和加拿大电影电视行业从业者的工会组织——国际戏剧舞台从业者联盟近日发布声明表示，鉴于加拿大与美国的文化和经济伙伴关系，美国政府需要采取措施，恢复公平竞争环境，维护美加两国的电影和电视行业利益。 国际戏剧舞台从业者联盟主席 马修·勒布：我们希望创造公平的竞争环境，并正在寻求惠及所有成员的解决方案，尤其是电视剧、小成本电影和独立电影。我们期待美国政府就拟议的关税计划提供更多信息，但任何的贸易决策都不能损害我们加拿大成员和整个行业的利益。 国际戏剧舞台从业者联盟是一个有着超百年历史的美国和加拿大联合工会组织。联盟成立于1893年，1898年以来一直代表美国和加拿大的影视业幕后从业者，在美加两地有超17万名业内人员。勒布表示，成千上万的家庭、小企业和社区承受着行业萎缩带来的经济压力，关税将对该联盟造成严重影响。此外，鉴于加拿大与美国独特的文化和经济伙伴关系，联盟认为应特别考虑加拿大的电影和电视制作。
        """
        ,
        txt_en="""
            This came as Israeli Prime Minister Benjamin Netanyahu said on Wednesday there is doubt over the survival of three hostages previously believed alive in Gaza. His statement came a day after US President Donald Trump said only 21 of 24 hostages believed alive had survived.The news sent families of remaining captives in Gaza into panic.The new bloodshed on Wednesday came days after Israel approved a plan to intensify its operations in the Palestinian enclave, which would include seizing Gaza, holding on to captured territories, forcibly displacing Palestinians to southern Gaza and taking control of aid distribution along with private security companies.
            """
    )

    combine_videos_with_transitions()


if __name__ == '__main__':
    temp()
