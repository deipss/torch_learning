from moviepy import *
import os
from moviepy.audio.fx import AudioLoop


if __name__ == '__main__':

    # 设置资源路径
    audio_paths = ["chinese_output.mp3", "chinese_output.mp3", "chinese_output.mp3"]   # 替换为你的音频文件路径
    image_paths = ["images.jpeg", "images.jpeg", "images.jpeg"]  # 确保所有图片文件存在
    background_music_path = "chinese_output.mp3"                    # 背景音乐文件路径
    output_path = "final_video.mp4"                           # 最终视频输出路径

    video_clips = []

    # 遍历音频和对应图片，生成视频片段
    for audio_file, image_file in zip(audio_paths, image_paths):
        # 加载音频
        audio_clip = AudioFileClip(audio_file)
        duration = audio_clip.duration

        # 加载图像作为背景
        image_clip = ImageClip(image_file).with_duration(duration).with_fps(24)

        # 为图像添加音频
        video = image_clip.with_audio(audio_clip)
        video_clips.append(video)

    # 合并所有片段
    final_video = concatenate_videoclips(video_clips, method="compose")


    # 添加背景音乐
    bg_music = AudioFileClip(background_music_path).with_volume_scaled(0.3)  # 音量调小一点避免盖住语音
    bg_music = bg_music.with_duration(final_video.duration)
    bg_music = bg_music.with_effects([afx.AudioLoop(duration=final_video.duration)])

    # 混合原音频和背景音乐
    # final_audio = CompositeAudioClip([final_video.audio, bg_music])
    # final_video = final_video.with_audio(final_audio)


    # 添加字幕
    subtitles = [
        ("00:00:00", "00:00:10", "we sd"),  # 格式: (start, end, text)
        ("00:00:10", "00:00:20", "we sd"),
        ("00:00:20", "00:00:30", "we sd")
    ]

    # 创建字幕剪辑
    subtitle_clips = []
    for start, end, text in subtitles:
        subtitle_clip = TextClip(text=text,font='Arial', font_size=50, color='white')
        subtitle_clip = subtitle_clip.with_position(('center', 'bottom')).with_duration(10).with_start('10.2')
        subtitle_clips.append(subtitle_clip)

    # 将字幕剪辑与最终视频合并
    final_video = CompositeVideoClip([final_video] + subtitle_clips)



    # 导出最终视频
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
