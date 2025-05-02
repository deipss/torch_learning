import os

from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip, concatenate_audioclips


def create_video_with_bgm(audio_files, image_files, bgm_path, output_path,
                          bgm_volume=0.3, transition_duration=1.0, fps=24):
    """
    参数说明：
    audio_files: 主音频文件列表（与图片一一对应）
    image_files: 背景图片列表（与音频一一对应）
    bgm_path: 背景音乐文件路径
    output_path: 输出视频路径
    bgm_volume: 背景音乐音量（0.0-1.0）
    transition_duration: 转场动画时长（秒）
    fps: 视频帧率
    """
    # 校验输入
    if len(audio_files) != len(image_files):
        raise ValueError("音频与图片数量必须一致")
    if not os.path.exists(bgm_path):
        raise FileNotFoundError(f"背景音乐文件 {bgm_path} 不存在")

    # 生成视频片段列表
    video_clips = []
    main_audios = []

    for audio_path, img_path in zip(audio_files, image_files):
        # 加载主音频
        main_audio = AudioFileClip(audio_path)
        main_audios.append(main_audio)

        # 创建图片剪辑（匹配音频时长）
        img_clip = ImageClip(img_path).with_duration(main_audio.duration)

        # 添加转场效果
        video_clip = img_clip.with_audio(main_audio)
        video_clips.append(video_clip)

    # 计算总时长
    total_duration = sum([a.duration for a in main_audios])
    #
    # # 加载并处理背景音乐
    # bgm = AudioFileClip(bgm_path).with_volume_scaled(bgm_volume)
    # if bgm.duration < total_duration:
    #     bgm = bgm.with_duration(duration=total_duration)  # 循环背景音乐
    # else:
    #     bgm = bgm.subclip(0, total_duration)  # 截取背景音乐
    #
    # # 混合音频轨道
    # final_audio = CompositeAudioClip([
    #     concatenate_audioclips(main_audios),  # 主音频轨道
    #     bgm.with_start(0)  # 背景音乐从0秒开始
    # ])

    # 生成视频轨道（带转场）
    final_video = concatenate_videoclips(
        [v.resized(video_clips[0].size) for v in video_clips],
        padding=-transition_duration,  # 负值产生重叠转场
        method="compose"
    )

    # 高级渲染设置
    final_video.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset='medium',
        ffmpeg_params=[
            '-crf', '20',
            '-movflags', '+faststart',
            '-filter_complex', '[0:a]loudnorm=I=-16:TP=-1.5:LRA=11[a]',  # 音频标准化
            '-map', '[a]'  # 应用音频滤镜
        ]
    )

if __name__ == '__main__':
    # 使用示例
    create_video_with_bgm(
        audio_files=["chinese_output.mp3", "chinese_output.mp3", "chinese_output.mp3"],  # 确保所有音频文件存在
        image_files=["images.jpeg", "images.jpeg", "images.jpeg"],  # 确保所有图片文件存在
        bgm_path="chinese_output.mp3",
        output_path="final_video.mp4",
        bgm_volume=0.25,  # 背景音乐音量设置为25%
        transition_duration=0.8,
        fps=30
    )
