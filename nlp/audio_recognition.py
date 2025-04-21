if __name__ == '__main__':
    import whisper

    model = whisper.load_model("medium")

    # 指定语言（中文）
    result = model.transcribe("/data/audio/人事处沟通.mp3", language="zh")
    print("原始中文文本：", result["text"])
    # 将中文音频翻译为英文文本
    result_translated = model.transcribe("/data/audio/人事处沟通.mp3", task="translate", language="zh")


    print("翻译为英文：", result_translated["text"])
