import os


def process_md_files(directory):
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                modify_md_file(file_path)


def modify_md_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            # 如果该行不以 "nav_order" 开头，则写回文件
            if not line.strip().startswith("nav_order"):
                file.write(line)


if __name__ == "__main__":
    process_md_files('/Users/deipss/workspace/self/blog/docs')
    print("处理完成")

