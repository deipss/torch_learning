# 导入 python-docx 库中的 Document 类，用于处理 .docx 文件
from docx import Document
import re
# 导入 WD_BREAK 枚举类型，用于检测分栏符等特殊格式
from docx.enum.text import WD_BREAK
from nltk.corpus.europarl_raw import english
from sympy import false


def recognized_cn_en(input_text: str):
    # 定义正则表达式模式，用于匹配中文和英文部分
    # 匹配中文、中文标点 / 以及空格
    chinese_pattern = r'[\u4e00-\u9fa5()，。、；：“”‘’（）—/ ]+'
    english_pattern = r'[a-zA-Z_\-, ;]+'
    # 使用正则表达式查找所有中文和英文部分
    chinese_list = re.findall(chinese_pattern, input_text)
    english_list = re.findall(english_pattern, input_text)
    # 去除 list 中的空串
    chinese_list = [chinese.replace(" ", "").strip() for chinese in chinese_list if chinese.strip()]
    english_list = [re.sub(r'\s+', ' ', eng).strip() for eng in english_list if eng.strip()]

    # 输出对齐的语料
    aligned_corpus = list(zip(chinese_list, english_list))
    warning_flag = False
    result = []
    warns = []
    for chinese, english in aligned_corpus:
        if len(chinese_list) != len(english_list):
            warning_flag = True
            # print(f'warning start ,raw text={input_text}')
        # print(f"{chinese.strip()}={english.strip()}")
        result.append(f'{chinese.strip()}={english.strip()}')
    if warning_flag:
        warns.append(f'{input_text}')
    return result, warns


# 定义函数 read_columns_from_docx，传入参数为 docx 文件路径
def read_vocabulary_from_docx(docx_path):
    # 打开并加载指定路径的 .docx 文件
    doc = Document(docx_path)
    result_list = []
    warn_list = []
    # 遍历文档中的所有段落
    for paragraph in doc.paragraphs:
        # 如果段落有非空文本，则将其添加到当前栏
        if not paragraph.text.strip():
            continue

        result, warns = recognized_cn_en(paragraph.text)
        result_list.extend(result)
        warn_list.extend(warns)
    with open('vocabulary.txt', 'w+', encoding='utf-8') as file:
        for line in result_list:
            file.write(line + '\n')

    with open('warns.txt', 'w+', encoding='utf-8') as file:
        for line in warn_list:
            file.write(line + '\n')


def extract_paragraphs(docx_path):
    """
    从给定的docx文件中提取段落。

    参数:
    docx_path (str): docx文件的路径。

    返回:
    list: 包含文档中所有段落的列表，不包含空行。
    """
    # 加载文档
    doc = Document(docx_path)
    # 初始化段落列表
    paragraphs = []

    # 遍历文档中的每个段落
    for paragraph in doc.paragraphs:
        # 提取段落文本（自动过滤空行）
        text = paragraph.text.strip()
        # 仅保留非空内容
        if text:
            paragraphs.append(text)

    # 返回提取的段落列表
    return paragraphs


def read_table_content(table, indent=0):
    """
    递归读取表格内容，包括合并单元格和嵌套表格。

    :param table: 要读取的表格对象。
    :param indent: 缩进级别，用于处理嵌套表格。
    """
    rows = []
    for i, row in enumerate(table.rows):
        cols = []
        for j, cell in enumerate(row.cells):
            # 检查单元格是否是合并单元格的一部分

            paragraphs = [para.text.strip() for para in cell.paragraphs if para.text.strip()]
            paragraphs_text = "=".join(paragraphs)
            cell_text = f"{' ' * indent}{paragraphs_text}"

            # # 检查单元格内是否有嵌套表格
            # nested_tables = cell.tables
            # if nested_tables:
            #     # todo 程序没有执行到这里，处理嵌套表格的情况没有被识别
            #     cell_text += "\n"
            #     for nested_table in nested_tables:
            #         cell_text += "Nested Table:\n"
            #         cell_text += read_table_content(nested_table, indent + 4)

            cols.append(cell_text)
        rows.append(cols)

    return "\n".join(["".join(row) for row in rows])


def extract_all_tables(doc):
    """
    提取文档中的所有表格内容，并打印出来。

    :param doc: 文档对象。
    """
    tables = doc.tables
    for idx, table in enumerate(tables):
        print(f"Table {idx + 1}:")
        print(read_table_content(table))
        print("\n" + "-" * 80 + "\n")


def vocabulary():
    # 主程序入口
    # 调用函数读取指定路径的 .docx 文件，并获取分栏后的文本内容
    columns = read_vocabulary_from_docx("/Users/deipss/Desktop/LZTD/供电专业课程国际化翻译/铁路中英文词汇（全）.docx")
    # 遍历每一栏的内容并打印
    # for idx, column_texts in enumerate(columns):
    #     print(f"Column {idx + 1}:")  # 打印栏号
    #     for line in column_texts:
    #         print(line)  # 打印栏中的每一行文本
    # print(f'total colums is ', {len(columns)})


if __name__ == '__main__':
    # paragraphs =extract_paragraphs('/Users/deipss/Desktop/LZTD/供电专业课程国际化翻译/test1.docx')
    # for id,paragraph in enumerate(paragraphs[3:]):
    #     print(f'{id} : {paragraph}')

    vocabulary()
