import pandas as pd
if __name__ == '__main__':

    # 读取文件
    excel_file = pd.ExcelFile('/Users/deipss/Desktop/LZTD/a宁局:沿海2025届2+1/同步通讯录模版.xls')

    # 获取所有表名
    sheet_names = excel_file.sheet_names

    # 获取指定工作表中的数据
    df = excel_file.parse('导入模板')

    # 查看数据的基本信息
    print('数据基本信息：')
    df.info()

    # 查看数据集行数和列数
    rows, columns = df.shape

    if rows < 100 and columns < 20:
        # 短表数据（行数少于100且列数少于20）查看全量数据信息
        print('数据全部内容信息：')
        print(df.to_csv(sep='\t', na_rep='nan'))
    else:
        # 长表数据查看数据前几行信息
        print('数据前几行内容信息：')
        print(df.head().to_csv(sep='\t', na_rep='nan'))


    import vobject

    # 创建一个 VCF 文件并写入内容
    with open('/Users/deipss/Desktop/LZTD/a宁局:沿海2025届2+1/同步通讯录模版.vcf', 'w', encoding='utf-8') as vcf_file:
        for index, row in df.iterrows():
            vcard = vobject.vCard()

            # 添加姓名
            vcard.add('n')
            vcard.n.value = vobject.vcard.Name(family='', given=row['姓名'], additional='', prefix='', suffix='')
            vcard.add('fn')
            vcard.fn.value = row['姓名']

            # 添加电话号码
            vcard.add('tel')
            vcard.tel.value = str(row['工作手机'])
            vcard.tel.params['TYPE'] = ['WORK']

            # 添加分组信息（非标准 VCF 字段，不同软件支持程度不同）
            group_field = vcard.add('X-OPPO-GROUP')
            group_field.value = row['分组']
            # 添加分组信息（非标准 VCF 字段，不同软件支持程度不同）
            group_field = vcard.add('PRINTABLE')
            group_field.value = row['分组']

            # 写入文件
            vcf_file.write(vcard.serialize())