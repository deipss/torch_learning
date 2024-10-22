import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def plt_show(df):
    # 绘制线条
    plt.figure(figsize=(16, 6))  # 增加图形宽度 width height
    plt.plot(df['year'], df['born'], label='出生人口')
    plt.plot(df['year'], df['NCEE'], label='高考人数')
    plt.plot(df['year'], df['graduated'], label='高校毕业')
    plt.plot(df['year'], df['master_examination'], label='硕士入学考试')
    plt.plot(df['year'], df['master_reading'], label='在读硕士')
    plt.plot(df['year'], df['master'], label='毕业硕士')
    plt.plot(df['year'], df['dead'], label='死亡')
    # 设置图例
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # 设置标题和坐标轴标签
    plt.title('人口趋势折线图')

    plt.axvspan(2025, 2056, color='green', alpha=0.2)
    plt.xlabel('year')
    plt.ylabel('population')
    plt.xticks(df['year'], rotation=75)
    plt.axvline(x=2016, color='g', linestyle='--', linewidth=1)  # 在x=2的位置添加一条红色虚线
    plt.axvline(x=2016-18, color='g', linestyle='--', linewidth=1)  # 在x=2的位置添加一条红色虚线
    plt.axvline(x=2022, color='b', linestyle='--', linewidth=1)  # 在x=2的位置添加一条红色虚线



    born_mean = top_year['born'].mean()
    ncee_mean = top_year['NCEE'].mean()

    plt.axhline(born_mean, color='r', linestyle='--', linewidth=2, label=f'born_mean')
    plt.axhline(ncee_mean, color='r', linestyle='--', linewidth=2, label=f'ncee_mean')

    # 显示图表
    plt.show()



def plt_show_statistic(df):
    # 绘制线条
    plt.figure(figsize=(12, 6))  # 增加图形宽度 width height
    plt.plot(df['year'], df['born'], label='出生人口')
    plt.plot(df['year'], df['NCEE'], label='高考人数')
    # 设置图例
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # 设置标题和坐标轴标签
    plt.title('人口趋势折线图')

    plt.axvspan(2025, 2056, color='green', alpha=0.2)
    plt.xlabel('year')
    plt.ylabel('population')
    plt.xticks(df['year'], rotation=75)



    born_mean = top_year['born'].mean()
    ncee_mean = top_year['NCEE'].mean()

    plt.axhline(born_mean, color='r', linestyle='--', linewidth=1, label='born_mean')
    plt.axhline(ncee_mean, color='r', linestyle='--', linewidth=1, label='ncee_mean')

    # 显示图表
    plt.show()

if __name__ == '__main__':


    
    matplotlib.rc("font", family='Heiti TC')
    df = pd.read_excel('data/under_graduate.xlsx')
    print(df.head())

    born = df[['year', 'born']]

    print(born)

    # 下标从0开始，打印第1行的数据
    print(df.xs(1))

    top_year  = df[df['year'] < 2023]

    print(top_year['born'].mean())
    print(top_year['NCEE'].mean())

    plt_show_statistic(top_year)
