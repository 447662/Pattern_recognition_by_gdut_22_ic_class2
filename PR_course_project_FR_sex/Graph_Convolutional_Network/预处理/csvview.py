import pandas as pd
import csv

def load_and_display_csv(csv_file, num_rows=5):
    """
    加载 CSV 文件并显示前几行数据、数据形状、列名和统计信息。

    参数:
    csv_file (str): CSV 文件路径。
    num_rows (int): 要显示的行数。
    """
    # 加载 CSV 文件
    df = pd.read_csv(csv_file)

    # 查看前 num_rows 行数据
    print(f"前 {num_rows} 行数据:")
    print(df.head(num_rows))

    # 查看数据的形状 (行数, 列数)
    print("数据形状:", df.shape)

    # 查看列名
    print("列名:", df.columns)

    # 查看统计信息
    print("统计信息:")
    print(df.describe())

    input('随意按键继续')

def print_csv_rows(csv_file, num_rows=None):
    """
    打印 CSV 文件的前几行。

    参数:
    csv_file (str): CSV 文件路径。
    num_rows (int): 要打印的行数。如果为 None，则打印所有行。
    """
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)

        # 打印每一行
        for i, row in enumerate(reader):
            print(row)
            if num_rows is not None and i + 1 >= num_rows:
                break

if __name__ == "__main__":
    # 定义 CSV 文件路径
    csv_file = "D:/Graph_Convolutional_Network/datas/facecsv/face.csv"

    # 加载并显示 CSV 文件信息
    load_and_display_csv(csv_file, num_rows=5)

    # 打印 CSV 文件的前 10 行
    print_csv_rows(csv_file, num_rows=10)