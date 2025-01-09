import pandas as pd

def read_excel_file(file_path):
    """
    读取 Excel 文件并返回 DataFrame
    :param file_path: Excel 文件路径
    :return: DataFrame
    """
    return pd.read_excel(file_path)

def process_data(df):
    """
    处理数据并生成包含 'picture' 和 'label' 的 DataFrame
    :param df: 输入的 DataFrame
    :return: 结果 DataFrame
    """
    result_df = pd.DataFrame(columns=['picture', 'label'])
    
    for index, row in df.iterrows():
        content = row.iloc[0].lstrip()
        
        if '(_missing descriptor)' in content:
            continue
        
        picture = content[:4]
        
        if '(_sex  male)' in content:
            label = 1
        elif '(_sex  female)' in content:
            label = 0
        else:
            continue
        
        new_row = pd.DataFrame({'picture': [picture], 'label': [label]})
        result_df = pd.concat([result_df, new_row], ignore_index=True)
    
    return result_df

def save_to_csv(df, file_path):
    """
    将 DataFrame 保存为 CSV 文件
    :param df: 要保存的 DataFrame
    :param file_path: CSV 文件路径
    """
    df.to_csv(file_path, index=False)
    print(f"已将数据保存为 CSV 文件: {file_path}")

# 主程序
if __name__ == "__main__":
    excel_file_path = 'D:/Graph_Convolutional_Network/facedata/face.xlsx'
    csv_file_path = 'D:/Graph_Convolutional_Network/datas/facecsv/face.csv'
    
    df = read_excel_file(excel_file_path)
    print(df.head())
    
    result_df = process_data(df)
    save_to_csv(result_df, csv_file_path)