import os
import cv2
import numpy as np
import mediapipe as mp

def initialize_face_mesh():
    """
    初始化 Mediapipe Face Mesh 模块。

    返回:
    FaceMesh: 初始化的 Face Mesh 对象。
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    return face_mesh

def process_images_to_npy(input_folder, output_folder):
    """
    处理输入文件夹中的图像，提取人脸关键点并保存为 .npy 文件。

    参数:
    input_folder (str): 输入图像文件夹路径。
    output_folder (str): 输出 .npy 文件夹路径。
    """
    os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在

    face_mesh = initialize_face_mesh()  # 初始化 Face Mesh

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".jpg"):
            image_path = os.path.join(input_folder, file_name)

            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法加载图像：{file_name}")
                continue

            # 转换为 RGB 格式
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 提取人脸关键点
            results = face_mesh.process(rgb_image)
            if not results.multi_face_landmarks:
                print(f"未检测到人脸：{file_name}")
                continue

            # 获取关键点的 3D 坐标 (x, y, z)
            for face_landmarks in results.multi_face_landmarks:
                keypoints = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]).T  # 转置为 3x478

                # 构造输出文件路径
                output_file_name = os.path.splitext(file_name)[0] + ".npy"
                output_file_path = os.path.join(output_folder, output_file_name)

                # 保存为 .npy 文件
                np.save(output_file_path, keypoints)
                print(f"保存成功：{output_file_path}")

    print("所有文件处理完成！")

if __name__ == "__main__":
    # 配置输入和输出路径
    input_folder = "D:/Graph_Convolutional_Network/output_jpg"
    output_folder = "D:/Graph_Convolutional_Network/datas/output_3npy"

    # 处理图像并生成 .npy 文件
    process_images_to_npy(input_folder, output_folder)