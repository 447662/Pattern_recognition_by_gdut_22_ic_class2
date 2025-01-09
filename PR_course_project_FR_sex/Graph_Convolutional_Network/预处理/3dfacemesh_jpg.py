import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def initialize_face_mesh():
    """
    初始化 Mediapipe Face Mesh 模块。

    返回:
    FaceMesh: 初始化的 Face Mesh 对象。
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    return face_mesh

def load_image(image_path):
    """
    加载并转换图像为 RGB 格式。

    参数:
    image_path (str): 图像文件路径。

    返回:
    ndarray: 转换为 RGB 格式的图像。
    """
    image = cv2.imread(image_path)
    if image is None:
        print("路径错误，无法加载图像")
        exit()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image

def extract_face_landmarks(face_mesh, rgb_image):
    """
    使用 Mediapipe 提取人脸关键点。

    参数:
    face_mesh (FaceMesh): 初始化的 Face Mesh 对象。
    rgb_image (ndarray): RGB 格式的图像。

    返回:
    list: 提取到的人脸关键点列表。
    """
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks
    else:
        print("未检测到人脸。")
        return None

def plot_3d_face_mesh(face_landmarks, rgb_image):
    """
    绘制 3D 人脸关键点图。

    参数:
    face_landmarks (list): 人脸关键点列表。
    rgb_image (ndarray): RGB 格式的图像。
    """
    h, w, _ = rgb_image.shape
    keypoints_x = [landmark.x for landmark in face_landmarks.landmark]
    keypoints_y = [landmark.y for landmark in face_landmarks.landmark]
    keypoints_z = [landmark.z for landmark in face_landmarks.landmark]  # 相对深度值

    # 创建 3D 散点图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制关键点, c='green' 表示绿色
    ax.scatter(keypoints_x, keypoints_y, keypoints_z, c='green', s=10)

    # 设置轴标签
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Face Mesh")

    # 调整视角
    ax.view_init(elev=-85, azim=-90)

    # 显示图像
    plt.show()

if __name__ == "__main__":
    # 初始化 Face Mesh 模块
    face_mesh = initialize_face_mesh()

    # 读取并转换图像
    image_path = "D:/Graph_Convolutional_Network/output_jpg/1223.jpg"
    rgb_image = load_image(image_path)

    # 提取人脸关键点
    face_landmarks_list = extract_face_landmarks(face_mesh, rgb_image)

    # 检查是否检测到人脸并绘制 3D 图
    if face_landmarks_list:
        for face_landmarks in face_landmarks_list:
            plot_3d_face_mesh(face_landmarks, rgb_image)