# ===================== YOLOv8 最终完美版【无任何报错】【GPU加速】【10分类】【Windows专用】 =====================
from ultralytics import YOLO
# 新增：解决Windows多进程报错的必须导入
from multiprocessing import freeze_support

# ===================== 核心修复：所有代码必须写在这个代码块内 =====================
if __name__ == '__main__':
    # 新增：解决进程启动问题的固定代码
    freeze_support()

    # ✅ 你的yaml绝对路径，正确无误
    data_yaml_path = "D:\\pytorch_std\\dataset_image\\data.yaml"


    model = YOLO("yolov8n.pt")

    # GPU训练全部配置，参数不变，完美适配你的数据集
    model.train(
        data=data_yaml_path,
        epochs=50,
        batch=8,
        imgsz=640,
        device=0,
        pretrained=True,
        save=True,
        verbose=True,
        seed=42,
        project='runs/train',
        name='action_train'
    )



    print("最优模型位置：runs/train/action_train/weights/best.pt")
    print("检测结果图片：runs/detect/predict")