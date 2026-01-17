from ultralytics import YOLO

from multiprocessing import freeze_support


if __name__ == '__main__':
   
    freeze_support()

    # 路径
    data_yaml_path = "D:\\pytorch_std\\dataset_image\\data.yaml"


    model = YOLO("yolov8n.pt")

    
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
