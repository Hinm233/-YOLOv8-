
from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    # ========== 只改这2行！！！ ==========
    model_path = "D:\\pytorch_std\\dataset_image\\runs\detect\\runs\\train\\action_train7\\weights\\best.pt"  # 你的模型路径
    test_img_path = "D:\\WXWork\\1688857976479456\\Cache\File\\2026-01\\法语_旅游学院_第33组\\frame_000504.jpg"  # 你的测试图片路径

    # 加载模型
    model = YOLO(model_path)
    results = model.predict(
        source=test_img_path,
        device=0,
        conf=0.5,
        save=True,
        show=False,
        save_dir="runs/detect/predict",

        hide_conf=True
    )
    print("结果在：D:\\pytorch_std\\dataset_image\\runs\\detect\\predict")