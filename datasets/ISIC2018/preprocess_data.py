if __name__ == '__main__':
    import os
    import sys
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(os.path.split(rootPath)[0])

    from datasets.ISIC2018 import load_training_data, load_test_data

    import os
    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定要使用的GPU序号

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)

    # _, _, _ = load_training_data(task_idx=1, output_size=224)
    # _, _, _ = load_training_data(task_idx=2, output_size=224)
    # _, _, _ = load_training_data(task_idx=3, output_size=224)

    # images, image_names, image_sizes = load_validation_data(task_idx=1, output_size=224)
    # images, image_names = load_validation_data(task_idx=3, output_size=224)

    images, image_names, image_sizes = load_test_data(task_idx=1, output_size=224)
    # images, image_names = load_test_data(task_idx=3, output_size=224)

