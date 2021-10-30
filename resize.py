from datasets.ISIC2018 import *
from PIL import Image
from misc_utils.visualization_utils import BatchVisualization
from skimage import io, transform
import cv2

if __name__ == '__main__':
    import os
    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 指定要使用的GPU序号

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)


    images = load_task3_training_images(output_size=None)
    image_ids = task3_image_ids
    image_masks = load_task3_training_mask_images
    # images = 'E:\code\ISIC2018-master\ISIC2018-master\datasets\ISIC2018\data'

    # images = load_task3_test_images(output_size=None)
    # image_ids = task3_test_image_ids
    # image_masks = load_task3_test_mask_images

    task3_resized_img = 'ISIC2018_Task3_Training_Input_Resize'

    # task3_resized_img = 'ISIC2018_Task3_Test_Input_Resize'

    output_dir = os.path.join(data_dir, task3_resized_img)

    # print(images[0])

    def isPos(num):
        if num >= 0:
            return num
        else:
            return 0

    def PixelCorrect(num, bon):
        if num < bon:
            return num
        else:
            return bon-1


    maskList = image_masks()
    pixel_blank = 45

    for i_image, i_name in enumerate(image_ids):
    # for i_image, i_name, i_masks in zip(enumerate(images), image_masks):

        # print('***', xxx[index])
        # print('---------', maskList[index].shape)

        i_masks = maskList[i_image]

        # up
        for row in range(i_masks.shape[0]):
            for col in range(i_masks.shape[1]):
                if i_masks[row, col] == 255:
                    row1 = row
                    col1 = col
                    break
            else:
                continue
            break

        # left
        for col in range(i_masks.shape[1]):
            for row in range(i_masks.shape[0]):
                if i_masks[row, col] == 255:
                    row2 = row
                    col2 = col
                    break
            else:
                continue
            break

        # down
        for row in range(i_masks.shape[0], 0, -1):
            for col in range(i_masks.shape[1]):
                # print(row, col)
                if i_masks[row-1, col] == 255:
                    row3 = row
                    col3 = col
                    break
            else:
                continue
            break

        # right
        for col in range(i_masks.shape[1], 0, -1):
            for row in range(i_masks.shape[0]):
                if i_masks[row, col-1] == 255:
                    row4 = row
                    col4 = col
                    break
            else:
                continue
            break

        seg_image = images[i_image][isPos(row1-55):PixelCorrect(row3+55, images.shape[1]), isPos(col2 - 55):PixelCorrect(col4+55, images.shape[2]), :]

        # if ((row3 - row1 + 60) / images.shape[1]) < ((col4 - col2 + 60) / images.shape[2]):
        #     rate = (col4 - col2 + 60) / images.shape[2]
        #     col_len = rate * images.shape[1]
        #     pixel = int((col_len - row3 + row1) // 2)
        #     row1 = row1 - pixel
        #     row3 = row3 + pixel
        #     col2 = col2 - 30
        #     col4 = col4 + 30
        # else:
        #     rate = (row3 - row1 + 60) / images.shape[1]
        #     row_len = rate * images.shape[2]
        #     pixel = int((row_len - col4 + col2) // 2)
        #     row1 = row1 - 30
        #     row3 = row3 + 30
        #     col2 = col2 - pixel
        #     col4 = col4 + pixel
        # seg_image = images[i_image][isPos(row1):PixelCorrect(row3, images.shape[1]), isPos(col2):PixelCorrect(col4, images.shape[2]), :]

        # print(type(seg_image))
        # seg_image = seg_image.resize((224, 224), Image.BILINEAR)
        # seg_image = cv2.resize(seg_image, (224, 224), interpolation=cv2.INTER_AREA)

        # seg_image = seg_image.resize((224, 224))

        # im = seg_image * 255
        # im = Image.fromarray(im.astype(np.uint8))
        # im = im.resize((224, 224), Image.BILINEAR)
        # print(im)
        # im.save(output_dir + '/' + i_name + '_resized.jpg')
        io.imsave(output_dir + '/' + i_name + '_resized.jpg', seg_image)


