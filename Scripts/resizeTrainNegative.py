import argparse
import cv2
import os
'''
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
    help="path of images to be resized")
args = vars(ap.parse_args())
'''
for f in os.listdir('/content/drive/My Drive/Colab Notebooks/EyeDetection/ConvNet/DiabeticRetinopathy/DRImages/Training/Negative'):
    "resized_" + f.replace(".jpg", ".png").replace(".jpeg", ".png").replace(".tif", ".png")
    im = cv2.imread(os.path.join('/content/drive/My Drive/Colab Notebooks/EyeDetection/ConvNet/DiabeticRetinopathy/DRImages/Training/Negative', f))
    resized = cv2.resize(im, (299, 299))
    filepath_wo_extension = f.split(".")[0]
    cv2.imwrite(os.path.join('/content/drive/My Drive/Colab Notebooks/EyeDetection/ConvNet/DiabeticRetinopathy/DRImages/Training/Negative',filepath_wo_extension+'.png',resized))