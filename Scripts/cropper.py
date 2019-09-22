import argparse
import os
import cv2

parser = argparse.ArgumentParser(description="Custom cropper program")
parser.add_argument("indir", nargs="?", type=str, default="test", help="Input dir for the images")
parser.add_argument("outdir", nargs="?", type=str, default="cropped", help="Output dir for the cropped images")

args = parser.parse_args()

for f in os.listdir(args.indir):
    im = cv2.imread(os.path.join(args.indir, f))
    width = im.shape[1]
    height = im.shape[0]

    if width > height:
        cropped = im[:, int(width / 2 - height / 2):int(width / 2 + height / 2)]
    else:
        cropped = im[:, int(height / 2 - width / 2):int(height / 2 + width / 2)]
    resized = cv2.resize(cropped, (224, 224))
    filepath = os.path.join(args.outdir, f.split(".")[0] + ".png")
    cv2.imwrite(filepath, resized)
