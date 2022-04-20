from PIL import Image
import argparse
import shutil
from pathlib import Path
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imgdir", help="input image directory path")
    parser.add_argument("cleanratio", help="input x out of 100 to remove images with > x% white pixels ")

    args = parser.parse_args()

    img_dir = Path(args.imgdir)
    cleanratio = args.cleanratio
    for img in img_dir.glob("*.png"):
        im = Image.open(img)
        pix = im.load()
        whitePixelCount = 0
        for x in range(1024):
            for y in range(1024):
                sumColorVector = pix[x,y][0] + pix[x,y][1] + pix[x,y][2]
                if sumColorVector > 680:   # 220 + 220 + 220 = 680
                    whitePixelCount += 1
        idx = int(img.stem)
        # print(str(idx) + ' ' + str(whitePixelCount))
        if (whitePixelCount > int(1024*1024/100*int(cleanratio))):
            print('removing: ' + str(idx))
            os.remove(img)

    exit()