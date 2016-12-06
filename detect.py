import cv2
import sys

def make_only_lsb(filename):
    img = cv2.imread(filename)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i,j,0] == 255 and img[i,j,1] == 0 and img[i,j,2] == 0:
                img[i,j,0] = 0
                img[i,j,0] = 255
                img[i,j,0] = 0
            else:
                t = img[i,j,0] & 1
                img[i,j,0] = t* 255
                img[i,j,1] = 0 
                img[i,j,2] = 0

    cv2.imwrite("lsb_"+filename[:-4]+".bmp", img)

def make_only_lsb_border(filename):
    img = cv2.imread(filename)
    f = open("secret","r")
    f.readline()
    pass

if __name__ == "__main__":
    make_only_lsb(sys.argv[1])
