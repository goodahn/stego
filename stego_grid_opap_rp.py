import cv2
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import aes
from optparse import OptionParser
import random
import util
import copy
import numpy as np
import math
import detect

parser = OptionParser()
MESSAGE_LENGTH = -1
BLOCK_NUMBER = -1
EMBEDDING_BIT = 4

def parser_load():
    global parser
    parser.add_option("-e","--embed"
                      ,action="store_true")
    parser.add_option("-d","--detect"
                      ,action="store_false")
    parser.add_option("-k","--key")
    parser.add_option("-m","--message")
    parser.add_option("-i","--image")
    parser.add_option("-b","--block")
    parser.add_option("-s","--secret")
    return parser.parse_args()

def do_stego(option_parsed):
    global BLOCK_NUMBER
    global MESSAGE_LENGTH
    option_parsed = option_parsed[0]
    embed = option_parsed.embed
    image_path = option_parsed.image
    key = option_parsed.key
    BLOCK_NUMBER = int(option_parsed.block)
    if embed:
        message = option_parsed.message
        embed_message(image_path, key, message)
    else:
        secret = option_parsed.secret
        detect_message(image_path, key, secret)

def embed_message(image_path, key, message):
    global BLOCK_NUMBER
    global MESSAGE_LENGTH
    global EMBEDDING_BIT
    aes_cipher = aes.AESCipher(key)
    encrypted = aes_cipher.encrypt(message)
    img = cv2.imread(image_path)

    h, w, d = img.shape
    BLOCK_HEIGHT = h//BLOCK_NUMBER
    BLOCK_WIDTH = w//BLOCK_NUMBER

    s_imgs = []
    KEYS = []

    ## split blocks
    for i in range(BLOCK_NUMBER):
        for j in range(BLOCK_NUMBER):
            s_imgs.append(img[i*BLOCK_HEIGHT:(i+1)*BLOCK_HEIGHT, j*BLOCK_WIDTH:(j+1)*BLOCK_WIDTH])

    ## make bits for embedding
    message_bits = util.tobits(encrypted)
    inverted_message_bits = util.invert_bits(util.tobits(encrypted))
    gray_message_bits = util.bin2gray(util.tobits(encrypted))
    inverted_gray_message_bits = util.invert_bits(util.bin2gray(util.tobits(encrypted)))

    output = []
    idx = 0
    MESSAGE_LENGTH = len(message_bits)
    r_num = random.sample(range(0,BLOCK_NUMBER*BLOCK_NUMBER),BLOCK_NUMBER*BLOCK_NUMBER)
    r_num_path = [random.sample(range(0,BLOCK_WIDTH * BLOCK_HEIGHT),BLOCK_WIDTH*BLOCK_HEIGHT) for x in range(BLOCK_NUMBER*BLOCK_NUMBER)]

    while len(message_bits) > 0 and idx < len(r_num):
        output.append([])
        idx_prng = r_num[idx]
        path = r_num_path[idx]

        ## message : compute MSE
        s = copy.deepcopy(s_imgs[idx_prng])
        MSE = 0
        tmp_message = message_bits
        for p in path:
            i = p//BLOCK_WIDTH
            j = p%BLOCK_WIDTH
            if len(tmp_message) == 0:
                break
            tmp = int(s[i, j, 0])
            rb = [int(x) for x in bin(s[i, j, 0])[2:].zfill(8)]
            eb = tmp_message[:EMBEDDING_BIT]
            tmp_message = tmp_message[EMBEDDING_BIT:]
            for k in range(len(eb)):
                rb[-(len(eb)-k)] = eb[k]
            s[i, j, 0] = np.uint8(util.bits_to_int(rb))
            opap = int(s[i, j, 0]) - tmp
            if opap > pow(2, EMBEDDING_BIT - 1) and opap < pow(2, EMBEDDING_BIT):
                if s[i, j, 0] >= pow(2, EMBEDDING_BIT):
                    s[i, j, 0] -= pow(2, EMBEDDING_BIT)
            elif opap > -pow(2, EMBEDDING_BIT) and opap < -pow(2, EMBEDDING_BIT - 1):
                if s[i, j, 0] < 256 - pow(2, EMBEDDING_BIT):
                    s[i, j, 0] += pow(2, EMBEDDING_BIT)
            tmp = max(s[i, j,0], tmp) - min(s[i, j, 0], tmp)
            MSE += util.square(tmp)
        MSE /= float(BLOCK_WIDTH) * BLOCK_HEIGHT
        output[-1].append(s)

        ## inverted message : compute MSE
        s = copy.deepcopy(s_imgs[idx_prng])
        invert_MSE = 0
        tmp_invert_message = inverted_message_bits
        for p in path:
            i = p//BLOCK_WIDTH
            j = p%BLOCK_WIDTH
            if len(tmp_invert_message) == 0:
                break
            tmp = int(s[i, j, 0])
            rb = [int(x) for x in bin(s[i, j, 0])[2:].zfill(8)]
            eb = tmp_invert_message[:EMBEDDING_BIT]
            tmp_invert_message = tmp_invert_message[EMBEDDING_BIT:]
            for k in range(len(eb)):
                rb[-(len(eb)-k)] = eb[k]
            s[i, j, 0] = np.uint8(util.bits_to_int(rb))
            opap = int(s[i, j, 0]) - tmp
            if opap > pow(2, EMBEDDING_BIT - 1) and opap < pow(2, EMBEDDING_BIT):
                if s[i, j, 0] >= pow(2, EMBEDDING_BIT):
                    s[i, j, 0] -= pow(2, EMBEDDING_BIT)
            elif opap > -pow(2, EMBEDDING_BIT) and opap < -pow(2, EMBEDDING_BIT - 1):
                if s[i, j, 0] < 256 - pow(2, EMBEDDING_BIT):
                    s[i, j, 0] += pow(2, EMBEDDING_BIT)
            tmp = max(s[i, j,0], tmp) - min(s[i, j, 0], tmp)
            invert_MSE += util.square(tmp)
        invert_MSE /= float(BLOCK_WIDTH) * BLOCK_HEIGHT
        output[-1].append(s)

        ## gray message : compute MSE
        s = copy.deepcopy(s_imgs[idx_prng])
        gray_MSE = 0
        tmp_gray_message = gray_message_bits
        for p in path:
            i = p//BLOCK_WIDTH
            j = p%BLOCK_WIDTH
            if len(tmp_gray_message) == 0:
                break
            tmp = int(s[i, j, 0])
            rb = [int(x) for x in bin(s[i, j, 0])[2:].zfill(8)]
            eb = tmp_gray_message[:EMBEDDING_BIT]
            tmp_gray_message = tmp_gray_message[EMBEDDING_BIT:]
            for k in range(len(eb)):
                rb[-(len(eb)-k)] = eb[k]
            s[i, j, 0] = np.uint8(util.bits_to_int(rb))
            opap = int(s[i, j, 0]) - tmp
            if opap > pow(2, EMBEDDING_BIT - 1) and opap < pow(2, EMBEDDING_BIT):
                if s[i, j, 0] >= pow(2, EMBEDDING_BIT):
                    s[i, j, 0] -= pow(2, EMBEDDING_BIT)
            elif opap > -pow(2, EMBEDDING_BIT) and opap < -pow(2, EMBEDDING_BIT - 1):
                if s[i, j, 0] < 256 - pow(2, EMBEDDING_BIT):
                    s[i, j, 0] += pow(2, EMBEDDING_BIT)
            tmp = max(s[i, j,0], tmp) - min(s[i, j, 0], tmp)
            gray_MSE += util.square(tmp)
        gray_MSE /= float(BLOCK_WIDTH) * BLOCK_HEIGHT
        output[-1].append(s)

        ## inverted gray message : compute MSE
        s = copy.deepcopy(s_imgs[idx_prng])
        invert_gray_MSE = 0
        tmp_invert_gray_message = inverted_gray_message_bits
        for p in path:
            i = p//BLOCK_WIDTH
            j = p%BLOCK_WIDTH
            if len(tmp_invert_gray_message) == 0:
                break
            tmp = int(s[i, j, 0])
            rb = [int(x) for x in bin(s[i, j, 0])[2:].zfill(8)]
            eb = tmp_invert_gray_message[:EMBEDDING_BIT]
            tmp_invert_gray_message = tmp_invert_gray_message[EMBEDDING_BIT:]
            for k in range(len(eb)):
                rb[-(len(eb)-k)] = eb[k]
            s[i,j,0] = np.uint8(util.bits_to_int(rb))
            opap = int(s[i, j, 0]) - tmp
            if opap > pow(2, EMBEDDING_BIT - 1) and opap < pow(2, EMBEDDING_BIT):
                if s[i, j, 0] >= pow(2, EMBEDDING_BIT):
                    s[i, j, 0] -= pow(2, EMBEDDING_BIT)
            elif opap > -pow(2, EMBEDDING_BIT) and opap < -pow(2, EMBEDDING_BIT - 1):
                if s[i, j, 0] < 256 - pow(2, EMBEDDING_BIT):
                    s[i, j, 0] += pow(2, EMBEDDING_BIT)
            tmp = max(s[i, j,0], tmp) - min(s[i, j, 0], tmp)
            invert_gray_MSE += util.square(tmp)
        invert_gray_MSE /= float(BLOCK_WIDTH) * BLOCK_HEIGHT
        output[-1].append(s)

        ## make stego and KEYS
        if MSE == min(MSE, invert_MSE, gray_MSE, invert_gray_MSE):
            s_imgs[idx_prng] = output[-1][0]
            KEYS.append("00")
        elif invert_MSE == min(MSE, invert_MSE, gray_MSE, invert_gray_MSE):
            s_imgs[idx_prng] = output[-1][1]
            KEYS.append("01")
        elif gray_MSE == min(MSE, invert_MSE, gray_MSE, invert_gray_MSE):
            s_imgs[idx_prng] = output[-1][2]
            KEYS.append("10")
        elif invert_gray_MSE == min(MSE, invert_MSE, gray_MSE, invert_gray_MSE):
            s_imgs[idx_prng] = output[-1][3]
            KEYS.append("11")

        ## cut message bits
        message_bits = tmp_message
        inverted_message_bits = tmp_invert_message
        gray_message_bits = tmp_gray_message
        inverted_gray_message_bits = tmp_invert_gray_message
        idx += 1

    result = [[] for x in range(BLOCK_NUMBER)]
    result_border = [[] for x in range(BLOCK_NUMBER)]
    bordersize = 2
    for i in range(BLOCK_NUMBER):
        for j in range(BLOCK_NUMBER):
            if len(result[i]) == 0:
                result[i].append(s_imgs[i*BLOCK_NUMBER+j])
            else:
                result[i][0] = np.hstack((result[i][0],s_imgs[i*BLOCK_NUMBER+j]))
            if len(result_border[i]) == 0:
                if i*BLOCK_NUMBER+j in r_num[:len(KEYS)]:
                    s_imgs[i*BLOCK_NUMBER+j] = cv2.copyMakeBorder(s_imgs[i*BLOCK_NUMBER+j][bordersize:BLOCK_HEIGHT-bordersize,bordersize:BLOCK_WIDTH-bordersize], top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,0,0] )
                result_border[i].append(s_imgs[i*BLOCK_NUMBER+j])
            else:
                if i*BLOCK_NUMBER+j in r_num[:len(KEYS)]:
                    s_imgs[i*BLOCK_NUMBER+j] = cv2.copyMakeBorder(s_imgs[i*BLOCK_NUMBER+j][bordersize:BLOCK_HEIGHT-bordersize,bordersize:BLOCK_WIDTH-bordersize], top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,0,0] )
                result_border[i][0] = np.hstack((result_border[i][0],s_imgs[i*BLOCK_NUMBER+j]))

    img = np.array([result[j][0][i,0:w] for j in range(BLOCK_NUMBER) for i in range(h/BLOCK_NUMBER)])
    img2 = np.array([result_border[j][0][i,0:w] for j in range(BLOCK_NUMBER) for i in range(h/BLOCK_NUMBER)])

    print "original message: ", message
    print "embedded message: ", encrypted

    cv2.imwrite("stego_"+image_path[:-4]+".bmp",img.copy())
    cv2.imwrite("stego_"+image_path[:-4]+"_border.bmp",img2)
    
    detect.make_only_lsb("stego_"+image_path[:-4]+".bmp")
    detect.make_only_lsb("stego_"+image_path[:-4]+"_border.bmp")

    ## save PRNG and KEYS
    f = open("secret","w")
    for i in range(len(KEYS)):
        f.write(str(r_num[i])+" ")
    f.write("\n")
    for i in range(len(KEYS)):
        f.write(KEYS[i]+ " ")
    f.write("\n")
    for i in range(len(KEYS)):
        for j in range(len(r_num_path[i])):
            f.write(str(r_num_path[i][j])+" ")
        f.write("\n")
    f.close()
    return img

def detect_message(image_path, key, secret):
    global BLOCK_NUMBER
    global MESSAGE_LENGTH
    global EMBEDDING_BIT

    f = open(secret, "r")
    line = f.readline()
    r_num = [int(x) for x in line.split()]
    line = f.readline()
    mse_key = line.split()
    r_num_path = []
    while line != "":
        line = f.readline()
        r_num_path.append([int(x) for x in line.split()])
    f.close()

    aes_cipher = aes.AESCipher(key)
    img = cv2.imread(image_path)

    h, w, d = img.shape
    BLOCK_HEIGHT = h/BLOCK_NUMBER
    BLOCK_WIDTH = w/BLOCK_NUMBER

    s_imgs = []

    ## split blocks
    for i in range(BLOCK_NUMBER):
        for j in range(BLOCK_NUMBER):
            s_imgs.append(img[i*BLOCK_HEIGHT:(i+1)*BLOCK_HEIGHT, j*BLOCK_WIDTH:(j+1)*BLOCK_WIDTH])

    idx = 0
    encoded_message = []

    while idx < len(r_num) and idx <len(mse_key):
        k = mse_key[idx]
        s = s_imgs[r_num[idx]]
        path = r_num_path[idx]

        if k == "01":
            for p in path:
                i = p//BLOCK_WIDTH
                j = p%BLOCK_WIDTH
                bits = [int(x) for x in bin(s[i,j,0])[2:].zfill(8)][-EMBEDDING_BIT:]
                bits = util.invert_bits(bits)
                encoded_message += bits[-EMBEDDING_BIT:]
        elif k == "10":
            for p in path:
                i = p//BLOCK_WIDTH
                j = p%BLOCK_WIDTH
                bits = [int(x) for x in bin(s[i,j,0])[2:].zfill(8)][-EMBEDDING_BIT:]
                bits = util.gray2bin(util.bin2gray(encoded_message) + bits)
                encoded_message += bits[-EMBEDDING_BIT:]
        elif k == "11":
            for p in path:
                i = p//BLOCK_WIDTH
                j = p%BLOCK_WIDTH
                bits = [int(x) for x in bin(s[i,j,0])[2:].zfill(8)][-EMBEDDING_BIT:]
                bits = util.gray2bin(util.invert_bits(util.bin2gray(encoded_message)+bits))
                encoded_message += bits[-EMBEDDING_BIT:]
        elif k == "00":
            for p in path:
                i = p//BLOCK_WIDTH
                j = p%BLOCK_WIDTH
                bits = [int(x) for x in bin(s[i,j,0])[2:].zfill(8)]
                encoded_message += bits[-EMBEDDING_BIT:]
        else:
            print "what happen?"
            exit()

        idx += 1

    encoded_message = util.frombits(encoded_message)
    print encoded_message
    print aes_cipher.decrypt(encoded_message)
    return r_num_path



if __name__ == "__main__":
    parsed = parser_load()
    do_stego(parsed)
