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
EMBEDDING_BIT = 1

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


    ## make path
    z_path= util.z_scan(int(math.log(w//BLOCK_NUMBER, 2)))
    hilbert_path = util.hilbert_scan(int(math.log(w//BLOCK_NUMBER, 2)))
    zigzag_path = util.Zigzag_scan(w//BLOCK_NUMBER-1)

    output = []
    idx = 0
    MESSAGE_LENGTH = len(message_bits)
    r_num = random.sample(range(0,BLOCK_NUMBER*BLOCK_NUMBER),BLOCK_NUMBER*BLOCK_NUMBER)
    while len(message_bits) > 0 and idx < len(r_num):
        output.append([])
        idx_prng = r_num[idx]

        ## message - z_path : compute MSE
        s = copy.deepcopy(s_imgs[idx_prng])
        z_MSE = 0
        tmp_message = copy.deepcopy(message_bits)
        for p in z_path:
            if len(tmp_message) == 0:
                break
            i = p[1]
            j = p[0]
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
            z_MSE += util.square(tmp)
        z_MSE /= float(BLOCK_WIDTH) * BLOCK_HEIGHT
        z_KEY = "000"
        output[-1].append(s)

        ## inverted message - z_path : compute MSE
        s = copy.deepcopy(s_imgs[idx_prng])
        z_invert_MSE = 0
        tmp_invert_message = copy.deepcopy(inverted_message_bits)
        for p in z_path:
            if len(tmp_invert_message) == 0:
                break
            i = p[1]
            j = p[0]
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
            z_invert_MSE += util.square(tmp)
        z_invert_MSE /= float(BLOCK_WIDTH) * BLOCK_HEIGHT

        if z_MSE > z_invert_MSE:
            z_MSE = z_invert_MSE
            z_KEY = "100"
            output[-1][-1] = s

        ## message - hilbert_path : compute MSE
        s = copy.deepcopy(s_imgs[idx_prng])
        hilbert_MSE = 0
        tmp_message = copy.deepcopy(message_bits)
        for p in hilbert_path:
            if len(tmp_message) == 0:
                break
            i = p[1]
            j = p[0]
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
            hilbert_MSE += util.square(tmp)
        hilbert_MSE /= float(BLOCK_WIDTH) * BLOCK_HEIGHT 
        hilbert_KEY = "001"
        output[-1].append(s)

        ## inverted message - hilbert_path : compute MSE
        s = copy.deepcopy(s_imgs[idx_prng])
        hilbert_invert_MSE = 0
        tmp_invert_message = copy.deepcopy(inverted_message_bits)
        for p in hilbert_path:
            if len(tmp_invert_message) == 0:
                break
            i = p[1]
            j = p[0]
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
            hilbert_invert_MSE += util.square(tmp)
        hilbert_invert_MSE /= float(BLOCK_WIDTH) * BLOCK_HEIGHT

        if hilbert_MSE > hilbert_invert_MSE:
            hilbert_MSE = hilbert_invert_MSE
            hilbert_KEY = "101"
            output[-1][-1] = s

        ## message - Zigzag_path : compute MSE
        s = copy.deepcopy(s_imgs[idx_prng])
        zigzag_MSE = 0
        tmp_message = copy.deepcopy(message_bits)
        for p in zigzag_path:
            if len(tmp_message) == 0:
                break
            i = p[1]
            j = p[0]
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
            zigzag_MSE += util.square(tmp)
        zigzag_MSE /= float(BLOCK_WIDTH) * BLOCK_HEIGHT 
        zigzag_KEY = "010"
        output[-1].append(s)

        ## inverted message - Zigzag_path : compute MSE
        s = copy.deepcopy(s_imgs[idx_prng])
        zigzag_invert_MSE = 0
        tmp_invert_message = copy.deepcopy(inverted_message_bits)
        for p in zigzag_path:
            if len(tmp_invert_message) == 0:
                break
            i = p[1]
            j = p[0]
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
            zigzag_invert_MSE += util.square(tmp)
        zigzag_invert_MSE /= float(BLOCK_WIDTH) * BLOCK_HEIGHT

        if zigzag_MSE > zigzag_invert_MSE:
            zigzag_MSE = zigzag_invert_MSE
            zigzag_KEY = "110"
            output[-1][-1] = s

        ## message - normal_path : compute MSE
        s = copy.deepcopy(s_imgs[idx_prng])
        MSE = 0
        tmp_message = copy.deepcopy(message_bits)
        for i in range(h//BLOCK_NUMBER):
            for j in range(w//BLOCK_NUMBER):
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
        KEY = "011"
        output[-1].append(s)

        ## inverted message - normal_path : compute MSE
        s = copy.deepcopy(s_imgs[idx_prng])
        invert_MSE = 0
        tmp_invert_message = copy.deepcopy(inverted_message_bits)
        for i in range(h//BLOCK_NUMBER):
            for j in range(w//BLOCK_NUMBER):
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

        if MSE > invert_MSE:
            MSE = invert_MSE
            KEY = "111"
            output[-1][-1] = s

        if len(output[-1]) != 4:
            print "length of output is not four? hmm..."
            exit()

        ## make stego and KEYS
        if z_MSE == min(z_MSE, hilbert_MSE, zigzag_MSE, MSE):
            s_imgs[idx_prng] = output[-1][0]
            KEYS.append(z_KEY)
        elif hilbert_MSE == min(MSE, hilbert_MSE, zigzag_MSE, MSE):
            s_imgs[idx_prng] = output[-1][1]
            KEYS.append(hilbert_KEY)
        elif zigzag_MSE == min(MSE, hilbert_MSE, zigzag_MSE, MSE):
            s_imgs[idx_prng] = output[-1][2]
            KEYS.append(zigzag_KEY)
        elif MSE == min(MSE, hilbert_MSE, zigzag_MSE, MSE):
            s_imgs[idx_prng] = output[-1][3]
            KEYS.append(KEY)

        ## cut message bits
        message_bits = tmp_message
        inverted_message_bits = tmp_invert_message
        
        if len(message_bits) != len(inverted_message_bits):
            print "messages are not same length? hmm..."
            exit()

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
    f.close()

    return img

def detect_message(image_path, key, secret):
    global BLOCK_NUMBER
    global MESSAGE_LENGTH
    global EMBEDDING_BIT

    ## read secret files
    f = open(secret, "r")
    line = f.readline()
    r_num = [int(x) for x in line.split()]
    line = f.readline()
    mse_key = line.split()
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

    ## make path
    z_path= util.z_scan(int(math.log(w//BLOCK_NUMBER, 2)))
    hilbert_path = util.hilbert_scan(int(math.log(w//BLOCK_NUMBER, 2)))
    zigzag_path = util.Zigzag_scan(w//BLOCK_NUMBER - 1)

    idx = 0
    encoded_message = []

    while idx < len(r_num) and idx <len(mse_key):
        k = mse_key[idx]
        s = s_imgs[r_num[idx]]

        if k[1:] == "01":
            for p in hilbert_path:
                i = p[1]
                j = p[0]
                bits = [int(x) for x in bin(s[i,j,0])[2:].zfill(8)][-EMBEDDING_BIT:]
                if k[0] == "1":
                    bits = util.invert_bits(bits)
                encoded_message += bits[-EMBEDDING_BIT:]
        elif k[1:] == "10":
            for p in zigzag_path:
                i = p[1]
                j = p[0]
                bits = [int(x) for x in bin(s[i,j,0])[2:].zfill(8)][-EMBEDDING_BIT:]
                if k[0] == "1":
                    bits = util.invert_bits(bits)
                encoded_message += bits[-EMBEDDING_BIT:]
        elif k[1:] == "11":
            for i in range(h//BLOCK_NUMBER):
                for j in range(w//BLOCK_NUMBER):
                    bits = [int(x) for x in bin(s[i,j,0])[2:].zfill(8)][-EMBEDDING_BIT:]
                    if k[0] == "1":
                        bits = util.invert_bits(bits)
                    encoded_message += bits[-EMBEDDING_BIT:]
        elif k[1:] == "00":
            for p in z_path:
                i = p[1]
                j = p[0]
                bits = [int(x) for x in bin(s[i,j,0])[2:].zfill(8)]
                if k[0] == "1":
                    bits = util.invert_bits(bits)
                encoded_message += bits[-EMBEDDING_BIT:]
        else:
            print "what happen?"
            exit()

        idx += 1

    encoded_message = util.frombits(encoded_message)
    print encoded_message
    print aes_cipher.decrypt(encoded_message)
    return img



if __name__ == "__main__":
    parsed = parser_load()
    do_stego(parsed)
