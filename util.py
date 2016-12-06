import math

def square(x):
    return x*x


def tobits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def frombits(bits):
    chars = []
    for b in range(len(bits) / 8):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)

def invert_bits(bits):
	return [1 if x == 0 else 0 for x in bits]

def bits_to_int(bits):
    result = 0
    for i in range(len(bits)):
        result += pow(2,i) if bits[len(bits) - 1 - i] == 1 else 0
    return result

def bin2gray(bits):
    return bits[:1] + [i ^ ishift for i, ishift in zip(bits[:-1], bits[1:])]

def gray2bin(bits):
	b = [bits[0]]
	for nextb in bits[1:]: b.append(b[-1] ^ nextb)
	return b

def z_scan(n):
    if n == 0:
        return [[0,0]]
    elif n > 0:
        left_upper = z_scan(n-1)
        added_len = int(math.sqrt(len(left_upper)))
        right_upper = []
        for i in range(len(left_upper)):
            right_upper.append([left_upper[i][0]+added_len,left_upper[i][1]])
        left_down = []
        for i in range(len(left_upper)):
            left_down.append([left_upper[i][0],left_upper[i][1]+added_len])
        right_down = []
        for i in range(len(left_upper)):
            right_down.append([left_upper[i][0]+added_len,left_upper[i][1]+added_len])
        result = left_upper + right_upper + left_down + right_down
        return result

def hilbert_scan(n):
    return hilbert(0,0,pow(2,n),0,0,pow(2,n),n)

def hilbert(x0, y0, xi, xj, yi, yj, n):
    if n <= 0:
        X = x0 + (xi + yi)/2
        Y = y0 + (xj + yj)/2
        
        # Output the coordinates of the cv
        return [[Y,X]]
    else:
        result = []
        result += hilbert(x0,               y0,               yi/2, yj/2, xi/2, xj/2, n - 1)
        result += hilbert(x0 + xi/2,        y0 + xj/2,        xi/2, xj/2, yi/2, yj/2, n - 1)
        result += hilbert(x0 + xi/2 + yi/2, y0 + xj/2 + yj/2, xi/2, xj/2, yi/2, yj/2, n - 1)
        result += hilbert(x0 + xi/2 + yi,   y0 + xj/2 + yj,  -yi/2,-yj/2,-xi/2,-xj/2, n - 1)  
        return result

def Zigzag_scan(n):
    x = 0
    y = 0
    k = 0
    direction = 1
    result = []
    while k <= n:
        for i in range(k+1):
            result.append([y,x])
            if direction == 1 and i != k:
                x += 1
                y -= 1
            elif direction == -1 and i != k:
                x -= 1
                y += 1
        if direction == 1 and k != n:
            direction = -1
            x += 1
        elif direction == -1 and k != n:
            direction = 1
            y += 1
        k += 1
    k -= 1
    if direction == 1:
        y += 1
        direction = -1
    else:
        x += 1
        direction = 1
    for j in range(k):
        for i in range(k,0,-1):
            result.append([y,x])
            if direction == 1 and i != 1:
                x += 1
                y -= 1
            elif direction == -1 and i != 1:
                x -= 1
                y += 1
        if direction == 1:
            direction = -1
            y += 1
        else:
            direction = 1
            x += 1
        k -= 1
    return result

def Moore_scan(img):
    pass
