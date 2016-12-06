## Implement STEGANOGRAPHY

# USAGE

 - Embed

python stego.py -e -b BLOCK_NUM -k KEY -m MESSAGE -i IMAGE_FILE_PATH  

 - Decode

python stego.py -d -b BLOCK_NUM -k KEY -i STEGO_IMAGE_FILE_PATH  

# ALGORITHMS

 - stego_row.py
  
Size of block is (IMAGE_HEIGHT/BLOCK_NUM) X IMAGE_WIDTH.  
Message is encrypted AES.  
Embedded messages are 4 types.  
plain message, inverted message, gray coded message, inverted gray coded message  
Choose type of message by minimal MSE.  

 - stego_grid.py
  
Size of block is (IMAGE_HEIGHT/BLOCK_NUM) X (IMAGE_WIDTH/BLOCK_NUM).  
Execpt size of block, everything is same of above.  

 - stego_grid_opap.py
  
Applying opap(optimized pixel adjustment process) on 2.  

 - stego_grid_opap_rp.py
  
Order of pixel in block is random.  

 - stego_rp.py

This algorithm is based on paper "An intelligent chaotic embedding approach to enhance stego-image quality"
