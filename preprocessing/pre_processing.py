import cv2
import numpy as np

def image_resizing(images):
    img = cv2.imread('filename.jpg')

    # Resize the image to a specific size (e.g., 54x140 pixels)
    resized_imgs = cv2.resize(img, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)
    return resized_imgs



# Omar & Masry
def images_normalization(images):
    pass

# 
def data_splitting(data):
    pass


def data_augmentation(data):
    pass

# Karim & Walid
# Why removing image background is a bad idea => https://shorturl.at/dqrzG
def remove_image_background(images):
    pass

# Mustafa

def noise_removal(images, filter,kernal, standard = 1):
    """_summary_

    Args:
        images (narray): 4d array of all RGB images
        filter (int): median(0), mean(1), gausian(2)
        kernal (int/(int,int)): median->int, mean & gausian -> (int,int)
        standard (int): control the distributionn of gusian filter, higher value will get more bluring
    """
    result = []

    for img in images:

        if(filter == 0):
            result.append(cv2.medianBlur(img, kernal) )
        elif(filter == 1):
            result.append(cv2.blur(img,kernal) )
        elif(filter == 2):
            result.append(cv2.GaussianBlur(img, kernal, standard) )


    return result

#Temp
image = cv2.imread('00260_00.jpg') 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

imgs = np.array([image,image])
filtered_img = noise_removal(imgs, 2, (21,21),0.2)


    
# Mody
def read_image(path):
  image = cv2.imread(path)
  # OpenCV reads images in BGR format, so you might want to convert it to RGB
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image
def sharpen(image):
    # Define a 3x3 sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    # Apply the sharpening filter using the filter2D function from OpenCV
    # -1 as the ddepth parameter means the output image will have the same depth as the input image
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    # Return the sharpened image
    return image_sharp
def sharpen_laplacian(image):
  # Sharpen the image using the Laplacian operator
  # This ensures that the Laplacian operation's output is stored in a 64-bit... 
  # ...floating-point format for more accurate representation of the calculated values.
  sharpened_image = cv2.Laplacian(image, cv2.CV_64F)



# YCpCr color space VS RGB(Youtube Video explanation) => https://shorturl.at/hikX2
# Stackoverflow thread of how to use YCpCr color space for efficient histogram equalization => https://shorturl.at/lrw56
def histogram_equalization(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    print len(channels)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img
    
