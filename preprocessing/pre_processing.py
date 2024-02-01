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

imgs = np.array([image,image])
filtered_img = noise_removal(imgs, 2, (21,21),0.2)


    
# Mody
def sharpening():
    #TODO
