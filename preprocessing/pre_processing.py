import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torchvision import transforms
import math

def image_resizing(images):
    img = cv2.imread('filename.jpg')

    # Resize the image to a specific size (e.g., 54x140 pixels)
    resized_imgs = cv2.resize(img, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)
    return resized_imgs



# Omar & Masry
def images_normalization(images):
    # load the image
    img_path = 'Lion.jpg'
    img = Image.open(img_path)
 
    # define custom transform function
    transform = transforms.Compose([
    transforms.ToTensor()
    ])
 
    # transform the pIL image to tensor 
    # image
    img_tr = transform(img)
 
    # calculate mean and std
    mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
 
 
    # define custom transform
    # here we are using our calculated
    # mean & std
    transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])
 
    # get normalized image
    img_normalized = transform_norm(img)
 
    # convert this image to numpy array
    img_normalized = np.array(img_normalized)
 
    # transpose from shape of (3,,) to shape of (,,3)
    img_normalized = img_normalized.transpose(1, 2, 0)
 
    # display the normalized image
    plt.imshow(img_normalized)
    plt.xticks([])
    plt.yticks([])

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


def brightness_control(img, value=30):
    """
    Args:
        - img (ndarray): BGR numpy array for an image
        - value (float): value of the (increasing/decreasing) brightness

    Returns:
        img (ndarray): the modified brightness image
    """
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def saturation_control(img, value=30):
    
    """
    Args:
        - img (ndarray): BGR numpy array for an image
        - value (float): value of the (increasing/decreasing) saturation

    Returns:
        - img (ndarray): the modified saturation image
    """
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    s[s > lim] = 255
    s[s <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def minmax_scaler(channel):
  mini = np.min(channel)
  maxi = np.max(channel)
  for i in range(channel.shape[0]):
    for j in range(channel.shape[1]):
        channel[i,j] = 255*(channel[i,j]-mini)/(maxi-mini)
  return channel

def minmax_contrast_stretching(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)
  y = minmax_scaler(y)
  stretched_img_hsv = cv2.merge((h, s, v))
  stretched_img_hsv = cv2.cvtColor(stretched_img_hsv, cv2.COLOR_HSV2BGR)
  return stretched_img_hsv

def needs_histogram_equalization(img):
    """
    Checks if an image might benefit from histogram equalization based on several criteria.

    Args:
        image (str): the image file.

    Returns:
        bool: True if HE is likely to improve the image, False otherwise.
    """
    # Calculate various metrics to assess the need for HE:
    # 1. Standard deviation (low SD suggests limited contrast)
    std_dev = np.std(img)

    # 2. Entropy (low entropy indicates low contrast/information)
    entropy = -np.sum(np.histogram(img, bins=256)[0] * np.log2(np.histogram(img, bins=256)[0] + 1e-8))

    # 3. Michelson contrast (measures average contrast across the image)
    max_intensity = np.max(img)
    min_intensity = np.min(img)
    michelson_contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity)

    # 4. Otsu's threshold (globally optimal thresholding can indicate bimodal/low-contrast distributions)
    #_, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Histogram shape (peaked histogram suggests HE potential)
    hist, _ = np.histogram(img.ravel(), bins=256)
    peakedness = np.max(hist) / np.sum(hist)  # Ratio of max value to total counts

    # Combine metrics using weighted decision rules:
    low_contrast_threshold = 55  # Adjust based on image types and preferences
    low_michelson_threshold = 1  # Adjust based on desired minimum contrast
    #low_information_threshold = 3  # Adjust based on expected entropy values

    low_contrast = std_dev < low_contrast_threshold
    low_michelson = michelson_contrast < low_michelson_threshold
    #low_information = entropy < low_information_threshold
    #peaked_histogram = peakedness > 0.1  # Adjust based on expected histogram shapes
    print(f'\n\n\n\t\t\t---Image Info---\n std == {std_dev}\n entropy == {entropy}\n mc == {michelson_contrast}\n peakedness == {peakedness}\n\n')
    if low_contrast or low_michelson or math.isinf(michelson_contrast):
        # Consider these indicators suggestive of potential HE benefit
        return True
    else:
        return False

