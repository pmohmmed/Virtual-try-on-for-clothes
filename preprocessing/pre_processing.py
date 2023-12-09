import cv2

def image_resizing(images):
    img = cv2.imread('filename.jpg')

    # Resize the image to a specific size (e.g., 54x140 pixels)
    resized_imgs = cv2.resize(img, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)
    return resized_imgs




def images_normalization(images):
    pass


def data_splitting(data):
    pass


def data_augmentation(data):
    pass


def remove_image_background(images):
    pass