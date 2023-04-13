import cv2
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
import multiprocessing
import pickle
import numpy as np

IMAGE_DIR = './datasets/train/images/'
SAVE_DIR = './figures/hist'
IMG_COUNTER = 1

IMAGES = glob.glob('{}/*.jpg'.format(SAVE_DIR))

if not os.path.exists('./figures/'):
    os.mkdir('./figures/')

if len(IMAGES) == 0:
    IMG_COUNTER = 1
else:
    IMG_COUNTER = len(IMAGES) + 1

def process_image(image_size):
    image = cv2.imread(image_size)
    return image.shape

def create_image_size(image_folder_path):

    # Get all the images in the folder

    image_files = glob.glob(os.path.join(IMAGE_DIR,'*.jpg'))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        image_sizes = list(tqdm(pool.imap(process_image, image_files), total=len(image_files)))

    

    # Save the image sizes

    with open('image_sizes.pkl', 'wb') as f:
        pickle.dump(image_sizes, f)

    with open('./image_sizes.pkl', 'rb') as f:
        image_sizes = pickle.load(f)

    # Create a histogram of the image sizes
    image_sizes = np.array(image_sizes)

    # Compute the histogram of the image sizes
    bins_width = np.linspace(0, max(image_sizes[:,0]), 50)
    bins_height = np.linspace(0, max(image_sizes[:,1]), 50)
    hist_width, _ = np.histogram(image_sizes[:, 0], bins_width)
    hist_height, _ = np.histogram(image_sizes[:, 1], bins_height)

    # Plot the histogram
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

    ax[0].bar(bins_width[:-1], hist_width, width=bins_width[1]-bins_width[0])
    ax[0].set_xlabel('Image Width')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Histogram of Image Widths')

    ax[1].bar(bins_height[:-1], hist_height, width=bins_height[1]-bins_height[0])
    ax[1].set_xlabel('Image Height')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('Histogram of Image Heights')

    
    plt.savefig(SAVE_DIR + '{}.png'.format(IMG_COUNTER))

if __name__ == '__main__':
    create_image_size(IMAGE_DIR)

    
