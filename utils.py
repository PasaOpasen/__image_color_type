

import os
from pathlib import Path

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pdf2image import convert_from_path

import cv2

def numpy_from_pdf(path_to_pdf: str, DPI: int = 150):
    images = convert_from_path(path_to_pdf, dpi = DPI)
    return [np.asarray(img.convert('RGB')) for img in images]


def plot_image_hist(img_rgb: np.ndarray, dir_path: str):
    """
    plots histogram for gray and r/g/b components of image
    :param img_rgb:
    :param path:
    :return:
    """
    to_folder = lambda s: os.path.join(dir_path, s)

    for log in (False, True):

        fig = plt.figure(figsize=(10, 8))

        grid = plt.GridSpec(5, 6, hspace=0.4, wspace=0.4)
        subs = (
            fig.add_subplot(grid[:2, :],  yticklabels=[]),
            fig.add_subplot(grid[2:, :2], yticklabels=[]),
            fig.add_subplot(grid[2:, 2:4], yticklabels=[]),
            fig.add_subplot(grid[2:, 4:], yticklabels=[]),
        )

        for sub, arr, color in zip(
            subs,
            (img_rgb[:,:,0]/3 + img_rgb[:,:,1]/3 + img_rgb[:,:,2]/3, img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]),
            ('gray', 'red', 'green', 'blue')
        ):
            sub.hist(arr.ravel(), bins = 255, color = color, histtype = 'stepfilled', log = log)

        fig.suptitle(f'log={log}', fontsize=16)
        plt.savefig(to_folder(f"hist_log={log}.png"), dpi = 200)
        plt.close()


    is_not_gray = (img_rgb.max(axis = 2) - img_rgb.min(axis = 2)) > 25

    Image.fromarray(cv2.bitwise_and(img_rgb, img_rgb, mask = is_not_gray.astype(np.uint8))).convert('RGB').save(to_folder("not_gray.jpg"))



def plot_pdf_hist(path_to_pdf: str, directory: str, DPI: int = 150):

    images = convert_from_path(path_to_pdf, dpi = DPI)

    def save(img: Image.Image, page_path: str):
        Path(page_path).mkdir(parents=True, exist_ok=True)

        img.save(
            os.path.join(page_path, "orig.jpg")
        )
        plot_image_hist(
            img_rgb=np.asarray(img.convert('RGB')),
            dir_path=page_path
        )


    if len(images) == 1:
        save(images[0],  directory)
    else:
        for page, img in enumerate(images):
            page_path = os.path.join(directory, f"page_{page+1}")
            save(img, page_path)



def plot_dirpdf_hist(path_to_directory: str, directory: str, DPI: int = 150):

    files = [os.path.join(path_to_directory, file) for file in os.listdir(path_to_directory) if file.endswith('.pdf')]

    for file in tqdm(files):
        plot_pdf_hist(
            path_to_pdf=file,
            directory=os.path.join(directory, Path(file).stem),
            DPI = DPI
        )



