

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


def convolve_hist(hist: np.ndarray, size: int = 16):

    c = np.empty(2*size-1)
    for i in range(size - 1):
        c[i] = i + 1
        c[-i-1] = i+1
    c[size - 1] = size

    return np.convolve(hist, c, mode = 'valid')

def plot_hist_convolved(img_gray: np.ndarray, dir_path: str):
    fig = plt.figure(figsize=(15, 12))
    # fig.tight_layout()

    grid = plt.GridSpec(4, 6, hspace=0.7, wspace=0.6)
    subs = (
        fig.add_subplot(grid[:2, :3], yticklabels=[]),
        fig.add_subplot(grid[:2, 3:], yticklabels=[]),
        fig.add_subplot(grid[2:4, :3], yticklabels=[]),
        fig.add_subplot(grid[2:4, 3:], yticklabels=[])
    )

    a, _ = np.histogram(img_gray.ravel(), bins = 255)
    s = 16
    conv = convolve_hist(a, s)

    for sub, arr, title, color in zip(
            subs,
            (a, conv, np.log10(a + 1), np.log10(conv + 1)),
            ('base hist', 'convolved log=False', 'base hist log=True', 'log=True'),
            ('blue', 'green', 'red', 'black')
    ):
        sub.plot(arr, color=color)
        sub.set_title(title)

    plt.savefig(os.path.join(dir_path, f"conv size {s}.png"), dpi=200)
    plt.close()



def plot_image_hist(img_rgb: np.ndarray, dir_path: str, plot_color_part: bool = False):
    """
    plots histogram for gray and r/g/b components of image
    :param img_rgb:
    :param path:
    :return:
    """
    to_folder = lambda s: os.path.join(dir_path, s)

    avg_gray = img_rgb[:,:,0]/3 + img_rgb[:,:,1]/3 + img_rgb[:,:,2]/3
    lum_gray = img_rgb[:,:,0]*0.2125 + img_rgb[:,:,1]*0.7174 + img_rgb[:,:,2]*0.0721

    if plot_color_part:

        for log in (False, True):

            fig = plt.figure(figsize=(9, 12))
            #fig.tight_layout()

            grid = plt.GridSpec(7, 6, hspace = 0.7, wspace=0.6)
            subs = (
                fig.add_subplot(grid[:2, :],  yticklabels=[]),
                fig.add_subplot(grid[2:4, :], yticklabels=[]),
                fig.add_subplot(grid[4:, :2], yticklabels=[]),
                fig.add_subplot(grid[4:, 2:4], yticklabels=[]),
                fig.add_subplot(grid[4:, 4:], yticklabels=[]),
            )

            for sub, arr, color in zip(
                subs,
                (avg_gray, lum_gray, img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]),
                ('gray', 'gray', 'red', 'green', 'blue')
            ):
                sub.hist(arr.ravel(), bins = 255, color = color, histtype = 'stepfilled', log = log)

            subs[0].set_title('avg gray')
            subs[1].set_title('709 gray')

            fig.suptitle(f'log={log}', fontsize=16)
            plt.savefig(to_folder(f"hist_log={log}.png"), dpi = 200)
            plt.close()


        is_not_gray = (img_rgb.max(axis = 2) - img_rgb.min(axis = 2)) > 25

        Image.fromarray(cv2.bitwise_and(img_rgb, img_rgb, mask = is_not_gray.astype(np.uint8))).convert('RGB').save(to_folder("not_gray.jpg"))


    plot_hist_convolved(lum_gray, dir_path)




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



