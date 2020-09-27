#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Shohei Fujii <fujii.shohei@gmail.com>'


from PIL import Image
from skimage.morphology import binary_dilation
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import logging
from tqdm import tqdm

# logging.basicConfig(format='[%(levelname)s] [%(name)s: %(funcName)s] %(message)s', level=logging.DEBUG)
console = logging.StreamHandler()
console_formatter = logging.Formatter('[%(levelname)s] [%(name)s: %(funcName)s] %(message)s')
console.setFormatter(console_formatter)
console.setLevel(logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(console)


g_eps = 1e-8

def load_image(path: pathlib.Path):
    # img = np.array(Image.open(path).convert('RGB'))
    img = np.array(Image.open(path).convert('L')) / 255.0
    return img


def save_image(img: np.array, path: pathlib.Path):
    newimg = Image.fromarray(img*255).convert('L')
    newimg.save(str(path))


def show_image(img: np.array):
    plt.imshow(img, cmap='gray', vmin=0.0, vmax=1.0)
    plt.show()


def initialize(img: np.array, scale: float, halfwinsize: int):
    target = np.zeros(tuple((np.array(img.shape)*scale).astype(np.int)))
    offset = (np.array(target.shape) - np.array(img.shape)) // 2
    # put img at the center of the target img
    target[offset[0]:offset[0]+img.shape[0],
           offset[1]:offset[1]+img.shape[1]] = img
    target_filled = np.zeros_like(target)
    target_filled[offset[0]:offset[0]+img.shape[0],
                  offset[1]:offset[1]+img.shape[1]] = 1

    return target, target_filled


def cache_sample_textures(img: np.array, winsize: int):
    """
    Return:
        sampletextures: [# of samples, H, W]
        vsamplepos: [(row1, col1), (row2, col2), ....]
    """
    sampletextures = []
    vsamplepos = []
    for r in range(0, img.shape[0] - winsize):
        for c in range(0, img.shape[1] - winsize):
            sampletextures.append(img[r:r+winsize, c:c+winsize])
            vsamplepos.append((r, c))
    sampletextures = np.array(sampletextures)
    vsamplepos = np.array(vsamplepos)
    return sampletextures, vsamplepos


def gaussian_kernel(winsize: int, sigma: float = 1.0, mu: float = 0.0):
    x, y = np.meshgrid(np.linspace(-1, 1, winsize), np.linspace(-1, 1, winsize))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
    g /= g.sum()  # accumulated probability must be 1.0
    return g


def plot_kernel(kernel):
    sx, sy = kernel.shape
    x, y = np.meshgrid(np.linspace(-1, 1, sx), np.linspace(-1, 1, sy))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, kernel, cmap='bwr', linewidth=0)
    fig.colorbar(surf)
    ax.set_title("Kernel Plot")
    fig.show()
    

def texture_synthesis(img: np.array, winsize: int, scale: float = 1.5, eps: float = 0.1, visualize: bool = False, progress_dir: str = None):
    assert(winsize % 2 == 1)
    halfwinsize = winsize//2

    # initialize
    target, target_filled = initialize(img, scale, halfwinsize)

    # WARN: do not forget to write values for all these variables
    # img_padded = np.pad(img, (halfwinsize, halfwinsize), 'reflect')
    target_padded = np.pad(
        target, (halfwinsize, halfwinsize), 'constant', constant_values=0)
    target_filled_padded = np.pad(
        target_filled, (halfwinsize, halfwinsize), 'constant', constant_values=0)

    # cache sample textures
    sampletextures, vsamplepos = cache_sample_textures(img, winsize)

    g = gaussian_kernel(winsize, sigma=winsize/6.4)

    # from IPython.terminal import embed; ipshell = embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())
    iprogress = 0
    with tqdm(total=np.size(target)) as progressbar:
        #if True:
        n_last_filled = 0
        while np.sum(target_filled) < np.size(target):
            #log.info(f'filled pixels: {int(np.sum(target_filled))}/{np.size(target)}')
            progressbar.set_description(f'filled pixels: {int(np.sum(target_filled))}/{np.size(target)}')
            progressbar.update(np.sum(target_filled) - n_last_filled)
            n_last_filled = np.sum(target_filled)

            # take the contour of filled img
            # indices = ([r1, r2, ...], [c1, c2, ...])
            indices = np.nonzero(binary_dilation(target_filled) - target_filled)

            # fill the pixel first where more number of surrounding pixels are known
            vknownpixels = np.zeros(len(indices[0]))
            for i, (r, c) in enumerate(zip(*indices)):
                vknownpixels[i] = np.sum(target_filled_padded[r:r+winsize, c:c+winsize])
            fillingorder = np.flip(np.argsort(vknownpixels))

            for i in tqdm(fillingorder, leave=False):
                # for every filling pixel
                r, c = indices[0][i], indices[1][i]
                # NOTE: r - halfwinsize (padding) + halfwinsize (half of window size) = r
                target_cropped = target_padded[r:r+winsize, c:c+winsize]
                mask = target_filled_padded[r:r+winsize, c:c+winsize]
                weight = np.multiply(g, mask)

                # compute SSD for every pair of sampletexture and target_cropped
                distance = (sampletextures - target_cropped)**2  # [# of samples, H, W]
                distance_filtered = (distance*weight) / np.sum(weight)  # [# of samples, H, W]
                # flatten 2-dimentional distance and compute summation
                ssd = np.sum(np.reshape(distance_filtered, (distance_filtered.shape[0], -1)), axis=1)  # [# of samples])
                minssd = min(ssd)

                pixels = []
                sampleindices = []
                threshold = minssd*(1.0 + eps)
                for isample, err in enumerate(ssd):
                    if err < threshold + g_eps:
                        pixels.append(sampletextures[isample][halfwinsize, halfwinsize])
                        sampleindices.append(isample)

                # debugging code to check sampled textures
                #  plt.subplot(2, len(pixels), 1)
                #  plt.imshow(target_cropped, cmap='gray', vmin=0.0, vmax=1.0)
                #  for j, isample in enumerate(sampleindices):
                #      plt.subplot(2, len(pixels), len(pixels)+j+1)
                #      plt.xlabel('sample '+ str(vsamplepos[isample]))
                #      plt.imshow(sampletextures[isample], cmap='gray', vmin=0.0, vmax=1.0)
                #  plt.pause(0.0001)
                #  # from IPython.terminal import embed; ipshell = embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())
                #  plt.clf()
                #log.info(f'pixel candidates: {len(pixels)}')
                assert(len(pixels) != 0)

                # uniform sampling
                p = np.random.choice(pixels)
                target[r, c] = p
                target_padded[r+halfwinsize, c+halfwinsize] = p
                target_filled[r, c] = 1
                target_filled_padded[r+halfwinsize, c+halfwinsize] = 1

                if visualize:
                    plt.imshow(target, cmap='gray', vmin=0.0, vmax=1.0)
                    plt.plot(c, r, marker='o')
                    plt.pause(0.0001)
                    plt.clf()
            iprogress += 1
            if progress_dir is not None:
                progressimg = Image.fromarray(target*255).convert('L')
                dir = pathlib.Path(progress_dir)
                dir.mkdir(exist_ok=True)
                progressimg.save(str(dir / (('%010d' % iprogress) + '.png')))

    return target


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgpath', help='path to image',
                        type=str, required=True)
    parser.add_argument('-winsize', help='window size', type=int, default=10)
    parser.add_argument('-eps', help='ssd margin for sample selection', type=int, default=0.1)
    parser.add_argument('-visualize', help='enable progress visualization', action='store_true')
    parser.add_argument('-saveprogress', help='dir to save images in progress', type=str)
    parser.add_argument('-savepath', help='path to save image', type=str, default='result.png')
    args = parser.parse_args()
    img = load_image(args.imgpath)
    winsize = args.winsize
    eps = args.eps
    visualize = args.visualize
    saveprogress = args.saveprogress

    newimg = texture_synthesis(img, winsize, scale=1.5, eps=eps, visualize=visualize, progress_dir=saveprogress)
    save_image(newimg, pathlib.Path(args.savepath))
