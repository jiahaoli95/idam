import numpy as np
from PIL import Image


def crop(path, save_path):
    thr = 250.
    im = Image.open(path)
    im = np.array(im)[:, :, :3]
    im = im.astype(np.float64)
    im[im>thr] = 255.
    idx = np.where(im.mean(-1)<255.)
    min_y, min_x = idx[0].min(), idx[1].min()
    max_y, max_x = idx[0].max(), idx[1].max()
    cropped = im[min_y:max_y+1, min_x:max_x+1]
    h, w = cropped.shape[:2]
    im = np.ones((max(h, w), max(h, w), 3)) * 255.
    if h > w:
        im[:, (h-w)//2: (h-w)//2+w] = cropped
    else:
        im[(w-h)//2: (w-h)//2+h, :] = cropped
    im = im.astype(np.uint8)
    Image.fromarray(im).save(save_path)


def arrange(paths, save_path, ncol=5, row_margin=20, col_margin=20):
    im_list = [Image.open(x) for x in paths]
    w = int(min([x.size[0] for x in im_list]) * 0.9)
    h = int(min([x.size[1] for x in im_list]) * 0.9)
    im_list = [x.resize((w, h), Image.BILINEAR) for x in im_list]
    arr_list = [np.array(im)[:, :, :3] for im in im_list]
    nrow = int(np.ceil(len(arr_list)/ncol))
    grid_h = nrow * h + (nrow - 1) * row_margin
    grid_w = ncol * w + (ncol - 1) * col_margin
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    for i in range(len(arr_list)):
        row_idx = i // ncol
        col_idx = i % ncol
        min_y = (h + row_margin) * row_idx
        min_x = (w + col_margin) * col_idx
        grid[min_y:min_y+h, min_x:min_x+w] = arr_list[i]
    Image.fromarray(grid).save(save_path)


def rm_bg(path, save_path):
    thr = 180
    assert save_path.endswith('.png')
    im = Image.open(path)
    w, h = im.size
    im = np.array(im)[:, :, :3]
    mask = im.astype(np.float32).mean(-1) < thr
    alpha = np.zeros((h, w, 1), dtype=np.uint8)
    alpha[mask] = 255
    im = np.concatenate([im, alpha], -1)
    Image.fromarray(im).save(save_path)


if __name__ == '__main__':
    pass
