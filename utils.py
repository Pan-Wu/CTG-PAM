'''
Description: Some used utils.
'''
from skimage import measure as skm
import scipy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
     
def get_all_files(folder_path):
    file_names = []
    for root, directories, files in os.walk(folder_path):
        for file in files:
            file_names.append(os.path.join(root, file))
    return file_names

def norm(a):
    amin = a.min()
    amax = a.max()
    if amin != amax:
        return (a - amin) / (amax - amin)
    elif amin == 0:
        return a * 0
    else:
        return a * 0 + 1
    
def col_norm(x):
    x_nor=(x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))
    return x_nor

def find_location(a):
    aa = (a>0)*1
    props = skm.regionprops(aa)[0]
    center = props['centroid']
    area = props['area']
    return center, area

def location_to_dist(location1, location2):
    dist_mat = scipy.spatial.distance.cdist(location1, location2)
    return dist_mat

def draw_contour(img, mask, save_path, color):
    contours = []
    mask_n = mask.max()
    for i in range(mask_n):
        cell = (mask == (i+1)) * 1
        contour = skm.find_contours(cell)
        contours.append(contour[0])
    plt.imshow(img)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], color+'-' , linewidth=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.close()
def draw_multi_contour(img, mask_list, save_path, color_list):
    plt.imshow(img)
    for mask, color in zip(mask_list, color_list):
        contours = skm.find_contours(mask)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], color+'-' , linewidth=2)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.close()

def floor_for_float(a, n):
    t = 10 ** n
    b = int(a*t)
    b /= t
    return round(b, n)