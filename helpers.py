import numpy as np
import rasterio
import cv2
from PIL import Image


def rio_open(
    img_path
):
    '''
    Open a geotiff via rasterio and return a channels-last array
    '''
    with rasterio.open(img_path, 'r') as src:
        arr = src.read()
        arr = np.moveaxis(arr, 0, len(arr.shape)-1) # Channels last
    return arr

def rio_save(
    arr,
    img_path
):
    '''
    Save the contents of a numpy array to an unreferenced geotiff.

    arr: Numpy array to be saved
    save_path: Path to the resulting image file
    '''
    if len(arr.shape) == 2: # Grayscale
        arr = np.expand_dims(arr, 0) # Channels first
    elif len(arr.shape) == 3: # RGB
        arr = np.moveaxis(arr, 2, 0) # Channels first
    with rasterio.open(img_path,
                       'w',
                       driver = 'GTiff',
                       count = arr.shape[0],
                       height = arr.shape[1],
                       width = arr.shape[2],
                       dtype = arr.dtype,
                       compress = 'LZW') as dst:
        dst.write(arr)

def resize(arr, width = 512):
    '''
    Resize an image to specified width and retain aspect ratio

    arr: Numpy array containing the image (channels last)
    width: New width

    Returns a numpy array with the new shape
    '''

    ratio = width / arr.shape[1]
    height = int(arr.shape[0] * ratio)
    new_shape = (width, height)
    arr_resize = cv2.resize(arr, new_shape, interpolation = cv2.INTER_AREA)
    return arr_resize


def make_tiles(
    arr,
    tile_shape = (512, 512),
    pad = 0
):
    '''
    Split an image into tiles possibily with reflective padding on each tile.

    arr: Array or path of the image to be tiled
    tile_shape: The desired 2D tile_shape (number of channels are preserved)
    pad: Length of padding on each side. If >0, pad pixels will be reflected on
    each side of the tile.

    Returns an array of tiles and the number of tiles along the x and y directions
    '''


    ## Read full image as array
    if isinstance(arr, str):
        ext = arr.split('.')[-1]
        if ext.lower() in ['tif', 'tiff']:
            arr = rio_open(arr)
        else:
            arr = np.asarray(Image.open(arr))

    ## Crop image to get a whole number of chunks
    n_row_per_chunk, n_col_per_chunk = tile_shape[0:2]
    n_chunks_x = int(arr.shape[1] / n_col_per_chunk)
    n_chunks_y = int(arr.shape[0] / n_row_per_chunk)
    arr = arr[0:n_row_per_chunk*n_chunks_y, 0:n_col_per_chunk*(n_chunks_x)]

    ## Make list of tiles
    arr = [arr[x:x+n_row_per_chunk,y:y+n_col_per_chunk]
             for x in range(0,arr.shape[0],n_row_per_chunk)
             for y in range(0,arr.shape[1],n_col_per_chunk)]

    if pad:
        arr = [np.pad(tile, ((pad, pad), (pad, pad), (0,0)), 'reflect') for tile in arr]

    return np.asarray(arr), n_chunks_x, n_chunks_y


def stitch_tiles(
    tile_arr,
    n_tiles_x,
    n_tiles_y,
    n_col_per_tile = 512,
    n_row_per_tile = 512,
    crop = 0
):

    '''
    Given an array of image tiles, stitch them together into a single image.
    The inverse of make_tiles().

    tile_arr: Array of shape (number tiles, rows, columns, channels)
    n_tiles_[x|y]: Number of tiles along the x (columns) and y (row) directions.
    n_[col|row]_per_tile: Dimensions of each tile after cropping
    crop: Length of pixels to crop from each side of the tile before adding
    to the stitched image.

    Returns a single image of the tiles stitched together
    '''

    n_channels = tile_arr.shape[3]
    stitch = np.zeros((n_row_per_tile*n_tiles_y, n_col_per_tile*n_tiles_x, n_channels))

    for ndx in range(tile_arr.shape[0]):
        i,j = np.unravel_index(ndx, (n_tiles_y, n_tiles_x))
        i = i * n_row_per_tile
        j = j * n_col_per_tile

        if crop:
            tile = tile_arr[ndx, crop:-crop, crop:-crop]
        else:
            tile = tile_arr[ndx]

        stitch[i:i+n_row_per_tile, j:j+n_col_per_tile] = tile

    return stitch
