'''
Basic PNG image loading support, with translation to numpy arrays.

Andrew H. Fagg


Files are organized as:
/BASE_DIRECTORY/CONDITION/OBJECT_ID/IMAGE_ddd.png
where:
- CONDITION is s1 ... s11
- OBJECT_ID is o1 ... o50
- ddd is 0 ... 299
'''

# From pypng
import png
import os
import fnmatch
import re
import numpy as np

# Image Reading Tools
def readPngFile(filename:str)->np.array:
    '''
    Read a single PNG file
    
    :param filename: fully qualified file name to load
    
    :return: 3D numpy array.  Shape = (rows x cols x chans)
    
    Note: all pixel values are floats in the range 0.0 .. 1.0
    
    This implementation relies on the pypng package (there are better ways)
    '''
    # Load in the image meta-data
    r = png.Reader(filename)
    it = r.read()
    
    # Load in the image itself and convert to a 2D array
    image_2d = np.vstack(list(map(np.uint8, it[2])))
    
    # Reshape into rows x cols x chans and scale
    image_3d = np.reshape(image_2d,
                         (it[0],it[1],it[3]['planes'])) / 255.0
    
    return image_3d

def read_images_from_directory(directory:str, file_regexp:str)->np.array:
    '''
    Read a set of images from a directory.  All of the images must be the same shape
    
    :param directory: Directory to search (/BASE_DIRECTORY/CONDITION/OBJECT_ID/)
    :param file_regexp: a regular expression to match the file names against (IMAGE_ddd.png)
    :return: 4D numpy array.  shape: (images, rows, cols, 3)
    '''
    
    print(directory, file_regexp)
    
    # Get all of the file names (sorted)
    files = sorted(os.listdir(directory))
    
    # Construct a list of images from those that match the regexp
    list_of_images = [readPngFile(directory + "/" + f) for f in files if re.search(file_regexp, f) ]
    
    # Create a 3D numpy array
    return np.array(list_of_images, dtype=np.float32)

def read_image_set_from_directories(directory:str, spec:str)->np.array:
    '''
    Read a set of images from a set of directories
    
    :param directory: base directory to read from (/BASE_DIRECTORY/CONDITION)
    :param spec: n-array of tuples of subdirectories and file regexps (OBJECT_ID, IMAGE_ddd.png)
    :return: 4D numpy array.  Shape: (images, rows, cols, 3)
    
    '''
    # First tuple
    out = read_images_from_directory(directory + "/" + spec[0][0], spec[0][1])

    # Subsequent tuples
    for sp in spec[1:]:
        # Append to each new set to out
        out = np.append(out, read_images_from_directory(directory + "/" + sp[0], sp[1]), axis=0)

    # one
    return out

def load_multiple_image_sets_from_directories(directory_base, directory_list, object_list, test_files):
    '''
    Read a set of images from a set of directories 
    
    :param directory_base: base directory to read from (/BASE_DIRECTORY)
    :param directory_list: subdirectory (CONDITION)
    :param spec: n-array of tuples of subdirectories and file regexps (OBJECT_ID, IMAGE_ddd.png)
    :return: 4D numpy array.  Shape: (images, rows, cols, 3)
    
    '''

    print("##################")
    # Create the list of object/image specs
    inputs = [[obj, test_files] for obj in object_list]
    
    # First directory
    ret = read_image_set_from_directories(directory_base + "/" + directory_list[0], inputs)
    
    # Loop over the rest of the directories
    for directory in directory_list[1:]:
        ret = np.append(ret,
                        read_image_set_from_directories(directory_base + "/" + directory, inputs),
                        axis=0)

    return ret

