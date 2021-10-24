import os
import albumentations as A
import numpy as np
import cv2
import pickle

#Load data and plot
CATEGORIES_ALL = ["Artifact", "BC", "cCry", "Dirt", "hCast", "LD", "nhCast", "nsEC", "RBC", "sCry", "sEC", "uCry", "Unclassified", "WBC"]
CATEGORIES_INLIER = ["cCry", "hCast", "nhCast", "nsEC", "RBC", "sCry", "sEC", "uCry", "WBC"]
CATEGORIES_OUTLIER1 = ["Artifact", "Dirt", "LD"]
CATEGORIES_OUTLIER2 = ["blankurine", "bubbles", "condensation", "dust", "feces", "fingerprint", "humanhair",
                      "Lipids", "Lotion", "pollen", "semifilled", "void", "wetslide", "yeast"]
UNCLASSIFIED = "Unclassified"
# Inliers
# cCry: 105, hCast: 91, nhCast: 62, nsEC: 1009, RBC: 2042, sCry: 529, sEC:819, uCry: 512, WBC:2841
# Outliers
# Artifact: 39, Dirt: 62, Ld: 12
# New Outliers
# blankurine: 121, bubbles: 61, cathair: 5, condensation: 119, dust: 4139, feces: 6045, fingerprint: 1014, humanhair: 52,
# Lipids: 502, Lotion: 3226, pollen: 1818, semifilled: 203, void: 186, wetslide: 730, yeast: 1849

def croppedImage(image, dim):
    height, width = dim
    if(image[0][0] == 0):
        if(image[0][width-1] == 0):
            return True
        elif(image[height-1][0] == 0):
            return True
    elif(image[height-1][width-1] == 0):
        if(image[height-1][0] == 0):
            return True
        elif(image[0][width-1] == 0):
            return True
    else:
        return False
def getNumberOfBlackedImage(folder):
    counter = 0
    if folder in CATEGORIES_OUTLIER2:
        DATASET_DIR = "/home/erdem/dataset/patches_contaminants_32_scaled"
    elif folder in CATEGORIES_ALL:
        DATASET_DIR = "/home/erdem/dataset/patches_urine_32_scaled"
    else:
        print("Wrong folder name")
        pass
        
    path = os.path.join(DATASET_DIR,folder)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            if(croppedImage(img_array, (32,32))):
                counter += 1
        except Exception as e:
            pass
    return counter
def createRawInlierData():
    inlier_dataset = []
    for category in CATEGORIES_INLIER:
        path = os.path.join(DATASET_DIR,category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                inlier_dataset.append(img_array)
            except Exception as e:
                pass
    return inlier_dataset

def createRawOutlierData():
    outlier_dataset = []
    for category in CATEGORIES_OUTLIER1:
        path = os.path.join(DATASET_DIR,category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                outlier_dataset.append(img_array)
            except Exception as e:
                pass
    return outlier_dataset

def createUnclassifiedData():
    unclassified_dataset = []
    path = os.path.join(DATASET_DIR,UNCLASSIFIED)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            unclassified_dataset.append(img_array)
        except Exception as e:
            pass
    return unclassified_dataset

def getRawImages(folder, dim):
    # Returns the unnormalized array of unnormalized numpy arrays (height, width)
    height, width = dim
    array = []
    if folder in CATEGORIES_OUTLIER2:
        DATASET_DIR = "/home/erdem/dataset/patches_contaminants_32_scaled"
    elif folder in CATEGORIES_ALL:
        DATASET_DIR = "/home/erdem/dataset/patches_urine_32_scaled"
    else:
        print("Wrong folder name")
        pass
        
    path = os.path.join(DATASET_DIR,folder)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            if(dim!=(32,32)):
                img_array = cv2.resize(img_array, dim, interpolation = cv2.INTER_LINEAR) 
            array.append(img_array)
        except Exception as e:
            pass
    # X_array = np.array(array).reshape(-1, height, width, 1)
    # # normalization
    # X_array = X_array.astype('float32') / 255
    # y_array = np.zeros(len(array))
    # for i in range(y_array.size):
    #     y_array[i] += label
    return array

def getImages(folder, dim):
    # Returns the unnormalized array of unnormalized numpy arrays (height, width)
    height, width = dim
    array = []
    if folder in CATEGORIES_OUTLIER2:
        DATASET_DIR = "/home/erdem/dataset/patches_contaminants_32_scaled"
    elif folder in CATEGORIES_ALL:
        DATASET_DIR = "/home/erdem/dataset/patches_urine_32_scaled"
    else:
        print("Wrong folder name")
        pass
        
    path = os.path.join(DATASET_DIR,folder)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            if(croppedImage(img_array, dim)):
                pass
            else:
                if(dim!=(32,32)):
                    img_array = cv2.resize(img_array, dim, interpolation = cv2.INTER_LINEAR) 
                array.append(img_array)
        except Exception as e:
            pass
    # X_array = np.array(array).reshape(-1, height, width, 1)
    # # normalization
    # X_array = X_array.astype('float32') / 255
    # y_array = np.zeros(len(array))
    # for i in range(y_array.size):
    #     y_array[i] += label
    return array


def getTransformedImages(folder, dim, transform, n_sample):
    # Returns the unnormalized array of unnormalized, albumentations-ShiftScaleRotated numpy arrays (height, width)
    height, width = dim
    array = []
    if folder in CATEGORIES_OUTLIER2:
        DATASET_DIR = "/home/erdem/dataset/patches_contaminants_32_scaled"
    elif folder in CATEGORIES_ALL:
        DATASET_DIR = "/home/erdem/dataset/patches_urine_32_scaled"
    else:
        print("Wrong folder name")
        pass
        
    path = os.path.join(DATASET_DIR,folder)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            if(croppedImage(img_array, dim)):
                pass
            else:
                if(dim!=(32,32)):
                    img_array = cv2.resize(img_array, dim, interpolation = cv2.INTER_LINEAR) 
                for a in range(n_sample):
                    transformed = transform(image=img_array)
                    transformed_image = transformed["image"]
                    array.append(transformed_image)

        except Exception as e:
            pass
    # X_array = np.array(array).reshape(-1, height, width, 1)
    # # normalization
    # X_array = X_array.astype('float32') / 255
    # y_array = np.zeros(len(array))
    # for i in range(y_array.size):
    #     y_array[i] += label
    return array
def getTestRawImages(folder, dim):
    height, width = dim
    images = getImages(folder, dim)
    test_images = np.array(images).reshape(-1, height, width, 1)
    test_images = test_images.astype('float32') / 255
    return test_images
def getTestOutliers(outliers, labels, dim):
    for i in range(outliers):
        cur_outlier = getTestRawImages(outliers[i], dim)
        
def mergeTwoX(X_arr1, X_arr2, y_arr1, y_arr2):
    X_all = np.concatenate((X_arr1, X_arr2))
    y_all = np.concatenate((y_arr1, y_arr2))
    return X_all, y_all

def mergeTwoX_shuffle(X_arr1, X_arr2, y_arr1, y_arr2):
    X_all = np.concatenate((X_arr1, X_arr2))
    y_all = np.concatenate((y_arr1, y_arr2))
    randomize = np.arange(len(X_all))
    np.random.shuffle(randomize)
    X_all = X_all[randomize]
    y_all = y_all[randomize]
    return X_all, y_all