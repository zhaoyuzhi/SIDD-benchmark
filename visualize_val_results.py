import os
import cv2
import numpy as np

# ----------------------------------------
#             PATH processing
# ----------------------------------------
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = [] 
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def extract_patch(path, keyword_list):

    keyword = keyword_list[0]
    size = keyword_list[1]
    h = keyword_list[2]
    w = keyword_list[3]

    filelist = get_files(path)
    for i in range(len(filelist)):
        #if keyword in filelist[i]:
        if keyword == filelist[i].split('\\')[-1]:
            img_name = filelist[i]

    img = cv2.imread(img_name)
    img = img[h:(h+size), w:(w+size), :]

    return img

if __name__ == "__main__":

    save_path = 'comparison'
    check_path(save_path)

    method_list = ['./val_results/input', \
        './val_results/gt', \
            './val_results/SGN', \
                './val_results/SGN_cutblur', \
                    './val_results/SGN_DWT_cutblur', \
                        './val_results/MultilevelSGN', \
                            './val_results/MultilevelSGN_cutblur', \
                                './val_results/MultilevelSGN_cutblur_tanhl1', \
                                    './val_results/MultilevelSGN_DWT_cutblur', \
                                        './val_results/REDI']
    
    # kryword - size - h - w
    keyword_list = [['2_6.png', 256, 0, 0], \
        ['1_15.png', 256, 0, 0], \
            ['1_17.png', 256, 0, 0], \
                ['1_23.png', 256, 0, 0], \
                    ['3_4.png', 256, 0, 0], \
                        ['4_23.png', 256, 0, 0], \
                            ['10_10.png', 256, 0, 0], \
                                ['2_23.png', 256, 0, 0], \
                                    ['2_8.png', 256, 0, 0], \
                                        ['38_1.png', 256, 0, 0]]

    for i in range(len(method_list)):
        path = os.path.join(method_list[i])
        for j in range(len(keyword_list)):
            # extract image
            img = extract_patch(path, keyword_list[j])
            # savename
            method_name = method_list[i].split('/')[-1]
            img_name = keyword_list[j][0]
            savepath = os.path.join(save_path, img_name + '_' + method_name + '.png')
            cv2.imwrite(savepath, img)
