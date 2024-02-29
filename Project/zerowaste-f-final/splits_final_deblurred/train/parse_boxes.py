import cv2
import numpy as np
import os

curr_folder_path = os.getcwd()

sem_seg_folder_path = os.path.join(curr_folder_path, "sem_seg")

annotation_folder_path = os.path.join(curr_folder_path, "data")

list_of_images = os.listdir(sem_seg_folder_path)

for image in list_of_images:

    # open image with the masks
    image_path = os.path.join(sem_seg_folder_path, image)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # find how many masks are in the image
    (contours, hierarchy) = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # go through the masks and find the bounding box
    box = []
    for contour in contours:
        contour = np.array(contour).reshape(-1, 2)  # idk why contours has an extra dimention as (-1, 1, 2)
        x_min = np.min(contour[:,0])
        y_min = np.min(contour[:,1])
        x_max = np.max(contour[:,0])
        y_max = np.max(contour[:,1])
        label = int(img[contour[0,1], contour[0,0]])-1

        # if label == 0:
        #     print(f"Error at {image}: label is zero")
        
        # normalize everything
        x_center = (x_max + x_min)/(2)/img.shape[1]
        y_center = (y_max + y_min)/(2)/img.shape[0]

        width = (x_max - x_min)/img.shape[1]
        height = (y_max - y_min)/img.shape[0]

        box.append((label, x_center, y_center, width, height))
    
    # write annotation
    annotation_file_path = os.path.join(annotation_folder_path, image[:-4]+".txt")
    with open(annotation_file_path, 'w') as file:
        for b in box:
            for data in b:
                file.write(str(data) + ' ')
            file.write('\n')
    
    