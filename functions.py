
import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from time import time

# Initializing mediapipe segmentation class.
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Setting up Segmentation function.
segment = mp_selfie_segmentation.SelfieSegmentation(0)

#input from user to chose what type of action to perform
#1 changing background from image
print('chose your action : \n press 1 to perform actions on an image \n press 2 for realtime background replacement')
user_choice=input()


def modifyBackground(image, background_image=255, blur=95, threshold=0.3, display=True, method='changeBackground'):
    '''
    This function will replace, blur, desature or make the background transparent depending upon the passed arguments.
    Args:
        image: The input image with an object whose background is required to modify.
        background_image: The new background image for the object in the input image.
        threshold: A threshold value between 0 and 1 which will be used in creating a binary mask of the input image.
        display: A boolean value that is if true the function displays the original input image and the resultant image
                 and returns nothing.
        method: The method name which is required to modify the background of the input image.
    Returns:
        output_image: The image of the object from the input image with a modified background.
        binary_mask_3: A binary mask of the input image.
    '''

    # Convert the input image from BGR to RGB format.
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the segmentation.
    result = segment.process(RGB_img)

    # Get a binary mask having pixel value 1 for the object and 0 for the background.
    # Pixel values greater than the threshold value will become 1 and the remainings will become 0.
    binary_mask = result.segmentation_mask > threshold

    # Stack the same mask three times to make it a three channel image.
    binary_mask_3 = np.dstack((binary_mask, binary_mask, binary_mask))

    if method == 'changeBackground':

        # Resize the background image to become equal to the size of the input image.
        background_image = cv2.resize(background_image, (image.shape[1], image.shape[0]))

        # Create an output image with the pixel values from the original sample image at the indexes where the mask have
        # value 1 and replace the other pixel values (where mask have zero) with the new background image.
        output_image = np.where(binary_mask_3, image, background_image)
        cv2.imwrite(r'E:\flasksaved\newimg.png', output_image)

    elif method == 'blurBackground':

        # Create a blurred copy of the input image.
        blurred_image = cv2.GaussianBlur(image, (blur, blur), 0)

        # Create an output image with the pixel values from the original sample image at the indexes where the mask have
        # value 1 and replace the other pixel values (where mask have zero) with the new background image.
        output_image = np.where(binary_mask_3, image, blurred_image)
        cv2.imwrite(r'E:\flasksaved\newimg.png', output_image)

    elif method == 'desatureBackground':

        # Create a gray-scale copy of the input image.
        grayscale = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

        # Stack the same grayscale image three times to make it a three channel image.
        grayscale_3 = np.dstack((grayscale, grayscale, grayscale))

        # Create an output image with the pixel values from the original sample image at the indexes where the mask have
        # value 1 and replace the other pixel values (where mask have zero) with the new background image.
        output_image = np.where(binary_mask_3, image, grayscale_3)
        cv2.imwrite(r'E:\flasksaved\newimg.png', output_image)

    elif method == 'transparentBackground':

        # Stack the input image and the mask image to get a four channel image.
        # Here the mask image will act as an alpha channel.
        # Also multiply the mask with 255 to convert all the 1s into 255.
        output_image = np.dstack((image, binary_mask * 255))
        cv2.imwrite(r'E:\flasksaved\newimg.png', output_image)

    else:
        # Display the error message.
        print('Invalid Method')

        # Return
        return

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22, 22])
        # plt.subplot(121);
        # plt.imshow(image[:, :, ::-1]);
        # plt.title("Original Image");
        # plt.axis('off');
        # plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');
        plt.show()

    # Otherwise
    else:

        # Return the output image and the binary mask.
        # Also convert all the 1s in the mask into 255 and the 0s will remain the same.
        # The mask is returned in case you want to troubleshoot.
        return output_image, (binary_mask_3 * 255).astype('uint8')

def realtime_bgremoval():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # width
    cam.set(4, 480)  # height
    cam.set(cv2.CAP_PROP_FPS, 60)
    segmentor = SelfiSegmentation()  # defining the segmentation model default=1 for landscape mode
    fps = cvzone.FPS()
    imagebg = cv2.imread("bgimg/1.jpg")
    # standard loop for webcam access
    listImg = os.listdir('bgimg')
    print(listImg)
    imgList = []
    for x in listImg:
        img = cv2.imread(f'bgimg/{x}')
        imgList.append(img)
    print(len(imgList))
    imageindex = 0
    while (True):
        success, image = cam.read()
        imageout = segmentor.removeBG(image, imgList[imageindex], threshold=0.7)  # this variable to get the output

        combined = cvzone.stackImages([image, imageout], 2, 1)
        _, combined = fps.update(combined, color=(0, 0, 255))
        cv2.imshow("live", image)
        cv2.imshow("liveout", imageout)
        key = cv2.waitKey(1)
        if key == ord('p'):
            if imageindex > 0:
                imageindex -= 1
        elif key == ord('n'):
            if imageindex < len(imgList) - 1:
                imageindex += 1
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if user_choice=='1':
    x ='changeBackground'
    y ='blurBackground'
    z ='desatureBackground'
    c ='transparentBackground'

    print('paste path of image')
    path_of_image=input()

    image_to_process=cv2.imread(path_of_image)
    print('chose type of action \n x:changeBackground \n y:blurBackground \n z:desatureBackground \n c:transparentBackground')
    to_do = input()
    if to_do=='x':
        print('paste path of bgimage')
        path_of_bgimage=input()

        bg_image=cv2.imread(path_of_bgimage)
        modifyBackground(image_to_process, bg_image, 0.7, method='changeBackground')

    elif to_do == 'y':
        modifyBackground(image_to_process, method='blurBackground')

    elif to_do == 'z':
        modifyBackground(image_to_process, method='desatureBackground')

    elif to_do == 'c':
        modifyBackground(image_to_process, method='transparentBackground')
    else:
        print('invalid input')

elif user_choice=='2':
    realtime_bgremoval()
else:
    print('invalid choice')
