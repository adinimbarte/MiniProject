import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os


cam=cv2.VideoCapture(0)
cam.set(3,640) #width
cam.set(4,480)#height
cam.set(cv2.CAP_PROP_FPS,60)
segmentor= SelfiSegmentation()#defining the segmentation model default=1 for landscape mode
fps=cvzone.FPS()
imagebg=cv2.imread("bgimg/1.jpg")
#standard loop for webcam access
listImg=os.listdir('bgimg')
print(listImg)
imgList=[]
for x in listImg:
    img=cv2.imread(f'bgimg/{x}')
    imgList.append(img)
print(len(imgList))
imageindex = 0
while(True):
    success, image = cam.read()
    imageout=segmentor.removeBG(image,imgList[imageindex],threshold=0.7)#this variable to get the output

    
    cv2.imshow("liveout",imageout)
    key=cv2.waitKey(1)
    if key == ord('p'):
        if imageindex>0:
            imageindex -=1
    elif key == ord('n'):
        if imageindex<len(imgList)-1:
            imageindex +=1
    elif key == ord('q'):
        break




cam.release()
cv2.destroyAllWindows()






