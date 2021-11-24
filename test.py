import cv2
import os,random
bg = os.listdir('bgimg')
x=random.choice(bg)
#for image
editImg = os.listdir('image')
y=random.choice(editImg) 
path=x
print(path)
# img=cv2.imread(r'bgimg/5.jpg')
# cv2.imshow('sample image',img)
 
# cv2.waitKey(0) # waits until a key is pressed
# cv2.destroyAllWindows()