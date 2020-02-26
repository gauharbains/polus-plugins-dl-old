import cv2

img=cv2.imread('/home/gauhar/TopNN_plugin/dsb2018_topcoders/x001_y001_c001.ome.tif')
cv2.imwrite('new.png',img )
print("hello")