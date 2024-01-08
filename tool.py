import cv2
import numpy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def rotate_and_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_cnt = cnt
    x,y,w,h = cv2.boundingRect(max_cnt)
    
    rect = cv2.minAreaRect(max_cnt)
    ((cx,cy),(cw,ch),angle) = rect
    
    M = cv2.getRotationMatrix2D((cx,cy), angle, 1)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_cnt = cnt
    x,y,w,h = cv2.boundingRect(max_cnt)
    cropped = rotated[y:y+h, x:x+w]
    return cropped


def convert_to_list(l):
    for i in range(len(l) - 1):
        for j in range(i,len(l)):
            if l[i][2] > l[j][2]:
                l[i],l[j] = l[j],l[i]

    print(l)
    for i in range(4-1):
        for j in range(i+1,4):
            if l[i][1] > l[j][1]:
                l[i],l[j] = l[j],l[i]

    print(l)
    for i in range(4,len(l)-1):
        for j in range(i+1,len(l)):
            if l[i][1] > l[j][1]:
                l[i],l[j] = l[j],l[i]

    rs = ""
    for i in l:
        rs += i[0]

    return rs


if __name__ == "__main__":
    image = cv2.imread("/Users/trHien/Python/MyProjects/YoloDetect/myenv/yolo-detect/plateImage.png")

    cropped = rotate_and_crop(image)


    # Làm mịn ảnh để giảm nền mờ
    image = cv2.GaussianBlur(cropped, (15, 15), 0)



    cv2.imshow("original", image)   
    cv2.imshow("cropped", cropped)


    # cv2.imshow("thresholded", thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()