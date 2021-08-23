#import necessary packages
import numpy as np
import cv2

def region_of_interest(edges):
    """
    focus on a small region of the screen :
    draw a polygon which presents the borders of this region 
    """
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (width/6, height),
        (width/4, 8*height/18),
        (3*width/4, 8*height/18),
        (5*width/6, height ),      
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def detect_line_segments(cropped_edges):
    """
    tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    """
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 100  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=30, maxLineGap=25)
    return line_segments

def draw_the_lines(img, lines):
    """
    after detecting the edges we draw green lines on them
    """
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=8)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result.avi',fourcc, 20.0, (640,480))


while(cap.isOpened()):
    # do the following instructions while the video stream is still on
    ret, frame = cap.read()
    if ret==True:
        height = frame.shape[0]
        width = frame.shape[1]
        # to make things easier we convert to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # use a canny filter to transform the image into edges
        edges = cv2.Canny(gray,100,200)
        # focus on the region of interest
        roi = region_of_interest(edges)
        # detect the lines
        lines = detect_line_segments(roi)
        # if lines detected , draw the lines
        if (not lines is None):
            frame = draw_the_lines(frame, lines)
        cv2.imshow('frame',frame)
        cv2.imshow('edges',roi)
        
        out.write(frame)
        # to stop the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
