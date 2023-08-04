
import numpy as np
import cv2


#cap = cv2.VideoCapture(0) # Uncomment it if you want stream video from a camera 
cap = cv2.VideoCapture('video.avi')



# We get some info about video 
fps = cap.get(cv2.CAP_PROP_FPS)
totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
durationInSeconds = totalNoFrames // fps
print("Video Duration In Seconds:", durationInSeconds, "s")



# The next line for saving the output
# saved = cv2.VideoWriter('output4.avi', -1, 10.0, (640,480))

# Take_first_frame_to_compare_it_with_next
ret,pastframe = cap.read()
#saved.write(pastframe)
pastframe = cv2.cvtColor(pastframe, cv2.COLOR_BGR2GRAY)


while cap.isOpened():  
  
  ret,presentframe = cap.read()
  if not ret:
      break
  
  img = presentframe 
  #saved.write(presentframe)
  presentframe = cv2.cvtColor(presentframe, cv2.COLOR_BGR2GRAY)
  
  # Here , we subtract two frames and handle the result into another 2D array
  # the idea of this step is to discover the difference in pixels between two consecutive frames 
  # during showing out result , if the "out" window is nearly black so it is no movement 
  # if the "out" window more white(gray) so there is a movement
  out = cv2.absdiff(pastframe, presentframe)
  
  
  
  # Apply noise reduction using Median or Gaussian 
  out = cv2.GaussianBlur(out, (7, 7), 0)
  
  # Threshold the Blurred image to label the difference (movement) as a white pixels   
  (T,thresholded) = cv2.threshold(out,10,255,cv2.THRESH_BINARY)
  
  # Apply dilation opertation to increase the region of white pixels 
  kernal = np.ones((3,3), np.uint8)
  dilation = cv2.dilate(thresholded, kernal, iterations=7)
  
  # Last, we find the contours of white regions 
  (cnts, _) = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  
  
  for contour in cnts :
      (x,y,w,h) = cv2.boundingRect(contour)
      cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0))
  
  
  
  
  cv2.imshow("abs_difference",out)
  cv2.imshow("thsh",thresholded)
  cv2.imshow("dilation",dilation)
  cv2.imshow("img",img)
  
  pastframe = presentframe
  
  # you can change the number of milliseconds in waitKay to change the speed of video streamig 
  if cv2.waitKey(30) == ord('q'):
     break
 
# When everything done, release the capture
cv2.destroyAllWindows()
cap.release()
#saved.release()
