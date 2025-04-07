import cv2
import numpy as np

#Params
TEMPLATE_PATH = "488568168_1029561008533084_6965510402974817025_n.jpg"  # pic
DISTANCE_THRESHOLD = 50  # threshold for match 
MIN_GOOD_MATCHES = 50  # minimax for match
SHOW_MATCHES = True  # showpics

template = cv2.imread(TEMPLATE_PATH)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#ORB detector
orb = cv2.ORB_create()


kp_template, des_template = orb.detectAndCompute(gray_template, None)

cap = cv2.VideoCapture(0)#opencam 1,2,3
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
    matches = bf.match(des_template, des_frame)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < DISTANCE_THRESHOLD]
    if len(good_matches) > MIN_GOOD_MATCHES:
        print("found")
    else:
        print("not found")

    if SHOW_MATCHES:
        img_matches = cv2.drawMatches(template, kp_template, frame, kp_frame, good_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Feature Matching with ORB", img_matches)

    if cv2.waitKey(1) & 0xFF == ord('q'): # ปิดโปรแกรมกด q 
        break

cap.release()
cv2.destroyAllWindows()


