import cv2
import mediapipe as mp

hModel = mp.solutions.hands
hands = hModel.Hands(max_num_hands=2)
draw = mp.solutions.drawing_utils
cam = cv2.VideoCapture(0)

while True:
    success,frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    mp_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res = hands.process(mp_rgb)
    fingers_count =0

    if res.multi_hand_landmarks:
        for hLandmark,handSide in zip(res.multi_hand_landmarks, res.multi_handedness):
            side = handSide.classification[0].label
            draw.draw_landmarks(frame,hLandmark,hModel.HAND_CONNECTIONS)
            landmarks = []
            for id,lm in enumerate(hLandmark.landmark):
                h,w,c = frame.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                landmarks.append((cx,cy))

            tips=[4,8,12,16,20]
            if side == 'Right':
                if landmarks[tips[0]][0] < landmarks[tips[0]-1][0]:
                    fingers_count+=1
            else:
                if landmarks[tips[0]][0] > landmarks[tips[0]-1][0]:
                    fingers_count+=1

            for id in range(1,5):
                if landmarks[tips[id]][1] < landmarks[tips[id]-2][1]:
                    fingers_count+=1

            cv2.putText(frame, side, (landmarks[0][0]-30, landmarks[0][1]-30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.rectangle(frame,(0,0),(580,100),(0,0,0),-1)
    cv2.putText(frame,f'Number of fingers : {fingers_count}',(10,70),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)
    cv2.imshow("Fingers Count Visualiser",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()