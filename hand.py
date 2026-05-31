import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_action_time = 0


def fingers_up(landmarks):
    fingers = []

    # Thumb
    fingers.append(1 if landmarks[4][1] > landmarks[3][1] else 0)

    # Other fingers
    tips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]

    for tip, joint in zip(tips, joints):
        fingers.append(1 if landmarks[tip][2] < landmarks[joint][2] else 0)

    return fingers


while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_img)

    landmarks = []
    fingers = None

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            # build landmarks
            for id, lm in enumerate(handLms.landmark):

                h, w, c = img.shape

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                landmarks.append((id, cx, cy))

            # draw hand
            mp_draw.draw_landmarks(
                img,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )

        # only compute if landmarks exist
        if len(landmarks) > 0:
            fingers = fingers_up(landmarks)

            current_time = time.time()

            # ✨ 1. Index finger → Screenshot
            if landmarks[8][2] < landmarks[6][2]:

                cv2.putText(img, "INDEX SCREENSHOT", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if current_time - last_action_time > 3:
                    pyautogui.screenshot("shot.png")
                    print("📸 Screenshot Taken")
                    last_action_time = current_time

            # ✨ 2. Two fingers → Next tab
            elif fingers == [0, 1, 1, 0, 0]:

                cv2.putText(img, "NEXT TAB", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if current_time - last_action_time > 1:
                    pyautogui.hotkey("ctrl", "tab")
                    last_action_time = current_time

            # ✨ 3. Fist → Play/Pause
            elif fingers == [0, 0, 0, 0, 0]:

                cv2.putText(img, "PLAY/PAUSE", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if current_time - last_action_time > 1:
                    pyautogui.press("playpause")
                    last_action_time = current_time

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()