import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Same camera selection logic as main code ---
CANDIDATE_INDICES = [0, 1, 2, 3, 4]
RESOLUTION = (640, 480)  # change to (640, 480) if that's what you use in main


def open_camera_with_settings(index, width, height):
    """Same style as main code: MJPG, resolution, low buffer."""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)  # adjust if you want

    if not cap.isOpened():
        cap.release()
        return None

    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        return cap
    cap.release()
    return None


def find_active_camera(width, height):
    for index in CANDIDATE_INDICES:
        logging.info(f"[*] Trying /dev/video{index} @ {width}x{height} ...")
        cap = open_camera_with_settings(index, width, height)
        if cap:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f" [SUCCESS] Connected to video{index}, actual: {actual_w}x{actual_h}")
            return cap, index
    return None, None


# --- Initialize camera using same logic as main ---
cap, cam_index = find_active_camera(RESOLUTION[0], RESOLUTION[1])
if not cap:
    print("Cannot open camera on any index in", CANDIDATE_INDICES)
    exit()

cv2.namedWindow('HSV Adjustments')

# Trackbars for tuning (single HSV range)
cv2.createTrackbar('H Lower', 'HSV Adjustments', 160, 179, lambda x: None)
cv2.createTrackbar('S Lower', 'HSV Adjustments', 100, 255, lambda x: None)
cv2.createTrackbar('V Lower', 'HSV Adjustments', 49, 255, lambda x: None)
cv2.createTrackbar('H Upper', 'HSV Adjustments', 179, 179, lambda x: None)
cv2.createTrackbar('S Upper', 'HSV Adjustments', 255, 255, lambda x: None)
cv2.createTrackbar('V Upper', 'HSV Adjustments', 255, 255, lambda x: None)
cv2.createTrackbar('Area Min', 'HSV Adjustments', 490, 10000, lambda x: None)

print("HSV Adjustment Program - SAME pipeline as main code")
print("Press 'q' to quit")
print("Press 'p' to print current HSV range & area threshold")

kernel3 = np.ones((3, 3), np.uint8)  # same as main

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab frame")
        break

    # Read trackbar values
    h_low = cv2.getTrackbarPos('H Lower', 'HSV Adjustments')
    s_low = cv2.getTrackbarPos('S Lower', 'HSV Adjustments')
    v_low = cv2.getTrackbarPos('V Lower', 'HSV Adjustments')
    h_high = cv2.getTrackbarPos('H Upper', 'HSV Adjustments')
    s_high = cv2.getTrackbarPos('S Upper', 'HSV Adjustments')
    v_high = cv2.getTrackbarPos('V Upper', 'HSV Adjustments')
    area_min = cv2.getTrackbarPos('Area Min', 'HSV Adjustments')

    lower_bound = np.array([h_low, s_low, v_low])
    upper_bound = np.array([h_high, s_high, v_high])

    # --- EXACTLY LIKE MAIN: BGR -> HSV, then mask, then OPEN+CLOSE ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Crosshair (optional)
    h, w = frame.shape[:2]
    cv2.line(res, (w // 3, 0), (w // 3, h), (255, 0, 0), 1)
    cv2.line(res, (2 * w // 3, 0), (2 * w // 3, h), (255, 0, 0), 1)
    cv2.line(res, (0, h // 3), (w, h // 3), (255, 0, 0), 1)
    cv2.line(res, (0, 2 * h // 3), (w, 2 * h // 3), (255, 0, 0), 1)

    # Contours (same logic type as main)
    contours_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_result) == 3:
        contours = contours_result[1]
    else:
        contours = contours_result[0]

    largest_hsv_text = None

    for c in contours:
        area = cv2.contourArea(c)
        if area > area_min:
            cv2.drawContours(res, [c], -1, (255, 255, 255), 2)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(res, (cx, cy), 5, (255, 0, 0), -1)

                # HSV at contour center
                if 0 <= cx < w and 0 <= cy < h:
                    h_val, s_val, v_val = hsv[cy, cx]
                    hsv_text = f"H:{int(h_val)} S:{int(s_val)} V:{int(v_val)}"
                    largest_hsv_text = hsv_text
                    cv2.putText(res, hsv_text, (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                coord_text = f"{cx}, {cy}"
                cv2.putText(res, coord_text, (cx + 10, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Status text
    cv2.putText(res, f"Area Min: {area_min}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(res, f"H: {h_low}-{h_high}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(res, f"S: {s_low}-{s_high}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(res, f"V: {v_low}-{v_high}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', res)

    if largest_hsv_text is not None:
        print("\rCurrent HSV at object center: " + largest_hsv_text, end="")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        print("\n--- Current ranges & area ---")
        print("Lower bound: [" + str(h_low) + ", " + str(s_low) + ", " + str(v_low) + "]")
        print("Upper bound: [" + str(h_high) + ", " + str(s_high) + ", " + str(v_high) + "]")
        print("Area threshold: " + str(area_min))

cap.release()
cv2.destroyAllWindows()