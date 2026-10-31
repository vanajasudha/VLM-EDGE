import os, time, cv2

os.makedirs("test_frames", exist_ok=True)
for cam in range(1, 5):
    for k in range(3):
        cap = cv2.VideoCapture(f"rtsp://localhost:8554/cam{cam}")
        frame = None
        for _ in range(30):            # skip early frames while the decoder settles
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        if frame is not None:
            cv2.imwrite(f"test_frames/cam{cam}_{k}.jpg", frame)
            print(f"saved cam{cam}_{k}.jpg")
        time.sleep(3)                   # next grab is from a different moment of the loop




