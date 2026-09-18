import cv2
from cvzone.HandTrackingModule import HandDetector

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)

# Inisialisasi Detektor Tangan (Maksimal 1 tangan, tingkat deteksi 80%)
detector = HandDetector(maxHands=1, detectionCon=0.8)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip kamera horizontal (seperti cermin)
    frame = cv2.flip(frame, 1)

    # Cari dan deteksi tangan
    hands, frame = detector.findHands(frame, draw=True)

    should_blur = False

    if hands:
        hand = hands[0]
        # Ambil status jari: [Jempol, Telunjuk, Tengah, Manis, Kelingking]
        # Nilai 1 = Terangkat, 0 = Tertutup
        fingers = detector.fingersUp(hand)

        # Kondisi: HANYA Telunjuk (index 1) dan Tengah (index 2) yang terangkat -> [0, 1, 1, 0, 0]
        if fingers == [0, 1, 1, 0, 0]:
            should_blur = True

    # Jika gesture 2 jari terdeteksi, berikan efek Gaussian Blur
    if should_blur:
        frame = cv2.GaussianBlur(frame, (55, 55), 0)
        cv2.putText(frame, "STATUS: BLUR ACTIVE", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "STATUS: CLEAR", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan Kamera
    cv2.imshow("Gesture Blur Camera", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()