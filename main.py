from cvzone.FaceDetectionModule import FaceDetector
import cv2
from cvzone import FPS
from cvzone.FaceMeshModule import FaceMeshDetector

# Initialisation
cap = cv2.VideoCapture("test_videos/video_4.mp4")
detector = FaceDetector()
fps_reader = FPS()
face_mesh_detector = FaceMeshDetector()

# Changing the DrawSpecifications
drawSpecifications = face_mesh_detector.drawSpec
drawSpecifications.color = (255, 0, 0)  # BGR
drawSpecifications.circle_radius = 1
drawSpecifications.thickness = 1

while True:
    success, img = cap.read()

    # To avoid an error at end of video
    if not success:
        cap.release()  # Release the capture
        break

    img, bboxs = detector.findFaces(img)
    img, faces_info = face_mesh_detector.findFaceMesh(img)

    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        center = bboxs[0]["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

    # Update of FPS(update) and show in Window
    fps_reader.update(img, scale=2, thickness=2)
    cv2.imshow("Image", img)

    # If 'q' is pressed, exit/quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

# Destroy all the opened windows if any..
cv2.destroyAllWindows()
