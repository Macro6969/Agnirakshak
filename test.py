from ultralytics import YOLO
import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0
from dashboard_push import send_frame


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO("weights/bestyolo.pt")

checkpoint = torch.load(
    "weights/best.pt",
    map_location=device,
    weights_only=False
)

classes = checkpoint["classes"]
image_size = checkpoint["image_size"]
mean = checkpoint["mean"]
std = checkpoint["std"]

cnn_model = efficientnet_b0()

cnn_model.classifier[1] = torch.nn.Linear(
    cnn_model.classifier[1].in_features,
    len(classes)
)

cnn_model.load_state_dict(checkpoint["model_state"])
cnn_model.to(device)
cnn_model.eval()

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean,
        std=std
    )
])

overlay_img = cv2.imread("overlay.png", cv2.IMREAD_UNCHANGED)


def cnn(frame, frameYOLO):

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = cnn_model(image)
        probabilities = torch.softmax(output, dim=1)

    class_id = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, class_id].item()

    label = f"{classes[class_id]}: {confidence:.2f}"
    if classes[class_id] in ["fire", "fire_smoke","smoke"] and confidence >= 0.70:
        frame = overlay(frameYOLO)
    else:
        frame = frameYOLO
    return frame


def overlay(frame):

    overlay = cv2.imread("overlay.png", cv2.IMREAD_UNCHANGED)

    width = 80
    height = 80

    overlay = cv2.resize(overlay, (width, height))

    x = frame.shape[1] - width - 20
    y = 20

    if overlay.shape[2] == 4:

        alpha = overlay[:, :, 3] / 255.0

        for c in range(3):
            frame[y:y+height, x:x+width, c] = (
                alpha * overlay[:, :, c]
                + (1 - alpha) * frame[y:y+height, x:x+width, c]
            )

    else:

        frame[y:y+height, x:x+width] = overlay

    return frame


def yolo(frame):

    results = yolo_model(frame, verbose=False)

    annotated = results[0].plot()

    return annotated


cap = cv2.VideoCapture("input.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break
    frame=cv2.resize(frame, (320, 240))
    frameYOLO = yolo(frame)

    frame = cnn(frame, frameYOLO)
    send_frame("EMBER", frame) 
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()