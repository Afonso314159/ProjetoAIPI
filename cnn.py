import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim, nn
from torch.utils.data import DataLoader

# Arquitetura do modelo
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# Definir dispositivo (GPU se disponível)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CACHED_MODEL = None

# Hiperparâmetros
IN_CHANNELS = 1
NUM_CLASSES = 10
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
NUM_EPOCHS = 5


# Carregar dados MNIST
def load_mnist_data(batch_size=BATCH_SIZE):
    train_dataset = datasets.MNIST(
        root="dataset/",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    test_dataset = datasets.MNIST(
        root="dataset/",
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Criar o modelo
def build_model(device=DEVICE):
    return CNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)


# Treinar a rede
def train_network(model, train_loader, device=DEVICE, num_epochs=NUM_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# Calcular accuracy
def check_accuracy(loader, model, device=DEVICE):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


# Carregar modelo existente ou treinar um novo
def load_or_train_model(model_path="mnist_cnn.pth", device=DEVICE):
    model = build_model(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    train_loader, test_loader = load_mnist_data()
    train_network(model, train_loader, device=device)
    torch.save(model.state_dict(), model_path)

    train_acc = check_accuracy(train_loader, model, device)
    test_acc = check_accuracy(test_loader, model, device)
    print(f"Accuracy treino: {train_acc * 100:.2f}%")
    print(f"Accuracy teste: {test_acc * 100:.2f}%")

    model.eval()
    return model


def get_model(model_path="mnist_cnn.pth", device=DEVICE):
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = load_or_train_model(model_path, device)
    return _CACHED_MODEL


# Processamento de imagem e previsão

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def _resize_with_padding(gray_image, size=28, padding_ratio=0.2):
    """
    Redimensiona o dígito e centra-o numa imagem 28x28 com margem.
    """
    h, w = gray_image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Imagem vazia")

    inner_size = int(size * (1 - 2 * padding_ratio))
    scale = min(inner_size / w, inner_size / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_digit = cv2.resize(gray_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_digit

    return padded


def preprocess_number_region(number_region):
    resized = _resize_with_padding(number_region, size=28, padding_ratio=0.15)

    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - MNIST_MEAN) / MNIST_STD

    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor


def predict_digit_with_confidence(number_region, model, device=DEVICE):
    model.eval()
    tensor = preprocess_number_region(number_region).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    return int(prediction.item()), float(confidence.item())


def extract_one_or_two_digits(tile, min_area=15):
    ink = np.count_nonzero(tile)
    if ink < 30:
        return []

    tile_clean = tile.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (tile_clean > 0).astype(np.uint8), connectivity=8
    )

    comps = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        aspect = w / max(h, 1)
        if aspect > 5:
            continue
        comps.append((x, y, w, h, area))

    if not comps:
        if ink > 30:
            ys, xs = np.where(tile > 0)
            if len(ys) > 0:
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                return [tile[y1:y2 + 1, x1:x2 + 1]]
        return []

    comps.sort(key=lambda c: c[0])

    if len(comps) >= 2:
        comps = sorted(comps, key=lambda c: c[4], reverse=True)[:2]
        comps.sort(key=lambda c: c[0])

        rois = []
        for x, y, w, h, _ in comps:
            rois.append(tile[y:y + h, x:x + w])
        return rois

    ys, xs = np.where(tile > 0)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return [tile[y1:y2 + 1, x1:x2 + 1]]


def predict_number_with_confidence(number_region, model, device=DEVICE, min_confidence=0.5):
    if number_region is None or number_region.size == 0:
        return 0, 0.0

    ink = np.count_nonzero(number_region)
    if ink < 20:
        return 0, 0.0

    digit_regions = extract_one_or_two_digits(number_region)

    if not digit_regions and ink > 30:
        ys, xs = np.where(number_region > 0)
        if len(ys) > 0:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            digit_regions = [number_region[y1:y2 + 1, x1:x2 + 1]]

    if not digit_regions:
        return 0, 0.0

    digits = []
    confidences = []

    for region in digit_regions:
        if region is None or region.size == 0 or np.sum(region > 0) < 10:
            continue

        digit, confidence = predict_digit_with_confidence(region, model, device)

        if digit == 0:
            tensor = preprocess_number_region(region).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                top2_conf, top2_pred = torch.topk(probs, 2, dim=1)

                second_digit = int(top2_pred[0, 1].item())
                second_conf = float(top2_conf[0, 1].item())

                if second_digit != 0 and second_conf > 0.1:
                    digit = second_digit
                    confidence = second_conf

        digits.append(str(digit))
        confidences.append(confidence)

    if not digits:
        return 0, 0.0

    value = int("".join(digits))
    overall_confidence = min(confidences) if confidences else 0.0

    if len(digits) == 2 and not (10 <= value <= 15):
        single_digit, single_conf = predict_digit_with_confidence(number_region, model, device)
        return single_digit, single_conf * 0.9

    if not (1 <= value <= 15):
        return value, overall_confidence * 0.1

    return value, overall_confidence


if __name__ == "__main__":
    trained_model = load_or_train_model()
    print("Modelo pronto")
