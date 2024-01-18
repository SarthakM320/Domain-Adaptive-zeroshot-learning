from torchvision import transforms


def _convert_image_to_rgb(image):
    return image.convert("RGB")

transforms_cityscape = transforms.Compose([
    transforms.Resize((224,224)),
    _convert_image_to_rgb,
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])