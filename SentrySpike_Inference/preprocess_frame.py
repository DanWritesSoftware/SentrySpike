import cv2
import numpy as np

def center_crop_to_square(img_bgr: np.ndarray) -> np.ndarray:
    '''
    Center-crops an OpenCV BGR frame to a square, perserving aspect ratio by cropping.
    Used before resizing to the model's input size.
    '''
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img_bgr[y0:y0 + side, x0:x0 + side]

def preprocess_frame_for_akidanet(frame_bgr: np.ndarray, image_size: int) -> np.array:
    '''
    Converts a webcam BGR frame into a batched RGB tensor for inference:
    center-crop to square, resize to (image_size, image_size), convert BGR->RGB,
    keep uint8 (0..255), and add batch dimension to produce shape (1, H, W, 3).
    '''
    sq = center_crop_to_square(frame_bgr)
    resized = cv2.resize(sq, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # keep pixel values as uint8
    x = rgb.astype(np.uint8)

    # add a batch dimension before returning for most inference APIs
    return np.expand_dims(x, axis=0)