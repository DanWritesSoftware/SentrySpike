import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2                                  # for OpenCV functions
import numpy as np                          # array math

from config import Config
CFG = Config()

class MotionDetect:
    '''
    Static utility class.
    '''

    def create_motion_detector(CFG):
        if CFG.motion_method == "diff":
            return DiffMotionDetector(CFG)
        elif CFG.motion_method == "bgsub":
            return BgSubMotionDetector(CFG)
        else:
            raise ValueError("Unkown motion_method in config file!")

    @staticmethod
    def clamp_bbox(x, y, w, h, W, H):
        '''
        Clamp a bounding box (x,y,w,h) so it stays inside an image of size (W,H).
        Ensures x/y are within bounds and w/h are at least 1 pixel and do not overflow
        '''
        # clamp left edge
        x = max(0, min(x, W - 1))
        # clamp top edge
        y = max(0, min(y, H - 1))
        # Width can't be <1 and cant extend past right edge
        w = max(1, min(w, W - x))
        # Height can't be <1 and can't extend past bottom edge
        h = max(1, min(h, H - y))
        return x, y, w, h

    @staticmethod
    def pad_bbox(x, y, w, h, W, H, pad=20):
        '''
        Expand a bbox by 'pad' pizels on all sides while staying inside (W,H).
        Returns a padded (x,y,w,h) bbox.
        '''
        # convert (x,y,w,h) into corners (x1,y1,x2,y2)
        x2 = x + w
        y2 = y + h
        # expand top left and bottom right outward
        x = max(0, x - pad)
        y = max(0, y - pad)
        # convert back to (x,y,w,h)
        x2 = min(W, x2 + pad)
        y2 = min(H, y2 + pad)
        return x, y, x2 - x, y2 - y

    @staticmethod
    def union_bbox(b1, b2):
        '''
        Returns the smallest bbox that contains both input bboxes.
        Handles None inputs so it can be used in incremental union building.
        '''
        if b1 is None:
            return b2
        if b2 is None:
            return b1
        # unpack both boxes
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        # compute corner coordinates
        a1, b1y, a2, b2y = x1, y1, x1 + w1, y1 + h1
        c1, d1, c2, d2 = x2, y2, x2 + w2, y2 + h2
        # union corners
        x = min(a1, c1)
        y = min(b1y, d1)
        X = max(a2, c2)
        Y = max(b2y, d2)
        # return union bbox
        return (x, y, X - x, Y - y)

    @staticmethod
    def bbox_from_two_frames(
        prev_bgr,
        curr_bgr,
        *,
        diff_thresh=25,
        min_area=1200,
        blur_ksize=9,
        morph_iters=2
        ):
        '''
        Compute a motin bounding box between two BGR frames using frame differencing.
        Steps: grayscale to optional blur to absdiff to threshold to morph cleanup to contours.
        Returns (x,y,w,h) for motion region or None if no motion above 'min_area'.
        '''
        # convert to grayscale
        prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        # blur
        if blur_ksize and blur_ksize > 1:
            prev = cv2.GaussianBlur(prev, (blur_ksize, blur_ksize), 0)
            curr = cv2.GaussianBlur(curr, (blur_ksize, blur_ksize), 0)

        # frame differencing 
        diff = cv2.absdiff(prev, curr)
        # threshold to binary mask (pixels above diff_thresh become 255(white), others 0(black))
        _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

       # morphology (clean up the mask)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=morph_iters)
        mask = cv2.erode(mask, kernel, iterations=morph_iters)
            # connects regions and fills holes, erosion shrinks back removing speckles

        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # build a bbox from contours
        bbox = None
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            bbox = MotionDetect.union_bbox(bbox, (x, y, w, h))

        return bbox

    @staticmethod
    def bbox_over_burst(
        frames_bgr,
        *,
        diff_thresh=25,
        min_area=1200,
        blur_ksize=9,
        morph_iters=2,
        pad=CFG.motion_pad
        ):
        '''
        Compute a single bbox that covers motin across an entire burst of frames.
        Unions per-pair motion bboxes, then clamps and pads the final bbox for safety/context.
        '''
    
        if frames_bgr is None or len(frames_bgr) < 2:
            return None

        # get image size from the first frame
        H, W = frames_bgr[0].shape[:2]
        # start with nothing
        union = None

        # compare frame i-1 to frame i
        for i in range(1, len(frames_bgr)):
            b = MotionDetect.bbox_from_two_frames(
                frames_bgr[i - 1],
                frames_bgr[i],
                diff_thresh=diff_thresh,
                min_area=min_area,
                blur_ksize=blur_ksize,
                morph_iters=morph_iters
                )
            # combine motion over time
            union = MotionDetect.union_bbox(union, b)
            
        if union is None:
            return None

        # clamp image to bounds, pad for context, return final bbox
        x, y, w, h = union
        x, y, w, h = MotionDetect.clamp_bbox(x, y, w, h, W, H)
        x, y, w, h = MotionDetect.pad_bbox(x, y, w, h, W, H, pad=pad)
        return (x, y, w, h)

    @staticmethod
    def crop_frames(frames_bgr, bbox):
        '''
        Crop every frame in a list using bbox (x,y,w,h). If bbox is None, returns frames unchanged.
        '''
        if bbox is None:
            return frames_bgr
        x, y, w, h = bbox
        return [f[y:y + h, x:x + w] for f in frames_bgr]

    @staticmethod
    def square_bbox_from_bbox(x, y, w, h, W, H, pad=20):
        '''
        Convert a rectangular bbox into an approximately square bbox centered on it.
        Expands to a side=max(w,h)+2*pad and clamps to image bounds. (W,H).
        '''
        # compute center
        cx = x + w / 2.0
        cy = y + h / 2.0
        # compute square side
        side = max(w, h) + 2 * pad

        # compute square corners around center
        x1 = int(round(cx - side / 2.0))
        y1 = int(round(cy - side / 2.0))
        x2 = int(round(cx + side / 2.0))
        y2 = int(round(cy + side / 2.0))

        # clamp to image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)

        # if clamping squishes side, accept clamped rectangle and pad later
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def square_bbox_in_bounds(x, y, w, h, W, H, pad=20):
        '''
        Make a square bbox around (x, y, w, h) while keeping it inside the image.
        Preserves square shape by shifting instead of squishing.
        Returns (sx, sy, side, side).
        '''
        # square size
        side = int(max(w, h) + 2 * pad)
        side = max(1, side)

        # if side bigger than image, cap it
        side = min(side, W, H)

        # center on original bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        sx = int(round(cx - side / 2.0))
        sy = int(round(cy - side / 2.0))

        # shift into bounds
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if sx + side > W:
            sx = W - side
        if sy + side > H:
            sy = H - side

        sx = max(0, sx)
        sy = max(0, sy)

        return sx, sy, side, side

    @staticmethod
    def bbox_from_two_frames_debug(
        prev_bgr,
        curr_bgr,
        *,
        diff_thresh=25,
        min_area=1200,
        blur_ksize=9,
        morph_iters=2
        ):
        '''
        Debug implementation of bbox_from_two_frames, for use in motion_capture_test.
        returns (bbox, dbg) where dbg contains intermediate images to save for debugging.

        dbg:
            prev_gray, curr_gray, prev_blur, curr_blur, diff, mask_raw, mask_morph, contours_vis, bbox_vis
        '''
        dbg = {}

        # convert to grayscale
        prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        dbg["prev_gray"] = prev_gray
        dbg["curr_gray"] = curr_gray

        # blur
        if blur_ksize and blur_ksize > 1:
            prev_blur = cv2.GaussianBlur(prev_gray, (blur_ksize, blur_ksize), 0)
            curr_blur = cv2.GaussianBlur(curr_gray, (blur_ksize, blur_ksize), 0)
            dbg["prev_blur"] = prev_blur
            dbg["curr_blur"] = curr_blur

        # frame differencing 
        diff = cv2.absdiff(prev_blur, curr_blur)
        dbg["diff"] = diff

        # threshold to binary mask (pixels above diff_thresh become 255(white), others 0(black))
        _, mask_raw = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
        dbg["mask_raw"] = mask_raw

       # morphology (clean up the mask)
        mask = mask_raw.copy()
        kernel = np.ones((3, 3), np.uint8)
        if morph_iters and morph_iters >0:
            mask = cv2.dilate(mask, kernel, iterations=morph_iters)
            mask = cv2.erode(mask, kernel, iterations=morph_iters)
            # connects regions and fills holes, erosion shrinks back removing speckles
        dbg["mask_morph"] = mask

        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_vis = curr_bgr.copy()
        cv2.drawContours(contours_vis, contours, -1, (0, 255, 255), 2)
        dbg["contours_vis"] = contours_vis

        if not contours:
            dbg["bbox_vis"] = curr_bgr.copy()
            return None, dbg

        # build a bbox from contours
        bbox = None
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            bbox = MotionDetect.union_bbox(bbox, (x, y, w, h))

        bbox_vis = curr_bgr.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(bbox_vis, (x,y), (x + w, y + h), (0, 255, 0), 2)
        dbg["bbox_vis"] = bbox_vis

        return bbox, dbg



class DiffMotionDetector:
    '''
    Uses two-frame diff pipeline
    Shared interface: detect(curr_bgr, prev_bgr) -> (bbox, dbg)
    '''
    def __init__(self, CFG):
        self.CFG = CFG

    def detect(self, curr_bgr, prev_bgr=None):
        if prev_bgr is None:
            # no previous frame yet, no motion, no debug
            return None, {}

        bbox, dbg = MotionDetect.bbox_from_two_frames_debug(
            prev_bgr,
            curr_bgr,
            diff_thresh=self.CFG.motion_difference_threshold,
            min_area=self.CFG.motion_min_area,
            blur_ksize=getattr(self.CFG, "motion_blur_ksize", 9),
            morph_iters=getattr(self.CFG, "motion_morph_iterations", 2)
            )
        return bbox, dbg

class BgSubMotionDetector:
    '''
    Background subtraction pipeline (MOG2/KNN).
    Shared interface: detect(curr_bgr, prev_bgr=None) -> (bbox,dbg)

    This holds state (the background model), so it must be created once and reused across frames.
    '''
    def __init__(self, CFG):
        self.CFG = CFG
        self.method = getattr(CFG, "bgsub_method", "MOG2").upper()

        self.history = getattr(CFG, "bgsub_history", 500)
        self.detect_shadows = getattr(CFG, "bgsub_detect_shadows", False)

        #Sensitivity knobs
        self.var_threshold = getattr(CFG, "bgsub_var_threshold", 16)        # MOG2
        self.dist2_threshold = getattr(CFG, "bgsub_dist2_threshold", 400.0) # KNN

        # Mask processing knobs
        self.blur_ksize = getattr(CFG, "motion_blur_ksize", 9)
        self.bin_thresh = getattr(CFG, "bgsub_bin_thresh", 200) # threshold fgmask -> binary
        self.open_iters = getattr(CFG, "bgsub_open_iters", 2)
        self.dilate_iters = getattr(CFG, "bgsub_dilate_iters", 2)

        self.min_area = getattr(CFG, "motion_min_area", 600)

        if self.method == "MOG2":
            self.bg = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows
                )
        elif self.method == "KNN":
            self.bg = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.dist2_threshold,
                detectShadows=self.detect_shadows
                )
        else:
            raise ValueError("bgsub_method must be 'MOG2' or 'KNN'")

        self.kernel = np.ones((3, 3), np.uint8)
        self.frame_count = 0
        self.warmup_frames = getattr(CFG, "bgsub_warmup_frames", 30)

        # freeze learning during active events
        self.freeze_during_event = getattr(CFG, "bgsub_freeze_during_event", True)

    def detect(self, curr_bgr, prev_bgr=None, *, event_active=False):
        self.frame_count += 1
        dbg = {}

        gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize and self.blur_ksize > 1:
            gray_blur = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        else:
            gray_blur = gray

        dbg["gray"] = gray
        dbg["gray_blur"] = gray_blur

        # Warmup - learn background but don't report motion yet
        if self.frame_count <= self.warmup_frames:
            _ = self.bg.apply(gray_blur, learningRate=-1)
            dbg["fgmask_raw"] = np.zeros_like(gray_blur, dtype=np.uint8)
            dbg["mask_bin"] = np.zeros_like(gray_blur, dtype=np.uint8)
            dbg["mask_morph"] = np.zeros_like(gray_blur, dtype=np.uint8)
            dbg["contours_vis"] = curr_bgr.copy()
            dbg["bbox_vis"] = curr_bgr.copy()
            return None, dbg

        # Freeze learning during event (prevents animal becoming background)
        learning_rate = -1
        if self.freeze_during_event and event_active:
            learning_rate = 0

        fgmask_raw = self.bg.apply(gray_blur, learningRate=learning_rate)
        dbg["fgmask_raw"] = fgmask_raw

        # Convert to clean binary mask (ignore shadows if enabled)
        _, mask_bin = cv2.threshold(fgmask_raw, self.bin_thresh, 255, cv2.THRESH_BINARY)
        dbg["mask_bin"] = mask_bin

        mask = mask_bin
        if self.open_iters and self.open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.open_iters)
        if self.dilate_iters and self.dilate_iters > 0:
            mask = cv2.dilate(mask, self.kernel, iterations=self.dilate_iters)
        dbg["mask_morph"] = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_vis = curr_bgr.copy()
        cv2.drawContours(contours_vis, contours, -1, (0, 255, 255), 2)
        dbg["contours_vis"] = contours_vis

        bbox = None
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            bbox = MotionDetect.union_bbox(bbox, (x, y, w, h))

        bbox_vis = curr_bgr.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(bbox_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        dbg["bbox_vis"] = bbox_vis

        return bbox, dbg