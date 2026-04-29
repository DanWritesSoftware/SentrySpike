from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    '''
    Camera / Input
    '''
    camera_index: int = 0                       # OpenCV camera index (0 = default webcam, 1+ = other cameras)
    image_size: int = 224                       # Input resolution expected by the Akida/ImageNet model (square)

    '''
    Motion watcher (always-on loop)
    '''
    
    motion_confirm_frames: int = 3              # How many consecutive motion hits required

    '''
    Burst capture (event confirmation)
    '''

    burst_frames: int = 15                      # Number of frames to capture in a burst when an event is triggered
    burst_sleep_seconds: float = 0.08           # Delay between frames during burst capture (controls burst duration)
    burst_topk: int = 5                         # Number of top predictions to compute from the aggregated burst output
    burst_min_frame_confidence: float = 0.10    # Optional per-frame confidence floor; frames below this can be ignored

    '''
    Burst decision thresholds
    '''
    gate_threshold: float = 3.5                 # Akida potential: scores above this are classified as animal
    stability_threshold: float = 0.60           # Minimum stability score (agreeing frames / total frames) required to accept a burst
    heavy_confidence_threshold: float = 0.40    # Minimum averaged softmax confidence for the heavy model to commit to a species label; below this falls back to "animal"
    cooldown_seconds: float = 10.0              # Cooldown period after a successful event to prevent retriggering on the same animal

    '''
    Motion detection parameters
    '''

    motion_method = "bgsub"                      # "diff" for frame diffrenceing, "bgsub" for background subtraction
    bgsub_method = "MOG2"                       # "MOG2" or "KNN"

    motion_difference_threshold: int = 7        # Pixel intensity difference threshold for motion detection
    motion_min_area: int = 80                   # Minimum contour area (in pixels) to count as real motion (filters noise, leaves)
    motion_blur_ksize: int = 5                  # Gaussian blur kernel size for motion detection (must be odd; higher = smoother)
    motion_morph_iterations: int = 1            # Number of dilation/erosion iterations to clean up the motion mask
    motion_pad: int = 15                        # Extra padding (pixels) added around motion bounding box for context

    bgsub_warmup_frames: int = 30               # Frames to learn background before reporting motion
    bgsub_history: int = 500                    # Number of frames used to build the background model
    bgsub_var_threshold: int = 16               # MOG2 variance threshold; lower = more sensitive
    bgsub_dist2_threshold: float = 400.0        # KNN distance threshold (unused when method is MOG2)
    bgsub_detect_shadows: bool = False          # Whether to detect and mark shadows (127) in the foreground mask
    bgsub_bin_thresh: int = 200                 # Threshold applied to fgmask to produce binary mask; excludes shadows
    bgsub_open_iters: int = 2                   # Morphological open iterations to remove speckle noise
    bgsub_dilate_iters: int = 2                 # Dilation iterations to connect nearby foreground regions
    bgsub_freeze_during_event: bool = True      # Freeze background learning while motion is active

    '''
    Saving / Output
    '''
    
    save_directory: str = "captures"            # Root directory where burst images and metadata are saved

    '''
    Database
    '''

    database_path: str = "SentrySpike_Events.db"

    '''
    Inference
    '''

    gate_model_path: str = "SentrySpike_Inference/gate_model/reconverted/gate_model_akida_reconverted.fbz"
    heavy_model_path: str = "SentrySpike_Inference/heavy_model/reconverted/heavy_model_akida_reconverted.fbz"