# Enhanced Image Processor

A Python library for image enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization) with support for multiple color spaces, automatic parameter optimization, and multi-threaded batch processing.

## Features

- **Advanced CLAHE Implementation**
  - Support for multiple color spaces (LAB, HSV, YUV, YCrCb, HLS, XYZ, LUV)
  - Automatic parameter optimization based on image characteristics
  - Handles various image formats and bit depths

- **Efficient Processing**
  - Multi-threaded batch processing
  - Progress tracking with progress bar
  - Comprehensive error handling

- **Supported Image Formats**
  - Common formats: PNG, JPG, JPEG, TIFF, BMP
  - Web formats: WebP
  - Scientific formats: PPM, PGM, PBM
  - High dynamic range: EXR, HDR, PIC
  - Others: SR, RAS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MichailSemoglou/enhanced-image-processor.git
cd enhanced-image-processor
```

2. Install required packages:
```bash
pip install opencv-python numpy tqdm
```

## Usage

### Basic Usage with Auto-Optimization

```python
from image_processor import ImageProcessor

# Create processor with auto-optimization
processor = ImageProcessor(auto_optimize=True)

# Process and save image with optimized parameters
enhanced_image = processor.enhance_image("input.jpg")
if enhanced_image is not None:
    processor.save_image(enhanced_image, "enhanced_input.jpg")
```

### Custom Output Directory

```python
# Process and save image in a specific directory
enhanced_image = processor.enhance_image("input.jpg")
if enhanced_image is not None:
    processor.save_image(enhanced_image, "output_folder/enhanced_input.jpg")
```

### Batch Processing

```python
# Process multiple images with progress tracking
results = processor.batch_process(
    input_dir="input_folder",
    output_dir="output_folder",
    max_workers=4,
    show_progress=True
)

# Check results
for filename, status in results:
    print(f"{filename}: {status}")
```

### Custom Color Space

```python
# Use specific color space
processor = ImageProcessor(
    clip_limit=2.0,
    tile_size=8,
    color_space='HSV',
    normalize_output=True
)

# Process and save image
enhanced_image = processor.enhance_image("input.jpg")
if enhanced_image is not None:
    processor.save_image(enhanced_image, "enhanced_hsv_input.jpg")
```

## Parameters

- `clip_limit` (float, default=2.0): Threshold for contrast limiting
- `tile_size` (int, default=8): Size of grid for histogram equalization
- `color_space` (str, default='LAB'): Color space for enhancement
- `normalize_output` (bool, default=True): Whether to normalize output values
- `auto_optimize` (bool, default=False): Enable automatic parameter optimization

## Save Image Parameters

- `image` (np.ndarray): The enhanced image to save
- `output_path` (str or Path): Where to save the image
- `create_dirs` (bool, default=True): Automatically create output directories if they don't exist

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- tqdm

## Error Handling

The library includes comprehensive error handling for:
- Invalid file paths
- Unsupported image formats
- Image loading failures
- Processing errors
- File saving issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License – see the LICENSE file for details.
