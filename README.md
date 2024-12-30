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

### Basic Usage

```python
from image_processor import ImageProcessor

# Create processor instance
processor = ImageProcessor()

# Process single image
processor.enhance_image("input.jpg")
```

### With Automatic Parameter Optimization

```python
# Create processor with auto-optimization
processor = ImageProcessor(auto_optimize=True)

# Process image with optimized parameters
processor.enhance_image("input.jpg")
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
```

## Parameters

- `clip_limit` (float, default=2.0): Threshold for contrast limiting
- `tile_size` (int, default=8): Size of grid for histogram equalization
- `color_space` (str, default='LAB'): Color space for enhancement
- `normalize_output` (bool, default=True): Whether to normalize output values
- `auto_optimize` (bool, default=False): Enable automatic parameter optimization

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
