"""
Enhanced Image Processor using CLAHE (Contrast Limited Adaptive Histogram Equalization).

This module provides functionality for enhancing image contrast using the CLAHE algorithm,
with support for both single image and batch processing operations.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    A class for processing images using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    This class provides methods for enhancing image contrast using the CLAHE algorithm,
    with support for both single image and batch processing.
    """
    
    def __init__(self, 
                 clip_limit: float = 2.0, 
                 tile_size: int = 8,
                 color_space: str = 'LAB',
                 normalize_output: bool = True,
                 auto_optimize: bool = False):
        """
        Initialize the ImageProcessor with CLAHE parameters.

        Args:
            clip_limit (float): Threshold for contrast limiting. Default is 2.0.
            tile_size (int): Size of grid for histogram equalization. Default is 8.
            color_space (str): Color space for enhancement. 
                         Options: 'LAB', 'HSV', 'YUV', 'YCrCb', 'HLS', 'XYZ', 'LUV'.
            normalize_output (bool): Whether to normalize the output image values.
            auto_optimize (bool): Whether to automatically optimize CLAHE parameters.
        
        Raises:
            ValueError: If parameters are invalid.
        """
        if clip_limit <= 0:
            raise ValueError("clip_limit must be positive")
        if tile_size < 2:
            raise ValueError("tile_size must be at least 2")
        if color_space not in ['LAB', 'HSV', 'YUV', 'YCrCb', 'HLS', 'XYZ', 'LUV']:
            raise ValueError("color_space must be one of: LAB, HSV, YUV, YCrCb, HLS, XYZ, LUV")
            
        # Initialize instance variables
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.color_space = color_space
        self.normalize_output = normalize_output
        self.auto_optimize = auto_optimize
        
        # Supported image formats
        self.supported_formats = {
            '.png', '.jpg', '.jpeg', '.tiff', '.bmp',
            '.webp', '.ppm', '.pgm', '.pbm', '.sr', '.ras',
            '.exr', '.hdr', '.pic'  # High dynamic range formats
        }
        
        # Color space conversion mappings
        self._color_space_conversions = {
            'LAB': (cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR),
            'HSV': (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR),
            'YUV': (cv2.COLOR_BGR2YUV, cv2.COLOR_YUV2BGR),
            'YCrCb': (cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR),
            'HLS': (cv2.COLOR_BGR2HLS, cv2.COLOR_HLS2BGR),
            'XYZ': (cv2.COLOR_BGR2XYZ, cv2.COLOR_XYZ2BGR),
            'LUV': (cv2.COLOR_BGR2LUV, cv2.COLOR_LUV2BGR)
        }
        
        # Auto-optimization parameters
        self._optimization_metrics = {
            'contrast': self._calculate_contrast,
            'entropy': self._calculate_entropy,
            'brightness': self._calculate_brightness
        }
        
        # Channel indices for different color spaces
        self._luminance_channel_index = {
            'LAB': 0,  # L channel
            'HSV': 2,  # V channel
            'YUV': 0,  # Y channel
            'YCrCb': 0,  # Y channel
            'HLS': 1,  # L channel
            'XYZ': 1,  # Y channel
            'LUV': 0   # L channel
        }

    # Public methods
    def enhance_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Apply CLAHE enhancement to a single image.
        
        Args:
            image_path (Union[str, Path]): Path to the input image.
            
        Returns:
            Optional[np.ndarray]: Enhanced image if successful, None otherwise.
            
        Raises:
            ValueError: If the image path is invalid or file format is unsupported.
        """
        image_path = Path(image_path)
        
        if not self._is_valid_image_path(image_path):
            raise ValueError(f"Invalid image path or unsupported format: {image_path}")
            
        try:
            # Read image and handle potential errors
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Handle different bit depths
            if img.dtype != np.uint8:
                # Normalize 16-bit or floating-point images to 8-bit
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img = img.astype(np.uint8)
            
            # Optimize parameters if requested
            if self.auto_optimize:
                self.clip_limit, self.tile_size = self._optimize_parameters(img)
                logger.info(f"Using optimized parameters: clip_limit={self.clip_limit:.1f}, "
                          f"tile_size={self.tile_size}")
            
            # Convert to desired color space
            to_color_space, from_color_space = self._color_space_conversions[self.color_space]
            converted = cv2.cvtColor(img, to_color_space)
            channels = list(cv2.split(converted))
            
            # Get luminance channel index for the current color space
            lum_idx = self._luminance_channel_index[self.color_space]
            
            # Apply CLAHE to the luminance channel
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=(self.tile_size, self.tile_size)
            )
            channels[lum_idx] = clahe.apply(channels[lum_idx])
            
            # Merge channels and convert back to BGR
            enhanced_color = cv2.merge(channels)
            enhanced_image = cv2.cvtColor(enhanced_color, from_color_space)
            
            # Normalize output if requested
            if self.normalize_output:
                enhanced_image = cv2.normalize(
                    enhanced_image, 
                    None, 
                    alpha=0, 
                    beta=255, 
                    norm_type=cv2.NORM_MINMAX
                )
            
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None

    def save_image(self,
                  image: np.ndarray,
                  output_path: Union[str, Path],
                  create_dirs: bool = True) -> bool:
        """
        Save the processed image to the specified path.
        
        Args:
            image (np.ndarray): Image to save.
            output_path (Union[str, Path]): Path where the image should be saved.
            create_dirs (bool): Create output directories if they don't exist.
            
        Returns:
            bool: True if save was successful, False otherwise.
        """
        output_path = Path(output_path)
        
        try:
            if create_dirs:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
            cv2.imwrite(str(output_path), image)
            return True
            
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {str(e)}")
            return False

    def batch_process(self, 
                     input_dir: Union[str, Path], 
                     output_dir: Union[str, Path],
                     max_workers: int = 4,
                     show_progress: bool = True) -> List[Tuple[str, str]]:
        """
        Process multiple images in a directory using CLAHE with multi-threading.
        
        Args:
            input_dir (Union[str, Path]): Directory containing input images.
            output_dir (Union[str, Path]): Directory for saving processed images.
            max_workers (int): Maximum number of concurrent threads.
            show_progress (bool): Whether to show progress bar.
            
        Returns:
            List[Tuple[str, str]]: List of tuples containing (filename, status).
            
        Raises:
            ValueError: If input_dir doesn't exist or isn't a directory.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.is_dir():
            raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")
        
        # Get list of valid image files
        image_files = [
            f for f in input_dir.glob('*') 
            if f.suffix.lower() in self.supported_formats
        ]
        
        if not image_files:
            logger.warning(f"No supported image files found in {input_dir}")
            return []
        
        results: List[Tuple[str, str]] = []
        
        # Process images in parallel with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_image, input_path, output_dir): input_path
                for input_path in image_files
            }
            
            # Create progress bar if requested
            if show_progress:
                pbar = tqdm(
                    total=len(future_to_file),
                    desc="Processing images",
                    unit="image"
                )
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                input_path = future_to_file[future]
                if show_progress:
                    pbar.update(1)
                
                try:
                    status = future.result()
                    results.append((input_path.name, status))
                except Exception as e:
                    results.append((input_path.name, f"Error: {str(e)}"))
                    logger.error(f"Error processing {input_path}: {str(e)}")
            
            if show_progress:
                pbar.close()
        
        # Log summary
        success_count = sum(1 for _, status in results if status == "Success")
        logger.info(f"Processed {len(results)} images: {success_count} successful, "
                   f"{len(results) - success_count} failed")
        
        return results

    # Private methods
    def _is_valid_image_path(self, path: Union[str, Path]) -> bool:
        """
        Check if the given path is a valid image file.
        
        Args:
            path (Union[str, Path]): Path to the image file.
            
        Returns:
            bool: True if path is valid and has supported format, False otherwise.
        """
        path = Path(path)
        return path.exists() and path.suffix.lower() in self.supported_formats

    def _process_single_image(self, input_path: Path, output_dir: Path) -> str:
        """
        Helper method to process a single image for batch processing.
        
        Args:
            input_path (Path): Path to input image.
            output_dir (Path): Directory for output image.
            
        Returns:
            str: Status message indicating success or failure.
        """
        try:
            enhanced_image = self.enhance_image(input_path)
            if enhanced_image is not None:
                output_path = output_dir / f"enhanced_{input_path.name}"
                if self.save_image(enhanced_image, output_path):
                    return "Success"
                return "Failed to save"
            return "Enhancement failed"
        except Exception as e:
            return f"Error: {str(e)}"

    def _optimize_parameters(self, img: np.ndarray) -> Tuple[float, int]:
        """
        Automatically optimize CLAHE parameters based on image characteristics.
        
        Args:
            img (np.ndarray): Input image in BGR format.
            
        Returns:
            Tuple[float, int]: Optimized (clip_limit, tile_size)
        """
        # Convert to grayscale for analysis
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Calculate image statistics
        contrast = self._calculate_contrast(gray)
        entropy = self._calculate_entropy(gray)
        brightness = self._calculate_brightness(gray)
        
        # Optimize clip limit based on image characteristics
        if contrast < 0.3:  # Low contrast image
            clip_limit = 3.0
        elif contrast > 0.7:  # High contrast image
            clip_limit = 1.5
        else:  # Medium contrast
            clip_limit = 2.0
            
        # Optimize tile size based on image characteristics
        if entropy < 6.0:  # Low detail
            tile_size = 16
        elif entropy > 7.5:  # High detail
            tile_size = 6
        else:  # Medium detail
            tile_size = 8
            
        # Adjust for brightness
        if brightness < 0.3:  # Dark image
            clip_limit *= 1.2
        elif brightness > 0.7:  # Bright image
            clip_limit *= 0.8
            
        logger.debug(f"Optimized parameters - Clip limit: {clip_limit:.1f}, "
                    f"Tile size: {tile_size}")
        
        return clip_limit, tile_size

    def _calculate_contrast(self, img: np.ndarray) -> float:
        """Calculate image contrast using standard deviation."""
        return float(np.std(img) / 255.0)

    def _calculate_entropy(self, img: np.ndarray) -> float:
        """Calculate image entropy."""
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zero probabilities
        return float(-np.sum(hist * np.log2(hist)))

    def _calculate_brightness(self, img: np.ndarray) -> float:
        """Calculate relative image brightness."""
        return float(np.mean(img) / 255.0)


def main():
    """Example usage of the ImageProcessor class with different configurations."""
    
    # Example 1: Basic usage with default parameters
    processor = ImageProcessor()
    try:
        # Process single image with default parameters
        image = processor.enhance_image("example.jpg")
        if image is not None:
            processor.save_image(image, "enhanced_default.jpg")
    except ValueError as e:
        logger.error(f"Error processing single image: {e}")

    # Example 2: Enhanced settings for high dynamic range images
    hdr_processor = ImageProcessor(
        clip_limit=4.0,      # Higher clip limit for HDR
        tile_size=16,        # Larger tile size
        color_space='HSV',   # Using HSV color space
        normalize_output=True # Ensure proper value range
    )
    try:
        # Process an HDR image
        image = hdr_processor.enhance_image("example_hdr.exr")
        if image is not None:
            hdr_processor.save_image(image, "enhanced_hdr.jpg")
    except ValueError as e:
        logger.error(f"Error processing HDR image: {e}")

    # Example 3: Batch processing with different color spaces
    for color_space in ['LAB', 'HSV', 'YUV']:
        processor = ImageProcessor(
            clip_limit=2.0,
            tile_size=8,
            color_space=color_space
        )
        try:
            output_dir = f"output_folder_{color_space.lower()}"
            results = processor.batch_process("input_folder", output_dir)
            logger.info(f"\nResults for {color_space} color space:")
            for filename, status in results:
                logger.info(f"{filename}: {status}")
        except ValueError as e:
            logger.error(f"Error during batch processing with {color_space}: {e}")

    # Example 4: Processing with different clip limits for comparison
    clip_limits = [1.5, 2.0, 3.0, 4.0]
    for clip_limit in clip_limits:
        processor = ImageProcessor(
            clip_limit=clip_limit,
            normalize_output=True
        )
        try:
            image = processor.enhance_image("example.jpg")
            if image is not None:
                processor.save_image(
                    image, 
                    f"enhanced_clip_{clip_limit:.1f}.jpg"
                )
        except ValueError as e:
            logger.error(f"Error processing with clip limit {clip_limit}: {e}")


if __name__ == "__main__":
    main()