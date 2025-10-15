import cv2
import os


def normalize_image(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    img = cv2.resize(img, (80, 80))
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(output_path, img)
    return True

def normalize_dataset(input_dir, output_dir):
    """
    Normalize images preserving the directory structure.
    
    Args:
        input_dir: Path to input directory (e.g., 'data/raw')
        output_dir: Path to output directory (e.g., 'data/normalized')
    """
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    count = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding output directory structure
        relative_path = os.path.relpath(root, input_dir)
        if relative_path == '.':
            output_subdir = output_dir
        else:
            output_subdir = os.path.join(output_dir, relative_path)
        
        os.makedirs(output_subdir, exist_ok=True)
        
        # Process each image file
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                input_path = os.path.join(root, filename)
                output_path = os.path.join(output_subdir, filename)
                if normalize_image(input_path, output_path):
                    count += 1
                    print(f"Processed: {input_path} -> {output_path}")
    
    return count