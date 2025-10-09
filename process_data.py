import os
from PIL import Image 
import shutil

def split_composite_image(image_path, save_path_a, save_path_b):
    """
    Opens a composite image, splits it into left (A) and right (B) halves,
    and saves them to the specified paths.

    Args:
        image_path (str): Path to the source composite image.
        save_path_a (str): Path to save the left half (A).
        save_path_b (str): Path to save the right half (B).
    """
    try:
        # Open the composite image
        with Image.open(image_path) as img:
            width, height = img.size

            # Calculate the middle point to split the image
            mid_x = width // 2
            # Define the coordinates for the left and right halves
            left_box = (0, 0, mid_x, height)
            right_box = (mid_x, 0, width, height)

            # Crop the image
            img_a = img.crop(left_box)
            img_b = img.crop(right_box)

            # Save the cropped images
            img_a.save(save_path_a)
            img_b.save(save_path_b)
            # print(f"Successfully split and saved: {os.path.basename(image_path)}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
def process_dataset(composite_folder, output_folder_a, output_folder_b):
    """
    Processes all images in a given folder (e.g., 'train'), splitting each one
    and saving the halves to output folders A and B.

    Args:
        composite_folder (str): Path to the folder containing composite images.
        output_folder_a (str): Path to the folder for saving A parts.
        output_folder_b (str): Path to the folder for saving B parts.
    """
    # Create output directories if they don't exist
    os.makedirs(output_folder_a, exist_ok=True)
    os.makedirs(output_folder_b, exist_ok=True)

    # Get list of all image files in the composite folder
    # Supports common image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(composite_folder)
                   if f.lower().endswith(valid_extensions)]

    print(f"Found {len(image_files)} images in '{composite_folder}'. Starting split...")

    for img_file in image_files:
        composite_path = os.path.join(composite_folder, img_file)
        
        # Create the output filenames (using the same original name)
        output_filename = img_file
        save_path_a = os.path.join(output_folder_a, output_filename)
        save_path_b = os.path.join(output_folder_b, output_filename)

        # Split and save the image
        split_composite_image(composite_path, save_path_a, save_path_b)

    print(f"Finished processing '{composite_folder}'! Files saved to:\n{output_folder_a}\n{output_folder_b}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    base_path = r"C:\Users\hp\Saclay-ai\cs230\image-to-image\datasets" 
    
    # Paths to your original composite folders
    train_composite_dir = os.path.join(base_path, "test")
    val_composite_dir = os.path.join(base_path, "val")
    
    # Paths for the new split folders we will create
    output_base = os.path.join(base_path, "split_datasets")
    train_a_dir = os.path.join(output_base, "testA")
    train_b_dir = os.path.join(output_base, "testB")
    val_a_dir = os.path.join(output_base, "valA")
    val_b_dir = os.path.join(output_base, "valB")

    # Process the training set
    process_dataset(train_composite_dir, train_a_dir, train_b_dir)
    
    # Process the validation set
    process_dataset(val_composite_dir, val_a_dir, val_b_dir)
    
    print("\nDataset splitting complete!")
    print(f"Your finalized directory structure under '{output_base}' is:")
    print("testA/ - Contains the input images (left halves)")
    print("testB/ - Contains the target images (right halves)")
    print("valA/   - Contains the input images for validation")
    print("valB/   - Contains the target images for validation")
