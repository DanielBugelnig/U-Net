from PIL import Image, ImageOps
import os

def mirror(input_path, output_path):
    try:
        with Image.open(input_path) as img:
            # Resize the input image to 512x512 to fit within the final 572x572 dimensions with borders
            img = img.resize((512, 512))

            # Dimensions
            base_width, base_height = img.size
            border_size = 30

            # Create mirrored sections
            mirror_left = ImageOps.mirror(img.crop((0, 0, border_size, base_height)))
            mirror_right = ImageOps.mirror(img.crop((base_width - border_size, 0, base_width, base_height)))
            mirror_top = ImageOps.flip(img.crop((0, 0, base_width, border_size)))
            mirror_bottom = ImageOps.flip(img.crop((0, base_height - border_size, base_width, base_height)))

            # Create corner mirrors
            corner_top_left = ImageOps.mirror(ImageOps.flip(img.crop((0, 0, border_size, border_size))))
            corner_top_right = ImageOps.mirror(ImageOps.flip(img.crop((base_width - border_size, 0, base_width, border_size))))
            corner_bottom_left = ImageOps.mirror(ImageOps.flip(img.crop((0, base_height - border_size, border_size, base_height))))
            corner_bottom_right = ImageOps.mirror(ImageOps.flip(img.crop((base_width - border_size, base_height - border_size, base_width, base_height))))

            # Create a new blank image with final size 572x572
            final_size = 572
            new_image = Image.new('RGB', (final_size, final_size))

            # Place the original image in the center
            new_image.paste(img, (border_size, border_size))

            # Place the side mirrors
            new_image.paste(mirror_left, (0, border_size))  # Left
            new_image.paste(mirror_right, (base_width + border_size, border_size))  # Right
            new_image.paste(mirror_top, (border_size, 0))  # Top
            new_image.paste(mirror_bottom, (border_size, base_height + border_size))  # Bottom

            # Place the corner mirrors
            new_image.paste(corner_top_left, (0, 0))  # Top-left
            new_image.paste(corner_top_right, (base_width + border_size, 0))  # Top-right
            new_image.paste(corner_bottom_left, (0, base_height + border_size))  # Bottom-left
            new_image.paste(corner_bottom_right, (base_width + border_size, base_height + border_size))  # Bottom-right

            # Save the augmented image
            new_image.save(output_path, format='JPEG')
        print(f"Augmented image saved at {output_path}")
    except Exception as e:
        print(f"Error augmenting {input_path}: {e}")

def images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith('.jpg'):
            input_path = os.path.join(input_dir, file_name)
            output_file_name = os.path.splitext(file_name)[0] + '_augmented.jpg'
            output_path = os.path.join(output_dir, output_file_name)

            mirror(input_path, output_path)

def merge_tiff(input_dir, output_tiff_path):
 
    try:
        # Get all JPG files in the directory
        jpg_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]

        # Sort files 
        jpg_files.sort()

        # Open all images
        images = [Image.open(jpg_file).convert("RGB") for jpg_file in jpg_files]

        # Save as multi-page TIFF
        images[0].save(output_tiff_path, save_all=True, append_images=images[1:], format="TIFF")
        print(f"Merged {len(images)} JPG files into {output_tiff_path}")
    except Exception as e:
        print(f"Error merging JPG files: {e}")

if __name__ == "__main__":
    # Directory paths
    input_directory =  "/mnt/c/Users/rebec/Desktop/dataset/DeepGlobe_Land_572/train" 
    augmented_directory = "/mnt/c/Users/rebec/Desktop/dataset/trainmirror"    
    output_tiff_file = "/mnt/c/Users/rebec/Desktop/dataset/trainmirror.tif"  


    images(input_directory, augmented_directory)

    #Merge augmented JPG images into a single TIFF file
    merge_tiff(augmented_directory, output_tiff_file)
