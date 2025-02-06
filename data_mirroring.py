from PIL import Image, ImageOps
import os

def mirror(input_path, output_path):
    try:
        with Image.open(input_path) as img:
            img = img.resize((512, 512))#resize the original image from 572x572 to 512x512

            base_width, base_height = img.size #512x512
            border_size = 30

            # Create mirrored sections
            left = ImageOps.mirror(img.crop((0, 0, border_size, base_height)))# mirror() flip image horizontally
            right = ImageOps.mirror(img.crop((base_width - border_size, 0, base_width, base_height)))
            top = ImageOps.flip(img.crop((0, 0, base_width, border_size))) #flip() flips the image vertically
            bottom = ImageOps.flip(img.crop((0, base_height - border_size, base_width, base_height)))

            # Create corner mirrors
            top_left = ImageOps.mirror(ImageOps.flip(img.crop((0, 0, border_size, border_size)))) # crop() to remove pixels
            top_right = ImageOps.mirror(ImageOps.flip(img.crop((base_width - border_size, 0, base_width, border_size)))) #flip() flips the image vertically
            bottom_left = ImageOps.mirror(ImageOps.flip(img.crop((0, base_height - border_size, border_size, base_height)))) # mirror() flip image horizontally
            bottom_right = ImageOps.mirror(ImageOps.flip(img.crop((base_width - border_size, base_height - border_size, base_width, base_height))))

            size = 572
            mirror = Image.new('RGB', (size, size))# new blank imae

            mirror.paste(img, (border_size, border_size))#place the original picture resize in the center

            # Place the side mirrors
            mirror.paste(left, (0, border_size))  # paste() paste an image form another image
            mirror.paste(right, (base_width + border_size, border_size)) 
            mirror.paste(top, (border_size, 0))  
            mirror.paste(bottom, (border_size, base_height + border_size)) 

            # Place the corner mirrors
            mirror.paste(top_left, (0, 0))  # Top-left
            mirror.paste(top_right, (base_width + border_size, 0))  # Top-right
            mirror.paste(bottom_left, (0, base_height + border_size))  # Bottom-left
            mirror.paste(bottom_right, (base_width + border_size, base_height + border_size))  # Bottom-right

            # Save the augmented image
            mirror.save(output_path, format='JPEG')
        print(f"Augmented image saved at {output_path}")
    except Exception as e:
        print(f"Error augmenting {input_path}: {e}")

def images(input_dir, output_dir):
    if not os.path.exists(output_dir):  #if the directory doesn't exits it create it 
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith('.jpg'): #just jpg pivtures
            input_path = os.path.join(input_dir, file_name)
            output_file_name = os.path.splitext(file_name)[0] + '_mirror.jpg'#changes the filename for the created picture
            output_path = os.path.join(output_dir, output_file_name)

            mirror(input_path, output_path)

def merge_tiff(input_dir, output_file):
 
    try:
        jpg_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.jpg')] #gets all jpg files from thedirectory

        jpg_files.sort()

        images = [Image.open(jpg_file).convert("RGB") for jpg_file in jpg_files]

        images[0].save(output_file, save_all=True, append_images=images[1:], format="TIFF") #saves pictur with the first one been the cover for tif file
    except Exception as e:
        print(f"Error merging JPG files: {e}")

if __name__ == "__main__":
    input_dir =  "/mnt/c/Users/rebec/Desktop/dataset/train-volume"
    mirrored = "/mnt/c/Users/rebec/Desktop/dataset/train-volumeisb"    
    output_file = "/mnt/c/Users/rebec/Desktop/dataset/train.tif"  

    images(input_dir, mirrored)

    merge_tiff(mirrored, output_file)
