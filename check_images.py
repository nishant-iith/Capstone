import PIL.Image as Image
import os

stained_dir = "1000-20260412T223922Z-3-001/1000/stained"
unstained_dir = "1000-20260412T223922Z-3-001/1000/unstained"

# Use one of the matching patches found
patch_id = "AS-5198-23-Z36_patch_27648_41984"
stained_path = os.path.join(stained_dir, f"{patch_id}_stained.tif")
unstained_path = os.path.join(unstained_dir, f"{patch_id}_unstained.tif")

def get_info(path):
    with Image.open(path) as img:
        return img.size, img.mode

s_size, s_mode = get_info(stained_path)
u_size, u_mode = get_info(unstained_path)

print(f"Stained: Size={s_size}, Mode={s_mode}")
print(f"Unstained: Size={u_size}, Mode={u_mode}")
