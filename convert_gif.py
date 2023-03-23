from PIL import Image
import glob
import natsort

png_files = natsort.natsorted(glob.glob('./results/*.png'))
print(png_files)
# png_files = ['image1.png', 'image2.png', 'image3.png']

gif_file = 'animated.gif'

with Image.open(png_files[0]) as im:
    im.save(gif_file, save_all=True, append_images=[Image.open(f) for f in png_files[1:]], duration=500, loop=0)
