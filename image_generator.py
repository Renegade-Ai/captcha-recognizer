import os
import random
import string

from captcha.image import ImageCaptcha

import config

NUM_OF_IMAGES = 30000

# create a new directory input, dont do anything if already exists
os.makedirs("input", exist_ok=True)
image = ImageCaptcha(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT)

for i in range(NUM_OF_IMAGES):
    text = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    data = image.generate(text)
    image.write(text, f"input/{text}.png")

print(f"Generated {NUM_OF_IMAGES} images")
