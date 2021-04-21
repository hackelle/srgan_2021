import torch
from utils import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
srgan_checkpoint = "./checkpoint_srgan.pth.tar"
srgan_checkpoint_base = "./checkpoint_srgan_base.pth.tar"
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

# Load models
if torch.cuda.is_available():
    srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
    srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
    srgan_generator_base = torch.load(srgan_checkpoint_base)['generator'].to(device)
else:
    srresnet = torch.load(srresnet_checkpoint, map_location=torch.device('cpu'))['model'].to(device)
    srgan_generator = torch.load(srgan_checkpoint, map_location=torch.device('cpu'))['generator'].to(device)
    srgan_generator_base = torch.load(srgan_checkpoint_base, map_location=torch.device('cpu'))['generator'].to(device)
srresnet.eval()
srgan_generator.eval()
srgan_generator_base.eval()



def visualize_sr(img, halve=False):
    """
    Visualizes the super-resolved images from the SRResNet and SRGAN for comparison with the bicubic-upsampled image
    and the original high-resolution (HR) image, as done in the paper.

    :param img: filepath of the HR iamge
    :param halve: halve each dimension of the HR image to make sure it's not greater than the dimensions of your screen?
                  For instance, for a 2160p HR image, the LR image will be of 540p (1080p/4) resolution. On a 1080p screen,
                  you will therefore be looking at a comparison between a 540p LR image and a 1080p SR/HR image because
                  your 1080p screen can only display the 2160p SR/HR image at a downsampled 1080p. This is only an
                  APPARENT rescaling of 2x.
                  If you want to reduce HR resolution by a different extent, modify accordingly.
    """
    # Load image, downsample to obtain low-res version
    hr_img = Image.open(img, mode="r")
    hr_img = hr_img.convert('RGB')
    if halve:
        hr_img = hr_img.resize((int(hr_img.width / 2), int(hr_img.height / 2)),
                               Image.LANCZOS)
    lr_img = hr_img.resize((int(hr_img.width / 2), int(hr_img.height / 2)),
                           Image.BILINEAR)

    # Bicubic Upsampling
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    # Super-resolution (SR) with SRResNet
    sr_img_srresnet = srresnet(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')

    # Super-resolution (SR) with SRGAN
    sr_img_srgan = srgan_generator(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')

    # Super-resolution (SR) with SRGAN Base
    sr_img_srgan_base = srgan_generator_base(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan_base = sr_img_srgan_base.squeeze(0).cpu().detach()
    sr_img_srgan_base = convert_image(sr_img_srgan_base, source='[-1, 1]', target='pil')

    # Create grid
    margin = 40
    grid_img = Image.new('RGB', (2 * hr_img.width + 3 * margin, 3 * hr_img.height + 4 * margin), (255, 255, 255))

    # Font
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("calibril.ttf", size=23)
        # It will also look for this file in your OS's default fonts directory, where you may have the Calibri Light font installed if you have MS Office
        # Otherwise, use any TTF font of your choice
    except OSError:
        print(
            "Defaulting to a terrible font. To use a font of your choice, include the link to its TTF file in the function.")
        font = ImageFont.load_default()

    # Place bicubic-upsampled image
    grid_img.paste(bicubic_img, (margin, margin))
    text_size = font.getsize("Bicubic")
    draw.text(xy=[margin + bicubic_img.width / 2 - text_size[0] / 2, margin - text_size[1] - 5], text="Bicubic",
              font=font,
              fill='black')

    # Place SRResNet image
    grid_img.paste(sr_img_srresnet, (2 * margin + bicubic_img.width, margin))
    text_size = font.getsize("SRResNet")
    draw.text(
        xy=[2 * margin + bicubic_img.width + sr_img_srresnet.width / 2 - text_size[0] / 2, margin - text_size[1] - 5],
        text="SRResNet", font=font, fill='black')

    # Place SRGAN image
    grid_img.paste(sr_img_srgan, (margin, 2 * margin + sr_img_srresnet.height))
    text_size = font.getsize("SRGAN")
    draw.text(
        xy=[margin + bicubic_img.width / 2 - text_size[0] / 2, 2 * margin + sr_img_srresnet.height - text_size[1] - 5],
        text="SRGAN", font=font, fill='black')

    # Place SRGAN-Base image
    grid_img.paste(sr_img_srgan, (margin, 3 * margin + sr_img_srresnet.height + sr_img_srgan.height))
    text_size = font.getsize("SRGAN-BASE")
    draw.text(
        xy=[margin + bicubic_img.width / 2 - text_size[0] / 2, 3 * margin + sr_img_srresnet.height + sr_img_srgan.height - text_size[1] - 5],
        text="SRGAN-BASE", font=font, fill='black')

    # Place diff of SRGANs images
    np_srgan = np.asarray(sr_img_srgan)
    np_srgan_base = np.asarray(sr_img_srgan_base)
    srgan_diff = np_srgan_base - np_srgan
    grid_img.paste(Image.fromarray(srgan_diff), (2 * margin + bicubic_img.width, 3 * margin + sr_img_srresnet.height + sr_img_srgan.height))
    text_size = font.getsize("diff(SRGAN-BASE, SRGAN)")
    draw.text(
        xy=[2 * margin + bicubic_img.width + sr_img_srresnet.width / 2 - text_size[0] / 2,
            3 * margin + sr_img_srresnet.height + sr_img_srgan.height - text_size[1] - 5],
        text="diff(SRGAN-BASE, SRGAN)", font=font, fill='black')

    # Place original HR image
    grid_img.paste(hr_img, (2 * margin + bicubic_img.width, 2 * margin + sr_img_srresnet.height))
    text_size = font.getsize("Original HR")
    draw.text(xy=[2 * margin + bicubic_img.width + sr_img_srresnet.width / 2 - text_size[0] / 2,
                  2 * margin + sr_img_srresnet.height - text_size[1] - 1], text="Original HR", font=font, fill='black')

    # Display grid
    grid_img.show()

    return grid_img


if __name__ == '__main__':
    grid_img = visualize_sr("./Images/test/airplane97.tif")
