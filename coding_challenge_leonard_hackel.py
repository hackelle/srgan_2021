import os
import shutil
from tqdm import tqdm
from utils import *
from train_srgan import *
import torch
from PIL import Image
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 250
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

# this toggles if we are training the network or only testing an existing one
training_active = False
# this toggles if we are testing the network or only testing a single image
eval_network = True

# base directory of the image files
image_dir = './Images/'

# get the number of classes
num_classes = len([name for name in os.listdir(image_dir)
                   if os.path.isdir(os.path.join(image_dir, name))
                   and name not in ["train", "val", "test", "data_lists"]])

# define percentages here
partition = {"train": 70, "val": 10, "test": 20}
assert (sum(partition.values()) == 100)

# calculate which images to copy into which split
ranges = {"train": [0, partition["train"]], "val": [partition["train"], partition["train"] + partition["val"]],
          "test": [partition["train"] + partition["val"], 100]}

# create a folder where to save the re-arranged files.
if not os.path.isdir("training_files/"):
    os.mkdir("training_files")

# copy files
for phase in partition:
    phase_dir = f"training_files/{phase}"
    # if folders for train/val/test don't exist, create and fill
    if not os.path.isdir(phase_dir):
        # create folder for phase
        os.mkdir(phase_dir)
        for folder in tqdm([name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))],
                           desc=f"Copying for {phase}",
                           total=len([name for name in os.listdir(image_dir) if os.path.isdir(name)])):
            # don't copy files with name "train", "val" or "test"
            if folder in partition.keys():
                continue
            for idx in tqdm(range(ranges[phase][0], ranges[phase][1]), desc=f"Copying for {folder}", leave=False,
                            total=len(range(ranges[phase][0], ranges[phase][1]))):
                # copy images 0..x-1 in train folder, x..y in val and y..99 in test for each class
                filepath = os.path.join(image_dir, folder, f'{folder}{idx:02d}.tif')
                shutil.copyfile(filepath, f'training_files/{phase}/{folder}{idx:02d}.tif')
    else:
        # assert that the correct number of images is in the folder
        num_images_is = len([name for name in os.listdir(phase_dir) if os.path.isfile(f'{phase_dir}/{name}')])
        num_images_should = num_classes * (ranges[phase][1] - ranges[phase][0])
        if num_images_is != num_images_should:
            print(
                f"Folder 'training_files/{phase}' has only {num_images_is} but should be {num_images_should}. This may indicate a configuration problem.")
        else:
            print(f"Number of images in folder training_files/{phase} as expected.")

create_data_lists(train_folders=['./training_files/train'],
                  valid_folders=['./training_files/val'],
                  test_folders=['./training_files/test'],
                  min_size=100,
                  output_folder='./')

if training_active:
    main_srgan()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
srgan_checkpoint = "./checkpoint_srgan.pth (version2 29).tar"

# Load models
if torch.cuda.is_available():
    srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
else:
    srgan_generator = torch.load(srgan_checkpoint, map_location=torch.device('cpu'))['generator'].to(device)

# set Network to correct state
srgan_generator.eval()


def evaluation_test(test_data_name='test'):
    print(f"\n    Running Tests on model {srgan_checkpoint}\n")
    test_dataset = SRDataset(data_folder,
                             split='test',
                             crop_size=0,
                             scaling_factor=2,
                             lr_img_type='imagenet-norm',
                             hr_img_type='[-1, 1]',
                             test_data_name=test_data_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=6,
                                              pin_memory=True)

    with torch.no_grad():
        # Batches
        metrics = {'rmse': 0, 'sre': 0, 'uqi': 0}
        for i, (lr_img, hr_img) in tqdm(enumerate(test_loader), desc="Testing", unit=' batch',
                                        position=1, leave=False, total=len(test_loader)):
            # Move to default device
            lr_img = lr_img.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
            hr_img = hr_img.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

            # Forward prop.
            sr_img_srgan = srgan_generator(lr_img)  # (1, 3, w, h), in [-1, 1]

            sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='y-channel').squeeze(0)
            hr_img = convert_image(hr_img, source='[-1, 1]', target='y-channel').squeeze(0)

            sr_img_srgan = sr_img_srgan.cpu().numpy()
            hr_img = hr_img.cpu().numpy()

            # add up all metrices for this image
            metric = get_metrics(sr_img_srgan, hr_img)
            for m in metric.keys():
                metrics[m] += metric[m]

        # calculate average metric
        print("")
        for m in metric.keys():
            metrics[m] /= len(test_loader)
            print(f"Average metric in test: {m} -> {metrics[m]}")


# run test
if eval_network:
    evaluation_test()

def prepare_images(img_path):
    # helper function for visualization
    # Load image, downsample to obtain low-res version
    hr_img = Image.open(img_path, mode="r")
    hr_img = hr_img.convert('RGB')
    hr_img = hr_img.resize((256, 256), Image.BILINEAR)
    lr_img = hr_img.resize((int(hr_img.width / 2), int(hr_img.height / 2)),
                           Image.BILINEAR)

    # Bicubic Upsampling
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)
    # Bilinear Upsampling
    bilinear_img = lr_img.resize((hr_img.width, hr_img.height), Image.BILINEAR)

    with torch.no_grad():
        # Super-resolution (SR) with SRGAN
        sr_img_srgan = srgan_generator(
            convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
        sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
        sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')

    return {'gt': hr_img,
            'lr': lr_img,
            'bicubic': bicubic_img,
            'bilinear': bilinear_img,
            'srgan': sr_img_srgan}


def plot_image(image, title=""):
    """
    helper function for visualization
    Plots images from image tensors.
    Args:
        image: 3D image tensor. [height, width, channels].
        title: Title to display in the plot.
    """
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)

# test all images in this list
img_list = ["./training_files/test/baseballdiamond93.tif", "./training_files/test/overpass80.tif", "./training_files/test/denseresidential99.tif", "./training_files/test/chaparral97.tif", "./training_files/test/buildings85.tif", "./training_files/test/runway84.tif", "./training_files/test/beach85.tif", "./training_files/test/storagetanks93.tif", "./training_files/test/freeway87.tif", "./training_files/test/mediumresidential86.tif", "./training_files/test/sparseresidential86.tif", "./training_files/test/intersection92.tif", "./training_files/test/buildings88.tif", "./training_files/test/baseballdiamond89.tif", "./training_files/test/mobilehomepark84.tif", "./training_files/test/freeway86.tif", "./training_files/test/harbor88.tif", "./training_files/test/golfcourse91.tif", "./training_files/test/sparseresidential97.tif", "./training_files/test/storagetanks96.tif", "./training_files/test/sparseresidential87.tif", "./training_files/test/sparseresidential88.tif", "./training_files/test/denseresidential86.tif", "./training_files/test/agricultural93.tif", "./training_files/test/mobilehomepark90.tif", "./training_files/test/mobilehomepark89.tif", "./training_files/test/forest99.tif", "./training_files/test/beach96.tif", "./training_files/test/overpass83.tif", "./training_files/test/baseballdiamond92.tif", "./training_files/test/baseballdiamond99.tif", "./training_files/test/overpass84.tif", "./training_files/test/mobilehomepark94.tif", "./training_files/test/river95.tif", "./training_files/test/denseresidential87.tif", "./training_files/test/buildings87.tif", "./training_files/test/forest86.tif", "./training_files/test/agricultural94.tif", "./training_files/test/baseballdiamond94.tif", "./training_files/test/forest88.tif", "./training_files/test/sparseresidential93.tif", "./training_files/test/mobilehomepark98.tif", "./training_files/test/tenniscourt97.tif", "./training_files/test/harbor85.tif", "./training_files/test/sparseresidential99.tif", "./training_files/test/runway89.tif", "./training_files/test/forest83.tif", "./training_files/test/buildings80.tif", "./training_files/test/baseballdiamond98.tif", "./training_files/test/river88.tif", "./training_files/test/airplane95.tif", "./training_files/test/buildings91.tif", "./training_files/test/sparseresidential82.tif", "./training_files/test/harbor92.tif", "./training_files/test/forest92.tif", "./training_files/test/river83.tif", "./training_files/test/harbor93.tif", "./training_files/test/mobilehomepark92.tif", "./training_files/test/parkinglot99.tif", "./training_files/test/intersection97.tif", "./training_files/test/freeway84.tif", "./training_files/test/runway95.tif", "./training_files/test/sparseresidential91.tif", "./training_files/test/chaparral87.tif", "./training_files/test/forest82.tif", "./training_files/test/forest94.tif", "./training_files/test/golfcourse87.tif", "./training_files/test/tenniscourt87.tif", "./training_files/test/chaparral98.tif", "./training_files/test/airplane82.tif", "./training_files/test/parkinglot95.tif", "./training_files/test/agricultural86.tif", "./training_files/test/overpass94.tif", "./training_files/test/buildings89.tif", "./training_files/test/denseresidential90.tif", "./training_files/test/parkinglot80.tif", "./training_files/test/forest95.tif", "./training_files/test/mobilehomepark97.tif", "./training_files/test/parkinglot90.tif", "./training_files/test/airplane87.tif", "./training_files/test/parkinglot94.tif", "./training_files/test/freeway99.tif", "./training_files/test/agricultural99.tif", "./training_files/test/harbor97.tif", "./training_files/test/overpass82.tif", "./training_files/test/golfcourse99.tif", "./training_files/test/overpass81.tif", "./training_files/test/mediumresidential83.tif", "./training_files/test/golfcourse85.tif", "./training_files/test/forest90.tif", "./training_files/test/overpass86.tif", "./training_files/test/golfcourse84.tif", "./training_files/test/baseballdiamond86.tif", "./training_files/test/denseresidential89.tif", "./training_files/test/freeway85.tif", "./training_files/test/denseresidential88.tif", "./training_files/test/harbor80.tif", "./training_files/test/baseballdiamond97.tif", "./training_files/test/tenniscourt90.tif", "./training_files/test/storagetanks83.tif", "./training_files/test/storagetanks80.tif", "./training_files/test/intersection84.tif", "./training_files/test/runway92.tif", "./training_files/test/tenniscourt85.tif", "./training_files/test/chaparral94.tif", "./training_files/test/tenniscourt92.tif", "./training_files/test/sparseresidential92.tif", "./training_files/test/tenniscourt93.tif", "./training_files/test/runway91.tif", "./training_files/test/freeway88.tif", "./training_files/test/parkinglot84.tif", "./training_files/test/buildings98.tif", "./training_files/test/buildings97.tif", "./training_files/test/beach90.tif", "./training_files/test/river99.tif", "./training_files/test/parkinglot87.tif", "./training_files/test/sparseresidential84.tif", "./training_files/test/golfcourse98.tif", "./training_files/test/river98.tif", "./training_files/test/buildings84.tif", "./training_files/test/agricultural90.tif", "./training_files/test/beach84.tif", "./training_files/test/agricultural81.tif", "./training_files/test/agricultural85.tif", "./training_files/test/river89.tif", "./training_files/test/denseresidential97.tif", "./training_files/test/parkinglot85.tif", "./training_files/test/golfcourse90.tif", "./training_files/test/mediumresidential97.tif", "./training_files/test/overpass85.tif", "./training_files/test/beach82.tif", "./training_files/test/parkinglot82.tif", "./training_files/test/harbor99.tif", "./training_files/test/overpass99.tif", "./training_files/test/beach81.tif", "./training_files/test/mobilehomepark93.tif", "./training_files/test/mediumresidential82.tif", "./training_files/test/river97.tif", "./training_files/test/airplane80.tif", "./training_files/test/denseresidential94.tif", "./training_files/test/agricultural83.tif", "./training_files/test/tenniscourt83.tif", "./training_files/test/runway98.tif", "./training_files/test/beach92.tif", "./training_files/test/parkinglot97.tif", "./training_files/test/sparseresidential89.tif", "./training_files/test/mobilehomepark99.tif", "./training_files/test/denseresidential82.tif", "./training_files/test/forest93.tif", "./training_files/test/sparseresidential94.tif", "./training_files/test/storagetanks92.tif", "./training_files/test/forest89.tif", "./training_files/test/mediumresidential85.tif", "./training_files/test/airplane81.tif", "./training_files/test/chaparral91.tif", "./training_files/test/mediumresidential93.tif", "./training_files/test/sparseresidential85.tif", "./training_files/test/mediumresidential95.tif", "./training_files/test/sparseresidential81.tif", "./training_files/test/intersection89.tif", "./training_files/test/denseresidential93.tif", "./training_files/test/intersection87.tif", "./training_files/test/golfcourse88.tif", "./training_files/test/tenniscourt94.tif", "./training_files/test/mediumresidential94.tif", "./training_files/test/baseballdiamond84.tif", "./training_files/test/chaparral82.tif", "./training_files/test/baseballdiamond82.tif", "./training_files/test/runway96.tif", "./training_files/test/overpass90.tif", "./training_files/test/agricultural88.tif", "./training_files/test/beach98.tif", "./training_files/test/tenniscourt95.tif", "./training_files/test/intersection83.tif", "./training_files/test/sparseresidential80.tif", "./training_files/test/river93.tif", "./training_files/test/storagetanks99.tif", "./training_files/test/beach88.tif", "./training_files/test/mobilehomepark91.tif", "./training_files/test/mobilehomepark83.tif", "./training_files/test/denseresidential92.tif", "./training_files/test/intersection90.tif", "./training_files/test/beach95.tif", "./training_files/test/mediumresidential80.tif", "./training_files/test/river90.tif", "./training_files/test/freeway80.tif", "./training_files/test/river81.tif", "./training_files/test/storagetanks94.tif", "./training_files/test/tenniscourt86.tif", "./training_files/test/airplane97.tif", "./training_files/test/mediumresidential96.tif", "./training_files/test/chaparral83.tif", "./training_files/test/baseballdiamond91.tif", "./training_files/test/overpass98.tif", "./training_files/test/beach83.tif", "./training_files/test/agricultural84.tif", "./training_files/test/runway86.tif", "./training_files/test/mediumresidential88.tif", "./training_files/test/storagetanks97.tif", "./training_files/test/runway80.tif", "./training_files/test/chaparral81.tif", "./training_files/test/harbor91.tif", "./training_files/test/buildings82.tif", "./training_files/test/parkinglot88.tif", "./training_files/test/denseresidential85.tif", "./training_files/test/mediumresidential92.tif", "./training_files/test/intersection93.tif", "./training_files/test/river86.tif", "./training_files/test/parkinglot98.tif", "./training_files/test/buildings94.tif", "./training_files/test/mobilehomepark88.tif", "./training_files/test/buildings86.tif", "./training_files/test/mobilehomepark85.tif", "./training_files/test/denseresidential81.tif", "./training_files/test/mediumresidential84.tif", "./training_files/test/golfcourse96.tif", "./training_files/test/baseballdiamond80.tif", "./training_files/test/buildings96.tif", "./training_files/test/mobilehomepark86.tif", "./training_files/test/storagetanks85.tif", "./training_files/test/intersection96.tif", "./training_files/test/storagetanks98.tif", "./training_files/test/parkinglot81.tif", "./training_files/test/intersection86.tif", "./training_files/test/mediumresidential98.tif", "./training_files/test/harbor98.tif", "./training_files/test/freeway97.tif", "./training_files/test/denseresidential84.tif", "./training_files/test/buildings83.tif", "./training_files/test/baseballdiamond81.tif", "./training_files/test/golfcourse97.tif", "./training_files/test/intersection94.tif", "./training_files/test/overpass93.tif", "./training_files/test/golfcourse81.tif", "./training_files/test/storagetanks95.tif", "./training_files/test/storagetanks81.tif", "./training_files/test/runway83.tif", "./training_files/test/beach94.tif", "./training_files/test/parkinglot83.tif", "./training_files/test/buildings81.tif", "./training_files/test/golfcourse89.tif", "./training_files/test/harbor94.tif", "./training_files/test/forest91.tif", "./training_files/test/agricultural87.tif", "./training_files/test/forest98.tif", "./training_files/test/harbor86.tif", "./training_files/test/agricultural92.tif", "./training_files/test/forest80.tif", "./training_files/test/runway94.tif", "./training_files/test/sparseresidential90.tif", "./training_files/test/parkinglot89.tif", "./training_files/test/intersection99.tif", "./training_files/test/airplane85.tif", "./training_files/test/river96.tif", "./training_files/test/baseballdiamond83.tif", "./training_files/test/intersection98.tif", "./training_files/test/river87.tif", "./training_files/test/chaparral84.tif", "./training_files/test/mobilehomepark80.tif", "./training_files/test/chaparral89.tif", "./training_files/test/agricultural91.tif", "./training_files/test/river82.tif", "./training_files/test/freeway91.tif", "./training_files/test/forest84.tif", "./training_files/test/freeway92.tif", "./training_files/test/overpass96.tif", "./training_files/test/intersection88.tif", "./training_files/test/harbor89.tif", "./training_files/test/beach80.tif", "./training_files/test/mediumresidential90.tif", "./training_files/test/storagetanks87.tif", "./training_files/test/river84.tif", "./training_files/test/mediumresidential81.tif", "./training_files/test/forest96.tif", "./training_files/test/parkinglot96.tif", "./training_files/test/mediumresidential89.tif", "./training_files/test/agricultural82.tif", "./training_files/test/freeway82.tif", "./training_files/test/chaparral80.tif", "./training_files/test/overpass89.tif", "./training_files/test/agricultural80.tif", "./training_files/test/sparseresidential83.tif", "./training_files/test/baseballdiamond85.tif", "./training_files/test/buildings92.tif", "./training_files/test/parkinglot92.tif", "./training_files/test/beach97.tif", "./training_files/test/denseresidential91.tif", "./training_files/test/chaparral93.tif", "./training_files/test/beach93.tif", "./training_files/test/golfcourse82.tif", "./training_files/test/agricultural95.tif", "./training_files/test/freeway83.tif", "./training_files/test/baseballdiamond87.tif", "./training_files/test/beach91.tif", "./training_files/test/tenniscourt96.tif", "./training_files/test/agricultural96.tif", "./training_files/test/runway90.tif", "./training_files/test/chaparral85.tif", "./training_files/test/baseballdiamond95.tif", "./training_files/test/storagetanks91.tif", "./training_files/test/intersection82.tif", "./training_files/test/overpass91.tif", "./training_files/test/parkinglot86.tif", "./training_files/test/beach89.tif", "./training_files/test/chaparral96.tif", "./training_files/test/chaparral92.tif", "./training_files/test/denseresidential80.tif", "./training_files/test/harbor90.tif", "./training_files/test/agricultural89.tif", "./training_files/test/buildings95.tif", "./training_files/test/freeway94.tif", "./training_files/test/airplane93.tif", "./training_files/test/denseresidential96.tif", "./training_files/test/freeway98.tif", "./training_files/test/overpass97.tif", "./training_files/test/denseresidential95.tif", "./training_files/test/tenniscourt81.tif", "./training_files/test/forest97.tif", "./training_files/test/tenniscourt98.tif", "./training_files/test/denseresidential98.tif", "./training_files/test/baseballdiamond88.tif", "./training_files/test/mobilehomepark81.tif", "./training_files/test/tenniscourt84.tif", "./training_files/test/parkinglot91.tif", "./training_files/test/airplane96.tif", "./training_files/test/airplane94.tif", "./training_files/test/golfcourse80.tif", "./training_files/test/harbor96.tif", "./training_files/test/runway81.tif", "./training_files/test/overpass92.tif", "./training_files/test/river85.tif", "./training_files/test/river92.tif", "./training_files/test/storagetanks89.tif", "./training_files/test/storagetanks90.tif", "./training_files/test/harbor82.tif", "./training_files/test/forest85.tif", "./training_files/test/freeway96.tif", "./training_files/test/buildings93.tif", "./training_files/test/sparseresidential98.tif", "./training_files/test/mobilehomepark95.tif", "./training_files/test/overpass95.tif", "./training_files/test/golfcourse94.tif", "./training_files/test/beach87.tif", "./training_files/test/denseresidential83.tif", "./training_files/test/runway82.tif", "./training_files/test/chaparral95.tif", "./training_files/test/tenniscourt99.tif", "./training_files/test/chaparral88.tif", "./training_files/test/freeway89.tif", "./training_files/test/airplane98.tif", "./training_files/test/runway93.tif", "./training_files/test/mobilehomepark87.tif", "./training_files/test/intersection91.tif", "./training_files/test/storagetanks82.tif", "./training_files/test/parkinglot93.tif", "./training_files/test/tenniscourt89.tif", "./training_files/test/golfcourse93.tif", "./training_files/test/airplane92.tif", "./training_files/test/freeway95.tif", "./training_files/test/airplane89.tif", "./training_files/test/golfcourse95.tif", "./training_files/test/chaparral99.tif", "./training_files/test/freeway93.tif", "./training_files/test/tenniscourt82.tif", "./training_files/test/buildings99.tif", "./training_files/test/river94.tif", "./training_files/test/baseballdiamond90.tif", "./training_files/test/harbor95.tif", "./training_files/test/harbor84.tif", "./training_files/test/runway85.tif", "./training_files/test/overpass88.tif", "./training_files/test/airplane83.tif", "./training_files/test/mobilehomepark82.tif", "./training_files/test/intersection95.tif", "./training_files/test/airplane86.tif", "./training_files/test/mediumresidential87.tif", "./training_files/test/harbor83.tif", "./training_files/test/chaparral86.tif", "./training_files/test/agricultural98.tif", "./training_files/test/golfcourse83.tif", "./training_files/test/runway88.tif", "./training_files/test/agricultural97.tif", "./training_files/test/baseballdiamond96.tif", "./training_files/test/golfcourse86.tif", "./training_files/test/tenniscourt80.tif", "./training_files/test/harbor87.tif", "./training_files/test/harbor81.tif", "./training_files/test/golfcourse92.tif", "./training_files/test/intersection81.tif", "./training_files/test/storagetanks88.tif", "./training_files/test/forest81.tif", "./training_files/test/mediumresidential99.tif", "./training_files/test/overpass87.tif", "./training_files/test/chaparral90.tif", "./training_files/test/tenniscourt88.tif", "./training_files/test/airplane84.tif", "./training_files/test/beach86.tif", "./training_files/test/intersection85.tif", "./training_files/test/intersection80.tif", "./training_files/test/airplane88.tif", "./training_files/test/storagetanks84.tif", "./training_files/test/airplane99.tif", "./training_files/test/airplane91.tif", "./training_files/test/river91.tif", "./training_files/test/river80.tif", "./training_files/test/sparseresidential96.tif", "./training_files/test/storagetanks86.tif", "./training_files/test/tenniscourt91.tif", "./training_files/test/runway87.tif", "./training_files/test/freeway81.tif", "./training_files/test/runway99.tif", "./training_files/test/beach99.tif", "./training_files/test/airplane90.tif", "./training_files/test/runway97.tif", "./training_files/test/freeway90.tif", "./training_files/test/sparseresidential95.tif", "./training_files/test/mediumresidential91.tif", "./training_files/test/mobilehomepark96.tif", "./training_files/test/forest87.tif", "./training_files/test/buildings90.tif"]

metrics_bl = {'rmse': 0, 'sre': 0, 'uqi': 0}
metrics_ml = {'rmse': 0, 'sre': 0, 'uqi': 0}

for img in tqdm(img_list):
    imgs = prepare_images(img)

    for key in imgs.keys():
        imgs[key] = np.asarray(imgs[key])

    metric_baseline = get_metrics(imgs["bilinear"], imgs["gt"])
    metric_ml = get_metrics(imgs["srgan"], imgs["gt"])

    for m in metric_ml.keys():
        metrics_ml[m] += metric_ml[m]
        metrics_bl[m] += metric_baseline[m]

# print metrics
for metric in metrics_ml.keys():
    print(f"ML sum: {metric} -> {metrics_ml[metric]}")
    print(f"ML avg: {metric} -> {metrics_ml[metric]/len(img_list)}")

print("\n" + "-" * 10 + "\n")

for metric in metrics_bl.keys():
    print(f"BL sum: {metric} -> {metrics_bl[metric]}")
    print(f"BL sum: {metric} -> {metrics_bl[metric]/len(img_list)}")

# show an example image
viz_image = 'training_files/test/beach99.tif'
img = prepare_images(viz_image)

fig, axes = plt.subplots(2, 2)
fig.tight_layout()
plt.subplot(221)
plot_image(img["gt"], title=f"Original, {img['gt'].size}")
plt.subplot(222)
fig.tight_layout()
plot_image(img["lr"], title=f"downscaled, {img['lr'].size}")
plt.subplot(223)
fig.tight_layout()
plot_image(img["bilinear"], title=f"bilinear, {img['bilinear'].size}")
plt.subplot(224)
fig.tight_layout()
plot_image(img["srgan"], title=f"SRGAN, {img['bilinear'].size}")
plt.show()

for key in img.keys():
    img[key] = np.asarray(img[key])

metric_baseline = get_metrics(img["bilinear"], img["gt"])
metric_ml = get_metrics(img["srgan"], img["gt"])

print("\n-- Perfect metric would be: rmse=0, sre=inf, uqi=1 --\n\n")

# print metrics
for metric in metric_ml.keys():
    print(f"ML img: {metric} -> {metric_ml[metric]}")

print("\n" + "-" * 10 + "\n")

for metric in metric_baseline.keys():
    print(f"BL img: {metric} -> {metric_baseline[metric]}")
