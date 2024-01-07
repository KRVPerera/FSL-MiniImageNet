import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import math
import cv2
import numpy as np
import os
from src.dataloader import EuroSatDownloader, createEuroSatDataLoaders, createMiniImageNetDataLoaders, MiniImageNetDownloader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getmean_Stddev(data_transforms, dataDir, class_name_to_calculate):
    dataloaders_eu, class_names_eu, _ = createEuroSatDataLoaders(data_transforms, dataDir, split=0.25, batch_size = 4, classLimit = 10, image_limit_per_class = 2000)
    class_names_eu_li = list(class_names_eu)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    total_mean = torch.zeros(3)  # Assuming RGB images
    total_std_dev = torch.zeros(3)
    names = list(class_names_eu)
    num_samples = 0

    # Iterate through the DataLoader
    for batch in dataloaders_eu['test']:
        inputs_test, classes_test = batch

        # Iterate through each image in the batch
        for i, image in enumerate(inputs_test):
            # Get the class label for the current image
            class_label = classes_test[i]
            # Check if the class label matches the target class name
            if names[classes_test[i].item()] == class_name_to_calculate:
                # Convert the PyTorch tensor to a NumPy ndarray
                image_np = image.numpy()

                # Apply the transformation to the image
                image_transformed = transform(image_np)

                # Calculate the mean and standard deviation of the transformed image
                image_mean = torch.mean(image_transformed, dim=(0, 2))
                image_std_dev = torch.std(image_transformed, dim=(0, 2))

                # Accumulate values
                total_mean += image_mean
                total_std_dev += image_std_dev
                num_samples += 1

    # Calculate the overall mean and standard deviation for the specified class
    overall_mean = total_mean / num_samples
    overall_std_dev = total_std_dev / num_samples

    print(f"{class_name_to_calculate} : Mean of RGB channels for class {class_name_to_calculate}: {overall_mean}")
    print(f"{class_name_to_calculate} : Standard Deviation of RGB channels for class {class_name_to_calculate}: {overall_std_dev}")
    print(overall_mean)
    print(overall_std_dev)

def calculate_folder_statistics(parent_folder):
    # Initialize variables to accumulate pixel values for all subfolders
    total_mean = np.zeros(3)
    total_std_dev = np.zeros(3)
    total_num_images = 0

    # Iterate through subfolders in the parent folder
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            # Initialize variables to accumulate pixel values for the current subfolder
            folder_mean = np.zeros(3)
            folder_std_dev = np.zeros(3)
            num_images = 0

            # Loop through all JPG files in the subfolder
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg"):
                    # Read the image using OpenCV
                    image_path = os.path.join(folder_path, filename)
                    img = cv2.imread(image_path)

                    # Convert the image to RGB format
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Calculate the mean and standard deviation of each RGB channel
                    mean = np.mean(img_rgb, axis=(0, 1))
                    std_dev = np.std(img_rgb, axis=(0, 1))

                    # Accumulate values for the current subfolder
                    folder_mean += mean
                    folder_std_dev += std_dev
                    num_images += 1

                    # Accumulate values for all subfolders
                    total_mean += mean
                    total_std_dev += std_dev
                    total_num_images += 1

            # Calculate the average statistics for the current subfolder
            avg_folder_mean = folder_mean / num_images
            avg_folder_std_dev = folder_std_dev / num_images

            # Print statistics for the current subfolder
            print(f"Subfolder: {folder_name}")
            print(f"Mean of RGB channels: {avg_folder_mean}")
            print(f"Standard Deviation of RGB channels: {avg_folder_std_dev}")
            print()

    # Calculate the average statistics for all subfolders combined
    avg_total_mean = total_mean / total_num_images
    avg_total_std_dev = total_std_dev / total_num_images

    # Print overall statistics for all subfolders
    print("Overall Statistics for All Subfolders:")
    print(f"Mean of RGB channels: {avg_total_mean}")
    print(f"Standard Deviation of RGB channels: {avg_total_std_dev}")
    return avg_total_mean, avg_total_std_dev

def calculate_rgb_statistics_for_folder(folder_path):
    # Initialize variables to accumulate pixel values
    total_mean = np.zeros(3)
    total_std_dev = np.zeros(3)
    num_images = 0

    # Loop through all JPG files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            # Read the image using OpenCV
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)

            # Convert the image to RGB format
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Calculate the mean and standard deviation of each RGB channel
            mean = np.mean(img_rgb, axis=(0, 1))
            std_dev = np.std(img_rgb, axis=(0, 1))

            # Accumulate values
            total_mean += mean
            total_std_dev += std_dev
            num_images += 1

    # Calculate the average statistics
    avg_mean = total_mean / num_images
    avg_std_dev = total_std_dev / num_images

    return avg_mean, avg_std_dev

def calculate_rgb_statistics(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Calculate the mean and standard deviation of each RGB channel
    mean = np.mean(img_rgb, axis=(0, 1))
    std_dev = np.std(img_rgb, axis=(0, 1))

    return mean, std_dev

def tensor_to_image_and_show(tensor):
    # Convert the tensor to a PIL image
    image = transforms.ToPILImage()(tensor)

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

    
    
def plot_histograms(images):
    plt.figure(figsize=(12, 6))

    for i, image_path in enumerate(images):
        # Read the image using OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Calculate the histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        # Plot the histogram
        plt.subplot(1, len(images), i + 1)
        plt.title(f'Image {i + 1}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.hist(img.ravel(), bins=256, range=[0, 256])
        plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()
    
def display_images_grid_from_dataloader(dataloader, num_images=16, grid_size=None):
    if grid_size is None:
        # Calculate the grid size automatically based on the number of images
        rows = int(math.sqrt(num_images))
        columns = math.ceil(num_images / rows)
    else:
        rows, columns = grid_size

    # Create an iterator for the DataLoader
    dataiter = iter(dataloader)
    
    # Create a new figure for displaying the grid of images
    fig, axes = plt.subplots(rows, columns, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5)

    for i in range(num_images):
        try:
            images, _ = next(dataiter)
            ax = axes[i // columns, i % columns]
            
            # Assuming you have 'tensor_to_image_and_show' function defined to display a single image from a tensor
            ax.imshow(tensor_to_image_and_show(images[0]))
            ax.axis('off')
        except StopIteration:
            break

    plt.show()

def visualize_models(model, dataloaders, num_images, class_names):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}\nactual: {class_names[labels[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
        
def imshow(inp, title=None):
    """Display image for Tensor."""
    image = transforms.ToPILImage()(inp)
    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()
    plt.pause(0.001)  # pause a bit so that plots are updated