import torch.cuda
import os
import argparse
from ultralytics import YOLO
from ultralytics import settings
import clearml
from clearml import Task

def initialize_cuda_settings():
    # Specify some CUDA setttings
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

def initialize_clearml():
    # Init ClearML
    os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml'
    os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'
    os.environ['CLEARML_FILES_HOST'] = 'https://files.clear.ml'
    os.environ['CLEARML_API_ACCESS_KEY'] = 'VHXADOOTOSMBAGRU5E4T'
    os.environ['CLEARML_API_SECRET_KEY'] = 'TCOeMkjTwyXGnWiFEgYs9nvWeiHHtGiJ2iwk8VhJo7M1CpiYTe'

def initialize_yolo_settings(datasets_dir: str, runs_dir: str):

    # Create the directories
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    # Update YOLO native settings file to look for datasets and store results in custom dirs
    settings.update({'datasets_dir': datasets_dir, 'runs_dir': runs_dir})

def assign_device():
    # Detect CUDA devices
    num_cuda_devices = torch.cuda.device_count()

    # Assign value to the device selection variable
    if num_cuda_devices == 1:
        device = "0"
    elif num_cuda_devices > 1:
        device = [int(i) for i in range(num_cuda_devices)]
    else:
        device = None

    print(f"CUDA available: {torch.cuda.is_available()}, Using CUDA device(s): {device}")

    return device

def assign_workers():
    # Detect CUDA devices
    num_cuda_devices = torch.cuda.device_count()

    # Assign value to the number of workers based on the number of cores used per GPU
    num_cpu_cores = os.cpu_count()

    num_workers = num_cpu_cores // num_cuda_devices

    print(f"Number of CPU cores: {os.cpu_count()}, Using workers: {num_workers}")

    return num_workers

def assign_batch_size(batch_size_per_gpu: int = 224):
    # Detect CUDA devices
    num_cuda_devices = torch.cuda.device_count()

    # Calculate batch size
    batch_size = batch_size_per_gpu * num_cuda_devices

    print(f"Batch size per GPU: {batch_size_per_gpu}, Using batch size: {batch_size}")

    return batch_size

def get_folder_names(directory):
    # Get a list of all folder names in the given directory to get dataset folders. Manually excluding the source data.
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name != 'Model training frames']

def update_dataset_paths(folder: str, datasets_dir: str):

    def update_txt_file(txt_file_path: str):

        # Read the contents of the file
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()

        # Replace "storage" with "auto" in each path
        modified_lines = [line.replace('storage', args.path_update) for line in lines]

        # Split the path and the extension
        file_path, file_extension = os.path.splitext(txt_file_path)

        # Append '_updated' to the file name
        new_txt_file_path = file_path + '_updated' + file_extension

        # Save the modified paths back to the file
        with open(new_txt_file_path, 'w') as file:
            file.writelines(modified_lines)

        print(f"File paths have been updated in {txt_file_path}")

    def update_yaml_file(yaml_file_path: str):
        import yaml

        # Load the YAML file
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Modify the paths in the YAML data
        data['train'] = data['train'].replace('storage', args.path_update)
        data['val'] = data['val'].replace('storage', args.path_update)

        # Split the path and the extension of the YAML file
        file_path, file_extension = os.path.splitext(yaml_file_path)

        # Append '_updated' to the file name
        new_yaml_file_path = file_path + '_updated' + file_extension

        # Save the modified data back to a new YAML file
        with open(new_yaml_file_path, 'w') as file:
            yaml.dump(data, file)

        print(f"YAML file paths have been updated in {new_yaml_file_path}")

    # Define the path to the original YAML file
    yaml_file_path = os.path.join(datasets_dir, folder, f"{folder}.yaml")

    # Change the paths in yaml
    update_yaml_file(yaml_file_path)

    for ds in ["train", "val"]:
        # Define the path to the text file
        txt_file_path = os.path.join(datasets_dir, folder, "images", ds, f"{ds}.txt")

        # Change the paths in txt
        update_txt_file(txt_file_path)


def main(args):

    # Review args
    print(f"Received arguments: {args}")

    # Initiate task
    task = Task.init(project_name=args.project_name, task_name=args.dataset)

    # Set some CUDA environment variables
    initialize_cuda_settings()

    # Initialize clearML auth variables
    initialize_clearml()

    # Get working directory if not specified
    if args.workdir is None:

        # Get current directory (from where the script is being run) - this should be the homedir
        current_directory = os.getcwd()

    else:

        # Assigns value specified by the user
        current_directory = args.workdir
    print(f"Current directory: {current_directory}")

    # Get datasets directory
    if args.datasets_dir is None:

        datasets_dir = os.path.join(current_directory, 'datasets')
    print(f"Datasets directory set to: {datasets_dir}")

    # Get runs directory
    runs_dir = os.path.join(current_directory, 'runs')
    print(f"Runs directory set to: {runs_dir}")

    # Update multiple settings
    initialize_yolo_settings(datasets_dir, runs_dir)

    # Get list of datasets
    if args.dataset is None:

        return

    # Paths will be updated to reflect the binding of directories or any necessary modifications
    if args.path_update is not None:

        # Update paths
        update_dataset_paths(args.dataset, datasets_dir)

    # Get devices for training
    device = assign_device()

    # Get number fo workers
    num_workers = assign_workers()

    # Get bacth size
    batch_size = assign_batch_size()

    # Load a pretrained model
    model = YOLO(args.model)

    # Put together the path to the dataset data file
    dataset_data = os.path.normpath(os.path.join(datasets_dir, args.dataset, f'{args.dataset}.yaml')) if args.path_update is not None else os.path.normpath(os.path.join(datasets_dir, args.dataset, f'{args.dataset}_updated.yaml'))

    # Train the model
    try:
        results = model.train(data=dataset_data, epochs=100, batch=batch_size, imgsz=640, workers=0, device=device,
                              verbose=True, project=args.project_name, name=args.dataset)
    except:
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your script description.')

    # Example arguments. Modify them according to your needs.
    parser.add_argument('--dataset', type=str, default=None, help='Dataset folder name')
    parser.add_argument('--datasets_dir', type=str, default=None, help='Path to the dataset folder')
    parser.add_argument('--workdir', type=str, default=None, help='Path to the working directory (homedir)')
    parser.add_argument('--path_update', type=str, default=None, help='Path to dataset images and/or labels will be updated by replacing "storage" with a preset path (to reflect binding)')
    parser.add_argument('--project_name', type=str, default="YOLOv8 train", help='Project name')
    parser.add_argument('--model', type=str, default="yolov8n.pt", help='Name or path to a pretrained model')

    args = parser.parse_args()
    main(args)