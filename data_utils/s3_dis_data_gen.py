import os
import shutil
from tqdm import tqdm

import argparse


def process_files(main_folder, class_dict):
    # Get list of area folders
    area_folders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
    
    # Iterate over all area folders in the main folder with tqdm progress bar
    for areafolder in tqdm(area_folders, desc="Processing all area folders"):
        areafolder_path = os.path.join(main_folder, areafolder)

        # Get list of room subfolders
        room_folders = [f for f in os.listdir(areafolder_path) if os.path.isdir(os.path.join(areafolder_path, f))]
        
        # Iterate over all subfolders in the main folder with tqdm progress bar
        for roomfolder in tqdm(room_folders, desc=f"Processing room folders in {areafolder}"):
            roomfolder_path = os.path.join(areafolder_path, roomfolder)
            annotations_folder = os.path.join(roomfolder_path, 'Annotations')
            
            # Check if the Annotations folder exists
            if os.path.exists(annotations_folder):
                output_file = os.path.join(areafolder_path, f'{roomfolder}.txt')
                
                # Open the output file in write mode
                with open(output_file, 'w') as outfile:
                    # Get list of txt files in Annotations folder
                    txt_files = [f for f in os.listdir(annotations_folder) if f.endswith(".txt")]
                    
                    # Iterate over all files in the Annotations folder with tqdm progress bar
                    for filename in tqdm(txt_files, desc=f"Processing files in {roomfolder}", leave=False):
                        file_path = os.path.join(annotations_folder, filename)
                        class_found = False

                        # Check if the file name contains any class name
                        for class_name, identifier in class_dict.items():
                            if class_name in filename:
                                class_found = True
                                # Open and read the file
                                with open(file_path, 'r') as infile:
                                    lines = infile.readlines()
                                    # Append the identifier to each line and write to the output file
                                    for line in lines:
                                        outfile.write(line.strip() + ' ' + str(identifier) + '\n')
                                break

                        if not class_found:
                            # If no class name was found in the file name, copy the lines as they are
                            print("Error: class not found in " + file_path + ".")
                            with open(file_path, 'r') as infile:
                                lines = infile.readlines()
                                for line in lines:
                                    outfile.write(line.strip() + '\n')
                
                # Delete all contents of the subfolder
                for filename in os.listdir(roomfolder_path):
                    file_path = os.path.join(roomfolder_path, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                
                # Move the created file into the subfolder
                new_output_path = os.path.join(roomfolder_path, f'{roomfolder}.txt')
                shutil.move(output_file, new_output_path)

if __name__ == "__main__":
    # Define the path to the main folder containing the subfolders
    parser = argparse.ArgumentParser(description='Process the files in the Stanford Indoor 3D dataset')
    # Argument that is the directory where the data is
    parser.add_argument('--data_dir', type=str, help='The directory where the Areas are')
    args = parser.parse_args()

    main_folder = args.data_dir
    if main_folder is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        up_dir = os.path.abspath(os.path.join(current_dir, '../'))
        main_folder = os.path.join( os.path.join(up_dir, 'data'), 'stanford_indoor3d')
    print(f"The main folder is: {main_folder}")

    # Define the dictionary of class names and their corresponding identifiers
    class_dict = {
        'ceiling'  : 0, 
        'floor'    : 1, 
        'wall'     : 2, 
        'beam'     : 3, 
        'column'   : 4, 
        'window'   : 5,
        'door'     : 6, 
        'table'    : 7, 
        'chair'    : 8, 
        'sofa'     : 9, 
        'bookcase' : 10, 
        'board'    : 11,
        'stairs'   : 12,
        'clutter'  : 13
    }
    
    if main_folder is not None:
        # Process the files and generate the concatenated output files
        process_files(main_folder, class_dict)
