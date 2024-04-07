# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 18:07:15 2024

@author: barre
"""

import os
import shutil
import random

random_seed = 42
random.seed(random_seed)

# Source directory containing compressed archive folders
source_directory = '/home/fletcher.barrett/Sex_Classifier/Images_original/'

# Destination directory where 'train', 'validation', and 'test' folders will be created
destination_directory = '/home/fletcher.barrett/Sex_Classifier/Images_cv/'

# Create destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Create 'development', and 'test' folders in the destination directory
development_folder = os.path.join(destination_directory, 'Development')
test_folder = os.path.join(destination_directory, 'Test')
os.makedirs(development_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Split ratio: 83% development, 17% test. This is so that 5-fold cv will have splits approximately equal in size to testing
development_ratio = 0.83
test_ratio = 0.17

# Function to split files into train, validation, and test sets while preserving class proportion
def split_files(files):
    # Separate files into male and female classes for each scanner type
    male_philips_files = [f for f in files if (f.endswith('_M.nii.gz') and 'philips' in f)]
    female_philips_files = [f for f in files if (f.endswith('_F.nii.gz') and 'philips' in f)]
    male_ge_files = [f for f in files if (f.endswith('_M.nii.gz') and 'ge' in f)]
    female_ge_files = [f for f in files if (f.endswith('_F.nii.gz') and 'ge' in f)]
    male_siemens_files = [f for f in files if (f.endswith('_M.nii.gz') and 'siemens' in f)]
    female_siemens_files = [f for f in files if (f.endswith('_F.nii.gz') and 'siemens' in f)]
    
    # Shuffle each class independently
    random.shuffle(male_philips_files)
    random.shuffle(female_philips_files)
    random.shuffle(male_ge_files)
    random.shuffle(female_ge_files)
    random.shuffle(male_siemens_files)
    random.shuffle(female_siemens_files)
    
    # Calculate number of files for each class in train, validation, and test sets
    total_male_philips_files = len(male_philips_files)
    total_female_philips_files = len(female_philips_files)
    total_male_ge_files = len(male_ge_files)
    total_female_ge_files = len(female_ge_files)
    total_male_siemens_files = len(male_siemens_files)
    total_female_siemens_files = len(female_siemens_files)
    
    development_male_philips_count = int(total_male_philips_files * development_ratio)
    development_female_philips_count = int(total_female_philips_files * development_ratio)
    development_male_ge_count = int(total_male_ge_files * development_ratio)
    development_female_ge_count = int(total_female_ge_files * development_ratio)
    development_male_siemens_count = int(total_male_siemens_files * development_ratio)
    development_female_siemens_count = int(total_female_siemens_files * development_ratio)
    
    # Select files for each class in train, validation, and test sets
    development_set = male_philips_files[:development_male_philips_count] + female_philips_files[:development_female_philips_count] + \
                      male_ge_files[:development_male_ge_count] + female_ge_files[:development_female_ge_count] + \
                      male_siemens_files[:development_male_siemens_count] + female_siemens_files[:development_female_siemens_count]
    test_set = male_philips_files[development_male_philips_count:] + female_philips_files[development_female_philips_count:] + \
               male_ge_files[development_male_ge_count:] + female_ge_files[development_female_ge_count:] + \
               male_siemens_files[development_male_siemens_count:] + female_siemens_files[development_female_siemens_count:]
    
    # Shuffle the sets
    random.shuffle(development_set)
    random.shuffle(test_set)
    
    return development_set, test_set

# Function to create 'Male' and 'Female' folders within a directory and move files accordingly
def organize_files(folder_path):
    male_folder = os.path.join(folder_path, 'Male')
    female_folder = os.path.join(folder_path, 'Female')
    filenames = os.listdir(folder_path)
    os.makedirs(male_folder, exist_ok=True)
    os.makedirs(female_folder, exist_ok=True)
    for filename in filenames:
        src = os.path.join(folder_path, filename)
        if filename.endswith('_F.nii.gz'):
            # Move files ending with '_F.nii.gz' to 'Female' folder
            dst = os.path.join(female_folder, filename)
            shutil.move(src, dst)
            print(f"Copied {filename} to 'Female' folder.")
        elif filename.endswith('_M.nii.gz'):
            # Move files ending with '_M.nii.gz' to 'Male' folder
            dst = os.path.join(male_folder, filename)
            shutil.move(src, dst)
            print(f"Copied {filename} to 'Male' folder.")
        else:
            print(f"Ignoring {filename} as it doesn't match the naming convention.")

# Iterate through files in the source directory
all_files = os.listdir(source_directory)
development_set, test_set = split_files(all_files)

# Move files to 'development' folder
for filename in development_set:
    src = os.path.join(source_directory, filename)
    dst = os.path.join(development_folder, filename)
    shutil.copy(src, dst)

# Move files to 'test' folder
for filename in test_set:
    src = os.path.join(source_directory, filename)
    dst = os.path.join(test_folder, filename)
    shutil.copy(src, dst)

# Organize files within each set
organize_files(development_folder)
organize_files(test_folder)


