import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to preprocessed images
PREPROCESSED_NORMAL_DIR = '/content/drive/MyDrive/preprocessed_images/normal/'
PREPROCESSED_GLUCOMA_DIR = '/content/drive/MyDrive/preprocessed_images/glaucoma/'
PREPROCESSED_MASK_DIR = '/content/drive/MyDrive/preprocessed_mask/'

# Directories for train, validation, and test sets
TRAIN_DIR = '/content/drive/MyDrive/dataset/train/'
VALIDATION_DIR = '/content/drive/MyDrive/dataset/validation/'
TEST_DIR = '/content/drive/MyDrive/dataset/test/'

# Function to create directories if they don't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create train, validation, and test directories
create_directory(TRAIN_DIR)
create_directory(VALIDATION_DIR)
create_directory(TEST_DIR)

# Function to split data and move images
def split_data_and_move(source_dir, train_dir, validation_dir, test_dir):
    filenames = os.listdir(source_dir)
    train_filenames, test_filenames = train_test_split(filenames, test_size=0.3, random_state=42)
    val_filenames, test_filenames = train_test_split(test_filenames, test_size=2/3, random_state=42)
    
    for filename in train_filenames:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(train_dir, filename))
    for filename in val_filenames:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(validation_dir, filename))
    for filename in test_filenames:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(test_dir, filename))

# Split and move normal images
split_data_and_move(PREPROCESSED_NORMAL_DIR, os.path.join(TRAIN_DIR, 'normal'), os.path.join(VALIDATION_DIR, 'normal'), os.path.join(TEST_DIR, 'normal'))

# Split and move glaucoma images
split_data_and_move(PREPROCESSED_GLUCOMA_DIR, os.path.join(TRAIN_DIR, 'glaucoma'), os.path.join(VALIDATION_DIR, 'glaucoma'), os.path.join(TEST_DIR, 'glaucoma'))

# Split and move mask images
split_data_and_move(PREPROCESSED_MASK_DIR, os.path.join(TRAIN_DIR, 'mask'), os.path.join(VALIDATION_DIR, 'mask'), os.path.join(TEST_DIR, 'mask'))