import os
import gzip
import numpy as np
from PIL import Image
from struct import pack
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_path, image_name, label_name, train_image_name, train_label_name, test_image_name, test_label_name, test_size=0.2, random_state=42):
        self.image_root_folder = data_path
        self.image_name = image_name
        self.label_name = label_name
        self.train_image_name = train_image_name
        self.train_label_name = train_label_name
        self.test_image_name = test_image_name
        self.test_label_name = test_label_name
        self.test_size = test_size
        self.random_state = random_state
    
    def load_mnist(self, image_path, label_path):
        """Load MNIST data from `path`"""
        labels_path = os.path.join(self.image_root_folder, label_path)
        images_path = os.path.join(self.image_root_folder, image_path)
        
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.float32, offset=16).reshape(len(labels), 784)

        labels = labels.reshape(len(labels), 1)
        
        return images, labels
    
    def convert_to_mnist(self):
        """Convert jpg images from multiple folders to MNIST format"""
        # Initialize image data and label arrays
        images = []
        labels = []
        
        # Get all subdirectories in the root folder
        subdirs = [d for d in os.listdir(self.image_root_folder) if os.path.isdir(os.path.join(self.image_root_folder, d))]
        
        # Iterate over all subdirectories
        for label, subdir in enumerate(subdirs):
            folder_path = os.path.join(self.image_root_folder, subdir)
            
            # Get all jpg image file names in the current folder
            image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            
            # Iterate over all image files in the current folder
            for image_file in image_files:
                # Read the image and convert to grayscale
                image = Image.open(os.path.join(folder_path, image_file)).convert('L')
                
                # Resize the image to 28x28 pixels
                image = image.resize((28, 28))
                
                # Convert image data to numpy array and normalize
                image_data = np.array(image) / 255.0
                
                # Flatten the image data to a 1D array
                image_data = image_data.reshape(-1)
                
                # Append the image data to the array
                images.append(image_data)
                
                # Append the label to the array
                labels.append(label)
        
        # Convert image data and labels to numpy arrays
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)
        
        # Save as MNIST format gzip files
        with gzip.open(os.path.join(self.image_root_folder, self.image_name), 'wb') as f:
            f.write(pack('>IIII', 2051, len(images), 28, 28))
            f.write(images.tobytes())
        
        with gzip.open(os.path.join(self.image_root_folder, self.label_name), 'wb') as f:
            f.write(pack('>II', 2049, len(labels)))
            f.write(labels.tobytes())
        
        print(f"Converted {len(images)} images from {len(subdirs)} folders to MNIST format and saved to {self.image_root_folder}")
    
    def create_labels(self):
        """Create label files for each image in the subdirectories of the root folder"""
        # Get all subdirectories in the root folder
        subdirs = [d for d in os.listdir(self.image_root_folder) if os.path.isdir(os.path.join(self.image_root_folder, d))]
        
        # Iterate over all subdirectories
        for label, subdir in enumerate(subdirs):
            folder_path = os.path.join(self.image_root_folder, subdir)
            
            # Iterate over all files in the current folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg"):
                    file_name_without_ext = os.path.splitext(filename)[0]
                    txt_file_path = os.path.join(folder_path, file_name_without_ext + ".txt")
                    with open(txt_file_path, "w") as txt_file:
                        txt_file.write(str(label))
        
        print("Label files created successfully!")

    def split_data(self):
        """Split the dataset into training and testing sets"""
        # Load the MNIST dataset
        images, labels = self.load_mnist(self.image_name, self.label_name)
        
        # Split the dataset into training and testing sets
        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, test_size=self.test_size, random_state=self.random_state, stratify=labels
        )
        
        # Save the training set
        with gzip.open(os.path.join(self.image_root_folder, self.train_image_name), 'wb') as f:
            f.write(pack('>IIII', 2051, len(train_images), 28, 28))
            f.write(train_images.tobytes())
        
        with gzip.open(os.path.join(self.image_root_folder, self.train_label_name), 'wb') as f:
            f.write(pack('>II', 2049, len(train_labels)))
            f.write(train_labels.tobytes())
        
        # Save the testing set
        with gzip.open(os.path.join(self.image_root_folder, self.test_image_name), 'wb') as f:
            f.write(pack('>IIII', 2051, len(test_images), 28, 28))
            f.write(test_images.tobytes())
        
        with gzip.open(os.path.join(self.image_root_folder, self.test_label_name), 'wb') as f:
            f.write(pack('>II', 2049, len(test_labels)))
            f.write(test_labels.tobytes())
        
        print(f"Dataset split into training set ({len(train_images)} samples) and testing set ({len(test_images)} samples)")