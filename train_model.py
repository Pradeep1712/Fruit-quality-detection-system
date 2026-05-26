import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Ensure your dataset is organized into two main folders:
# 1. 'dataset/fruit_types' -> Subfolders: 'Apple', 'Banana', 'Orange'
# 2. 'dataset/freshness'   -> Subfolders: 'Fresh', 'Rotten'

DATASET_TYPE_DIR = 'dataset/fruit_types'
DATASET_FRESHNESS_DIR = 'dataset/freshness'

# Must match the PIL resizing in the Flask app
IMG_WIDTH, IMG_HEIGHT = 100, 100  
BATCH_SIZE = 32
EPOCHS = 25

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
def build_cnn_model(num_classes):
    """Builds a custom CNN optimized for 100x100 images."""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Fully Connected Classifier
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Prevents overfitting
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ==========================================
# DATA AUGMENTATION & LOADING
# ==========================================
def create_data_generators(dataset_dir):
    """Creates training and validation generators with data augmentation."""
    datagen = ImageDataGenerator(
        rescale=1.0/255.0, # Normalizes pixel values
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 # Reserves 20% of images for validation
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_gen, val_gen

# ==========================================
# TRAINING EXECUTION
# ==========================================
def train_and_save_model(dataset_dir, save_filename):
    print(f"\n--- Training Model for dataset: {dataset_dir} ---")
    
    train_gen, val_gen = create_data_generators(dataset_dir)
    num_classes = train_gen.num_classes
    
    model = build_cnn_model(num_classes)
    
    # Callbacks to save the best version and stop early if it stops improving
    checkpoint = ModelCheckpoint(save_filename, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping]
    )
    
    print(f"Model successfully saved as: {save_filename}\n")

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    
    # 1. Train the Fruit Type Model
    if os.path.exists(DATASET_TYPE_DIR):
        train_and_save_model(DATASET_TYPE_DIR, 'models/local_fruit_final.h5')
    else:
        print(f"Warning: '{DATASET_TYPE_DIR}' not found. Skipping Type training.")

    # 2. Train the Freshness Model
    if os.path.exists(DATASET_FRESHNESS_DIR):
        train_and_save_model(DATASET_FRESHNESS_DIR, 'models/local_rotten_lr2_final.h5')
    else:
        print(f"Warning: '{DATASET_FRESHNESS_DIR}' not found. Skipping Freshness training.")