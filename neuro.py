import random
import shutil
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras as k
from keras import layers
from sklearn.utils import shuffle
import tensorflow as tf
import multiprocessing
import sys
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

def ensurefilepaths():
    pre = "./"
    os.makedirs(pre + "goalcontours", exist_ok=True)
    os.makedirs(pre + "model", exist_ok=True)
    os.makedirs(pre + "sol", exist_ok=True)
    os.makedirs(pre + "sol/" + "sol", exist_ok=True)
    os.makedirs(pre + "sol/" + "white",exist_ok=True)
    os.makedirs(pre + "sol/" + "arrow_to_bottom",exist_ok=True)
    os.makedirs(pre + "sol/" + "arrow_to_right",exist_ok=True)
    os.makedirs(pre + "sol/" + "double_arrow",exist_ok=True)
    os.makedirs(pre + "sol/" + "handle_bottom_to_right",exist_ok=True)
    os.makedirs(pre + "sol/" + "handle_left_to_bottom",exist_ok=True)
    os.makedirs(pre + "sol/" + "handle_right_to_bottom",exist_ok=True)
    os.makedirs(pre + "sol/" + "handle_top_to_right",exist_ok=True)
    os.makedirs(pre + "sol/" + "text",exist_ok=True)
    os.makedirs(pre + "trainingdata/base/arrow", exist_ok=True)
    os.makedirs(pre + "trainingdata/base/double_arrow", exist_ok=True)
    os.makedirs(pre + "trainingdata/base/handle/bottom_to_right", exist_ok=True)
    os.makedirs(pre + "trainingdata/base/handle/top_to_right", exist_ok=True)
    os.makedirs(pre + "trainingdata/base/sol", exist_ok=True)
    os.makedirs(pre + "trainingdata/base/white", exist_ok=True)
    os.makedirs(pre + "trainingdata/generated/arrow/to_bottom", exist_ok=True)
    os.makedirs(pre + "trainingdata/generated/arrow/to_right", exist_ok=True)
    os.makedirs(pre + "trainingdata/generated/handle/top_to_right", exist_ok=True)
    os.makedirs(pre + "trainingdata/generated/handle/bottom_to_right", exist_ok=True)
    os.makedirs(pre + "trainingdata/generated/handle/right_to_bottom", exist_ok=True)
    os.makedirs(pre + "trainingdata/generated/handle/left_to_bottom", exist_ok=True)
    os.makedirs(pre + "trainingdata/generated/sol", exist_ok=True)
    os.makedirs(pre + "trainingdata/generated/white", exist_ok=True)
    os.makedirs(pre + "trainingdata/generated/double_arrow", exist_ok=True)
    os.makedirs(pre + "trainingdata/generated/temp_whites", exist_ok=True)
    os.makedirs(pre + "trainingdata/trainingdata", exist_ok=True)
    return 0

def progress_print(string):
    print(string)
    sys.stdout.write("\033[F")  # Move cursor up one line
    sys.stdout.write("\033[K")  # Clear the line
    return 0

def predict(modified_image):

    model = k.models.load_model("./model/cnn_model_arrow_white.keras")
    if len(modified_image.shape) == 3:
            modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)
    modified_image = cv2.resize(modified_image,(100,100))
    # Flatten the image (convert 2D array to 1D)
    flattened_image = modified_image.flatten()
    image_size = 100

    # Append the label to the flattened image data
    row_with_label = list(flattened_image)  # Convert to list
    # Make a prediction on one sample
    sample = np.array(row_with_label).reshape(-1, image_size, image_size, 1) # Reshape if needed
    predictions = model.predict(sample)

    # Get the predicted class
    predicted_class = np.argmax(predictions)  # Returns the class with highest probability

    return predicted_class

def learn():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.set_visible_devices([], 'GPU')

    path_trainingdata = "./trainingdata/trainingdata/"
    output_csv = "./trainingdata/trainingdata/trainingdata.csv"
    k.backend.clear_session()
    # Load CSV data
    csv_file = output_csv # Replace with your file path
    data = pd.DataFrame()  # Initialize empty DataFrame
    chunksize = 10 ** 3  # Process 10,000 rows at a time
    loading = 0  # Progress counter

    # Read the first few rows to infer column names
    df_sample = pd.read_csv(output_csv, nrows=5)
    col_names = df_sample.columns.tolist()
    # Define dtype mapping for all columns as uint8
    dtype_map = {col: "uint8" for col in col_names}
    # Read CSV in chunks
    reader = pd.read_csv(output_csv, chunksize=chunksize, dtype=dtype_map)

    chunks = []  # List to store chunks before merging

    for chunk in reader:
        loading += chunksize*0.5  # Track actual rows loaded
        progress_print("Loaded Data: " + str(loading)) # Print progress
        chunk = shuffle(chunk, random_state=42)[:int(len(chunk)*0.5)]
        chunks.append(chunk)  # Store chunk

    # Combine all chunks into a single DataFrame
    print("finished loading data")
    data = pd.DataFrame()
    chunkcount = len(chunks)
    count = 0
    print("loading to dataframe")
    for c in chunks:
        progress_print("processing chunk: " + str(count) + "/" + str(chunkcount))
        data.add(c)
        chunks.remove(c)
    print("dataset size pre shuffle: ",len(data))

    # Shuffle the data
    print("shuffeling")
    data = shuffle(data, random_state=42)  # Randomize data order
    print("dataset size: ",len(data))
    unique_labels = data["Label"].unique()
    print("All Labels",unique_labels)
    # Separate features (X) and labels (y)
    X = data.drop(columns=['Label'])  # All columns except 'label'
    y = data['Label']  # Target column

    print("splitting")
    # Split data into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape the data for CNN if working with image-like data
    # Assuming the features are flattened pixel values (e.g., 100x100 image)
    image_size = 100  # Example: Image width and height
    X_train = np.array(X_train).reshape(-1, image_size, image_size, 1)  # Add channel dimension
    X_val = np.array(X_val).reshape(-1, image_size, image_size, 1)

    # Normalize the data (scale pixel values between 0 and 1)
    X_train = X_train / 255.0
    X_val = X_val / 255.0

    num_classes = 8
    print("num_labels",num_classes)
    print("Unique labels in y_train:", np.unique(y_train))
    print("Unique labels in y_val:", np.unique(y_val))
    print(0 in y_train)
    print(1 in y_train)
    print(2 in y_train)
    print(3 in y_train)
    print(4 in y_train)
    print(5 in y_train)
    print(6 in y_train)
    print(7 in y_train)
    # Get unique classes
    classes = np.arange(num_classes)  # Ensures labels are 0 to 7
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)

    # Convert to dictionary
    class_weights_dict = dict(enumerate(class_weights))

    print("Class Weights:", class_weights_dict)

    print("init model")
    # Define the CNN model
    model = k.Sequential([
        # First Convolutional Layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
        layers.MaxPooling2D((2, 2)),

        # Second Convolutional Layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Third Convolutional Layer
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten the output
        layers.Flatten(),

        # Fully connected dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Dropout to prevent overfitting
        layers.Dense(num_classes, activation='softmax')  # Output layer with softmax for classification
    ])

    print("compile model")
    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',  # For integer labels,
                metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Train the model
    epochs = 10  # Number of training epochs
    batch_size = 16  # Batch size for training
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)
    print("y_train unique values:", np.unique(y_train))
    print("y_val unique values:", np.unique(y_val))
    print("train")
    print("y_train: ",y_train)
    print("y_val: ",y_val)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(multiprocessing.cpu_count())
    print("Devices:", tf.config.list_physical_devices())

    history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val), class_weight=class_weights_dict)
    print(history.history)
    print("save")
    filepath = "./model/cnn_model_arrow_white.keras"
    tf.keras.models.save_model(model, filepath, overwrite=True)

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    filepath = './model/training_accuracy.png'
    # Save as an image file (e.g., PNG, JPG)
    plt.savefig(filepath)  # Save the plot
    plt.close()  # Close the plot to free memory
    return 0

def format_to_training_data_and_validat_data():

     # Define column names
    pixel_columns = [f"pixel_{i}" for i in range(10000)]
    columns = pixel_columns + ['Label']

    # Create an empty DataFrame with column names
    data = pd.DataFrame(columns=columns)

    # Save to CSV
    output_csv = "./trainingdata/trainingdata/trainingdata.csv"
    data.to_csv(output_csv, index=False)

    #mapping
    white = 0
    arrow_to_bottom = 1
    arrow_to_right = 2
    double_arrow = 3
    handle_bottom_to_right = 4
    handle_left_to_bottom = 5
    handle_right_to_bottom = 6
    handle_top_to_right = 7

    #paths
    prefix = "./trainingdata/generated/"
    path_white = prefix + "white/"
    path_arrow_to_bottom = prefix + "arrow/to_bottom/"
    path_arrow_to_right = prefix + "arrow/to_right/"
    path_double_arrow = prefix + "double_arrow/"
    path_handle_bottom_to_right = prefix + "handle/bottom_to_right/"
    path_handle_left_to_bottom = prefix + "handle/left_to_bottom/"
    path_handle_right_to_bottom = prefix + "handle/right_to_bottom/"
    path_handle_top_to_right = prefix + "handle/top_to_right/"
    print("writing trainingdata 0/8 Label: ", white)
    write_into_trainingdata(path_white,white)
    print("writing trainingdata 1/8 Label: ", arrow_to_bottom)
    write_into_trainingdata(path_arrow_to_bottom,arrow_to_bottom)
    print("writing trainingdata 2/8 Label: ", arrow_to_right)
    write_into_trainingdata(path_arrow_to_right,arrow_to_right)
    print("writing trainingdata 3/8 Label: ", double_arrow)
    write_into_trainingdata(path_double_arrow,double_arrow)
    print("writing trainingdata 4/8 Label: ", handle_bottom_to_right)
    write_into_trainingdata(path_handle_bottom_to_right,handle_bottom_to_right)
    print("writing trainingdata 5/8 Label: ", handle_left_to_bottom)
    write_into_trainingdata(path_handle_left_to_bottom,handle_left_to_bottom)
    print("writing trainingdata 6/8 Label: ", handle_right_to_bottom)
    write_into_trainingdata(path_handle_right_to_bottom,handle_right_to_bottom)
    print("writing trainingdata 7/8 Label: ", handle_top_to_right)
    write_into_trainingdata(path_handle_top_to_right,handle_top_to_right)
    print("writing trainingdata fin")

    return 0

def write_into_trainingdata(dir,label):

    output_csv = "./trainingdata/trainingdata/trainingdata.csv"
    num_files = len(os.listdir(dir))
    num_files_counter = 0
    for file in os.listdir(dir):
        num_files_counter = num_files_counter + 1
        progress_print("write file: " + str(num_files_counter) + "/" + str(num_files))
        file_path = os.path.join(dir, file)
        image = cv2.imread(file_path)
        if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(100,100))
        # Flatten the image (convert 2D array to 1D)
        flattened_image = image.flatten()

        # Append the label to the flattened image data
        row_with_label = list(flattened_image)  # Convert to list
        row_with_label.append(label)  # Add the label at the end

        # Convert to a DataFrame
        df = pd.DataFrame([row_with_label])  # Each row is one image with a label


        df.to_csv(output_csv, mode='a', index=False, header=False)
    print("wrote lines: ", num_files)
    return 0

def make_training_data():
    dirpre = "./trainingdata/base/"
    traget_dirpre = "./trainingdata/generated/"
    delete_png_files(traget_dirpre)
    post = "white/"
    whites_path = dirpre + post
    print("generate_whites")
    generate_whites(dirpre + post,traget_dirpre + post)
    post = "sol/"
    sol_path = dirpre + post
    post = "arrow/"
    print("generate_arrows")
    generate_arrows(dirpre + post, whites_path, sol_path, traget_dirpre + post)
    post = "handle/top_to_right/"
    print("generate_handles")
    generate_arrowhandles(dirpre + post, whites_path, sol_path, traget_dirpre +"handle/") #arrows top to right
    post = "handle/bottom_to_right/"
    generate_mirror_arrowhandles(dirpre + post, whites_path, sol_path, traget_dirpre+"handle/") #arrows bottom to right
    post = "double_arrow/"
    print("generate_double_arrows")
    generate_double_arrow(dirpre + post, whites_path, sol_path, traget_dirpre + post)
    generate_double_arrow_from_arrows(dirpre + "arrow", whites_path, sol_path, traget_dirpre + post)

def add_sol(image,sol_dir,whites_dir,target_dir):
    for solname in os.listdir(sol_dir):
        sol_path = os.path.join(sol_dir, solname)
        if os.path.isfile(sol_path):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sol = cv2.imread(sol_path)  
            if len(sol.shape) == 3:
                sol= cv2.cvtColor(sol, cv2.COLOR_BGR2GRAY)

            # Ändere die Größe von white auf die Größe von image
            image_height, image_width = image.shape[:2]
            sol = cv2.resize(sol, (image_width, image_height), interpolation=cv2.INTER_AREA)

            # Gewichte kombinieren
            fused = cv2.addWeighted(sol, 0.5, image, 0.5, 0)

            fused[fused <= 200] = 0
            cv2.imwrite(name_in_target(target_dir), fused)
            whites = os.listdir(whites_dir)
            delete_png_files("./trainingdata/generated/temp_whites/")
            for i in range(0,int(len(whites)/100)):
                filename =whites[random.randint(0,len(whites)-1)]
                src_file = os.path.join(whites_dir,filename)
                dest_file = os.path.join("./trainingdata/generated/temp_whites/", filename)
                shutil.copy(src_file, dest_file)

            add_whites(fused,"./trainingdata/generated/temp_whites/",target_dir)
    return 0

def generate_double_arrow_from_arrows(dir, whites_dir, sol_dir, target_dir):
    num_arrows = count_files(dir)
    count = 0
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            to_right = cv2.imread(file_path)

            for filename2 in os.listdir(dir):
                file_path2 = os.path.join(dir, filename2)
                
                # Ensure it's a file (not a directory)
                if os.path.isfile(file_path2):
                    progress_print(str(count) + "/" + str(num_arrows*num_arrows))
                    count = count + 1
                    to_bottom = cv2.imread(file_path)
                    image = cv2.addWeighted(to_bottom, 0.5,to_right, 0.5, 0)
                    image[image <= 200] = 0
                    cv2.imwrite(name_in_target(target_dir),image)
                    add_whites(image,whites_dir,target_dir)
                    add_sol(image,sol_dir,whites_dir,target_dir)
    return 0

def generate_double_arrow(dir, whites_dir, sol_dir, target_dir):
    num_arrows = count_files(dir)
    count = 0
    for filename in os.listdir(dir):
        
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            progress_print(str(count) + "/" + str(num_arrows))
            count = count + 1
            image = cv2.imread(file_path)
            cv2.imwrite(name_in_target(target_dir), image)
            add_whites(image,whites_dir,target_dir)
            add_sol(image,sol_dir,whites_dir,target_dir)
    return 0

def generate_arrows(dir, whites_dir, sol_dir, target_dir):
    num_arrows = count_files(dir)
    count = 0
    # Loop through all files in the directory
    for filename in os.listdir(dir):
        
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            progress_print(str(count) + "/" + str(num_arrows))
            count = count + 1
            image = cv2.imread(file_path)
            cv2.imwrite(name_in_target(target_dir + "to_right/"), image)
            add_whites(image,whites_dir,target_dir + "to_right/")
            add_sol(image,sol_dir,whites_dir,target_dir + "to_right/")

            # Bild um 90° nach rechts drehen
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Gedrehtes Bild speichern
            cv2.imwrite(name_in_target(target_dir + "to_bottom/"), rotated_image)
            add_whites(rotated_image,whites_dir,target_dir + "to_bottom/")
            add_sol(image,sol_dir,whites_dir,target_dir + "to_bottom/")
    return 0

def generate_mirror_arrowhandles(dir, whites_dir, sol_dir, target_dir):
    count = 0
    num_arrows = count_files(dir)
    for filename in os.listdir(dir):
        
        file_path = os.path.join(dir, filename)
        progress_print(str(count) + "/" + str(num_arrows))
        count = count + 1
        if os.path.isfile(file_path):
            image = cv2.flip(cv2.imread(file_path),0)
            generate_arrowhandles_core(image, whites_dir, sol_dir, target_dir)
    return 0

def generate_arrowhandles(dir, whites_dir, sol_dir, target_dir):
    count = 0
    num_arrows = count_files(dir)
    for filename in os.listdir(dir):
        
        file_path = os.path.join(dir, filename)
        progress_print(str(count) + "/" + str(num_arrows))
        count = count + 1
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            generate_arrowhandles_core(image, whites_dir, sol_dir, target_dir)
    return 0

def generate_arrowhandles_core(image,whites_dir, sol_dir, target_dir):
    cv2.imwrite(name_in_target(target_dir + "top_to_right/"), image)
    add_whites(image,whites_dir,target_dir + "top_to_right/")
    add_sol(image,sol_dir,whites_dir,target_dir + "top_to_right/")

    cv2.imwrite(name_in_target(target_dir + "bottom_to_right/"),cv2.flip(image, 0))
    add_whites(cv2.flip(image, 0),whites_dir,target_dir + "bottom_to_right/")
    add_sol(cv2.flip(image, 0),sol_dir,whites_dir,target_dir + "bottom_to_right/")
    # Bild um 90° nach rechts drehen
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Gedrehtes Bild speichern
    cv2.imwrite(name_in_target(target_dir + "right_to_bottom/"), rotated_image)
    add_whites(rotated_image,whites_dir,target_dir + "right_to_bottom/")
    add_sol(rotated_image,sol_dir,whites_dir,target_dir + "right_to_bottom/")
    
    cv2.imwrite(name_in_target(target_dir + "left_to_bottom/"),cv2.flip(rotated_image, 1))
    add_whites(cv2.flip(rotated_image, 0),whites_dir,target_dir + "left_to_bottom/")
    add_sol(cv2.flip(rotated_image, 0),sol_dir,whites_dir,target_dir + "left_to_bottom/")

    return 0

def add_whites(image,whites_dir,target_dir):
    for whitename in os.listdir(whites_dir):
        white_path = os.path.join(whites_dir, whitename)

        if os.path.isfile(white_path):
            for i in range(0,4):
                if i == 0 :
                    white = cv2.imread(white_path)   
                elif i == 1:
                    white = cv2.imread(white_path)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                elif i == 2:
                    white = cv2.imread(white_path)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                elif i == 3:
                    white = cv2.imread(white_path)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)

                for j in range(-1,2):
                    white = cv2.flip(white, j)
                    if len(image.shape) == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Ändere die Größe von white auf die Größe von image
                    image_height, image_width = image.shape[:2]
                    if len(white.shape) == 3:
                        white = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
                    white = cv2.resize(white, (image_width, image_height), interpolation=cv2.INTER_AREA)

                    # Gewichte kombinieren
                    fused = cv2.addWeighted(white, 0.5, image, 0.5, 0)

                    fused[fused <= 200] = 0
                    cv2.imwrite(name_in_target(target_dir), fused)
    return 0

def generate_whites(dir, target_dir):
    num_arrows = count_files(dir)
    count = 0
    for whitename in os.listdir(dir):
        white_path = os.path.join(dir, whitename)
        progress_print(str(count) + "/" + str(num_arrows))
        count = count + 1
        if os.path.isfile(white_path):
            for i in range(0,4):
                if i == 0 :
                    white = cv2.imread(white_path)   
                elif i == 1:
                    white = cv2.imread(white_path)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                elif i == 2:
                    white = cv2.imread(white_path)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                elif i == 3:
                    white = cv2.imread(white_path)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)
                    white = cv2.rotate(white, cv2.ROTATE_90_CLOCKWISE)

                for j in range(-1,2):
                    white = cv2.flip(white, j)

                    # Konvertiere white ebenfalls in Graustufen, um die Anzahl der Kanäle anzupassen
                    white_gray = cv2.cvtColor(white, cv2.COLOR_RGB2GRAY)

                    cv2.imwrite(name_in_target(target_dir), white_gray)

    return 0

def count_files(dir):
    return len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])

def name_in_target(dir):
    n = count_files(dir)
    return dir + str(n+1) + ".png"

def delete_png_files(directory):

    if not os.path.exists(directory):
        print(f"Das Verzeichnis '{directory}' existiert nicht.")
        return

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):  # Prüfen, ob es eine .png-Datei ist
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)  # Datei löschen
                except Exception as e:
                    print(f"Fehler beim Löschen von {file_path}: {e}")