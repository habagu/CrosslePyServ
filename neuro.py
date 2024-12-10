import cv2
import numpy as np
import os

def make_training_data():
    dirpre = "./trainingdata/base/"
    traget_dirpre = "./trainingdata/generated/"
    delete_png_files(traget_dirpre)
    post = "white/"
    whites_path = traget_dirpre + post
    print("generate_whites")
    generate_whites(dirpre + post,traget_dirpre + post)
    post = "sol/"
    sol_path = traget_dirpre + post
    post = "arrow/"
    print("generate_arrows")
    generate_arrows(dirpre + post, whites_path, sol_path, traget_dirpre + post)
    post = "handle/"
    print("generate_handles")
    generate_arrowhandles(dirpre + post, whites_path, sol_path) #arrows top to right
    post = "handle/"
    generate_mirror_arrowhandles(dirpre + post, whites_path, sol_path) #arrows bottom to right
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
            add_whites(fused,whites_dir,target_dir)
    return 0

def generate_double_arrow_from_arrows(dir, whites_dir, sol_dir, target_dir):
    for filename in os.listdir(dir):
        print("Quelle file: ",filename)
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            to_right = cv2.imread(file_path)

            for filename2 in os.listdir(dir):
                file_path2 = os.path.join(dir, filename2)
                
                # Ensure it's a file (not a directory)
                if os.path.isfile(file_path2):
                    to_bottom = cv2.imread(file_path)
                    image = cv2.addWeighted(to_bottom, 0.5,to_right, 0.5, 0)
                    image[image <= 200] = 0
                    cv2.imwrite(name_in_target(target_dir),image)
                    add_whites(image,whites_dir,target_dir)
                    add_sol(image,sol_dir,whites_dir,target_dir)
    return 0

def generate_double_arrow(dir, whites_dir, sol_dir, target_dir):
    for filename in os.listdir(dir):
        print("Quelle file: ",filename)
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            cv2.imwrite(name_in_target(target_dir), image)
            add_whites(image,whites_dir,target_dir)
            add_sol(image,sol_dir,whites_dir,target_dir)
    return 0

def generate_arrows(dir, whites_dir, sol_dir, target_dir):
    # Loop through all files in the directory
    for filename in os.listdir(dir):
        print("Quelle file: ",filename)
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
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
    for filename in os.listdir(dir):
        print("Quelle file: ",filename)
        file_path = os.path.join(dir, filename)
        
        if os.path.isfile(file_path):
            image = cv2.flip(cv2.imread(file_path),0)
            generate_arrowhandles_core(image, whites_dir, sol_dir, target_dir)
    return 0

def generate_arrowhandles(dir, whites_dir, sol_dir, target_dir):
    for filename in os.listdir(dir):
        print("Quelle file: ",filename)
        file_path = os.path.join(dir, filename)
        
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            generate_arrowhandles_core(image, whites_dir, target_dir)
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
                    white = cv2.cvtColor(white, cv2.COLOR_BGR2RGB)
                    white = cv2.resize(white, (image_width, image_height), interpolation=cv2.INTER_AREA)

                    # Konvertiere white ebenfalls in Graustufen, um die Anzahl der Kanäle anzupassen
                    white_gray = cv2.cvtColor(white, cv2.COLOR_RGB2GRAY)

                    # Gewichte kombinieren
                    fused = cv2.addWeighted(white_gray, 0.5, image, 0.5, 0)

                    fused[fused <= 200] = 0
                    cv2.imwrite(name_in_target(target_dir), fused)
    return 0

def generate_whites(dir, target_dir):
    for filename in os.listdir(dir):
        print("Quelle file: ",filename)
        file_path = os.path.join(dir, filename)
        
        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            image = cv2.imread(file_path)
            cv2.imwrite(name_in_target(target_dir) , image)

            # Bild um 90° nach rechts drehen
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Gedrehtes Bild speichern
            cv2.imwrite(name_in_target(target_dir), rotated_image)

            # Bild um 90° nach rechts drehen
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Gedrehtes Bild speichern
            cv2.imwrite(name_in_target(target_dir), rotated_image)

            # Bild um 90° nach rechts drehen
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Gedrehtes Bild speichern
            cv2.imwrite(name_in_target(target_dir), rotated_image)

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