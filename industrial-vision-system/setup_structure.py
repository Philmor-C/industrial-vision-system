import os

# Base project directory (current folder)
base_dir = os.getcwd()

structure = {
    "": ["app.py", "requirements.txt", "README.md"],
    "models": [],
    "data": [],
    "core": [
        "pipeline.py",
        "yolo.py",
        "patchcore.py",
        "cv_measure.py",
        "decision.py"
    ]
}

for folder, files in structure.items():
    folder_path = os.path.join(base_dir, folder)

    # Create folder if not root
    if folder and not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create files
    for file in files:
        file_path = os.path.join(folder_path, file)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("")  # empty file

print("Project structure created successfully.")