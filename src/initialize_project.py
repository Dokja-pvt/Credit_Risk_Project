import os

def initialize_workspace():

    directories = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src",
        "reports/figures"
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✔️ Created directory: {directory}")
        else:
            print(f"⚠️ Directory already exists: {directory}")

    init_path = "src/__init__.py"
    if not os.path.exists(init_path):
        with open(init_path, 'w') as f:
            pass
        print(f"✔️ Created: {init_path}")

    print("\nWorkspace structure finalized.")
    print("👉 ACTION REQUIRED: Place 'application_train.csv' and 'previous_application.csv' inside 'data/raw/'")

if __name__ == "__main__":
    initialize_workspace()