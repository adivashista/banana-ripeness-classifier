import os


train_path = r"C:\Banana Classification Project\banana\banana_classification\train"


classes = [
    "raw",
    "ripe",
    "overripe",
    "rotten"
]

total_images = 0

print("=" * 50)
print("BANANA DATASET IMAGE COUNT")
print("=" * 50)

for label in classes:

    folder = os.path.join(train_path, label)

    images = [
        file for file in os.listdir(folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    count = len(images)

    total_images += count

    print(f"{label.capitalize():<12}: {count}")

print("=" * 50)
print(f"Total Images : {total_images}")
print("=" * 50)