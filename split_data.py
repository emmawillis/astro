#!/usr/bin/env python3

import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# ------------------------------------------------------------------------------
# Adjust these paths as needed
# ------------------------------------------------------------------------------
root_dir = "./patches"  # or an absolute path like "/path/to/patches"
raw_dir = os.path.join(root_dir, "raw")
cal_dir = os.path.join(root_dir, "cal")

train_raw_dir = os.path.join(root_dir, "train", "raw")
train_cal_dir = os.path.join(root_dir, "train", "cal")
test_raw_dir  = os.path.join(root_dir, "test", "raw")
test_cal_dir  = os.path.join(root_dir, "test", "cal")

# # ------------------------------------------------------------------------------
# # Create train/test subfolders if they don't already exist
# # ------------------------------------------------------------------------------
# os.makedirs(train_raw_dir, exist_ok=True)
# os.makedirs(train_cal_dir, exist_ok=True)
# os.makedirs(test_raw_dir, exist_ok=True)
# os.makedirs(test_cal_dir, exist_ok=True)

# # ------------------------------------------------------------------------------
# # 1. Gather all imageIDs from the raw directory by removing the last underscore
# # ------------------------------------------------------------------------------
# all_raw_files = os.listdir(raw_dir)
# imageIDs = set()

# for filename in all_raw_files:
#     if filename.lower().endswith((".fit", ".fits")):
#         # Remove only the last underscore-part for coordinates
#         imageID = filename.rsplit("_", 1)[0]
#         imageIDs.add(imageID)

# # Convert to list and shuffle
# imageIDs = list(imageIDs)
# random.shuffle(imageIDs)

# # 80–20 split
# split_index = int(0.8 * len(imageIDs))
# train_imageIDs = set(imageIDs[:split_index])
# test_imageIDs  = set(imageIDs[split_index:])

print(f"Total unique imageIDs: {len(imageIDs)}")
# print(f"Train: {len(train_imageIDs)} IDs, Test: {len(test_imageIDs)} IDs")

# # ------------------------------------------------------------------------------
# # 2. Move the RAW files
# # ------------------------------------------------------------------------------
# for filename in all_raw_files:
#     if filename.lower().endswith((".fit", ".fits")):
#         imageID = filename.rsplit("_", 1)[0]
#         src = os.path.join(raw_dir, filename)
        
#         if imageID in train_imageIDs:
#             dst = os.path.join(train_raw_dir, filename)
#         else:
#             dst = os.path.join(test_raw_dir, filename)
        
#         shutil.move(src, dst)

# # ------------------------------------------------------------------------------
# # 3. Move the CAL files
# # ------------------------------------------------------------------------------
# all_cal_files = os.listdir(cal_dir)
# for filename in all_cal_files:
#     if filename.lower().endswith((".fit", ".fits")):
#         imageID = filename.rsplit("_", 1)[0]
#         src = os.path.join(cal_dir, filename)
        
#         if imageID in train_imageIDs:
#             dst = os.path.join(train_cal_dir, filename)
#         else:
#             dst = os.path.join(test_cal_dir, filename)
        
#         shutil.move(src, dst)

# print("Dataset split complete.")

# ------------------------------------------------------------------------------
# 4. Count files in each subfolder
# ------------------------------------------------------------------------------
def count_fits_in_folder(folder_path):
    return sum(
        1 for f in os.listdir(folder_path)
        if f.lower().endswith((".fit", ".fits"))
    )

train_raw_count = count_fits_in_folder(train_raw_dir)
train_cal_count = count_fits_in_folder(train_cal_dir)
test_raw_count  = count_fits_in_folder(test_raw_dir)
test_cal_count  = count_fits_in_folder(test_cal_dir)

print("\nFile counts after splitting:")
print(f"  train/raw: {train_raw_count}")
print(f"  train/cal: {train_cal_count}")
print(f"  test/raw:  {test_raw_count}")
print(f"  test/cal:  {test_cal_count}")

# ------------------------------------------------------------------------------
# 5. Verify that each file in train/cal has a counterpart in train/raw
#    and similarly for test/cal vs test/raw
# ------------------------------------------------------------------------------
train_cal_files = {
    f for f in os.listdir(train_cal_dir)
    if f.lower().endswith((".fit", ".fits"))
}
train_raw_files = {
    f for f in os.listdir(train_raw_dir)
    if f.lower().endswith((".fit", ".fits"))
}
test_cal_files = {
    f for f in os.listdir(test_cal_dir)
    if f.lower().endswith((".fit", ".fits"))
}
test_raw_files = {
    f for f in os.listdir(test_raw_dir)
    if f.lower().endswith((".fit", ".fits"))
}

# Check train: every cal file also in raw
missing_in_train_raw = train_cal_files - train_raw_files
# Check test: every cal file also in raw
missing_in_test_raw  = test_cal_files - test_raw_files

if missing_in_train_raw:
    print("\nWARNING: The following files are in train/cal but NOT in train/raw:")
    for f in sorted(missing_in_train_raw):
        print(f"  {f}")
else:
    print("\nAll train/cal files have a matching file in train/raw.")

if missing_in_test_raw:
    print("\nWARNING: The following files are in test/cal but NOT in test/raw:")
    for f in sorted(missing_in_test_raw):
        print(f"  {f}")
else:
    print("\nAll test/cal files have a matching file in test/raw.")

print("\nDone.")
