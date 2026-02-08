from glob import glob
import numpy as np
import nibabel as nib
import os

im_path = "data/images_raw"
labels_path = "data/labels_raw"

image_list = glob(f"{im_path}/*.nii.gz")
label_list = glob(f"{labels_path}/*.nii.gz")

# preprocess images
for img_str in image_list:
    patient_no = img_str.rsplit("_", 2)[1] # extract patient number
    img_vol = nib.load(img_str)
    vol_data = img_vol.get_fdata(dtype=np.float32)
    print(f"Processing patient {patient_no} image...")

    for slice_ in range(vol_data.shape[-1]):
        im = vol_data[:,:,slice_]
        np.save(f"data/images/{patient_no}_{slice_}.npy", im)


# preprocess labels
for lab_str in label_list:
    patient_no = lab_str.rsplit("_", 2)[1] # extract patient number
    img_vol = nib.load(lab_str)
    vol_data = img_vol.get_fdata()
    print(f"Processing patient {patient_no} label...")

    for slice_ in range(vol_data.shape[-1]):
        im = vol_data[:,:,slice_]
        mask = (im == 500).astype(np.float32) # left ventricle as binary mask
        np.save(f"data/labels/{patient_no}_{slice_}.npy", mask)


# make split
print("Splitting data")
all_patients = [x.rsplit("_",2)[1] for x in label_list]
nsamples = len(label_list)
np.random.seed(42)
shuffled = np.random.permutation(all_patients)

NTRAIN = 10
NVAL = 5
NTEST = 5

train_patient_no = shuffled[:NTRAIN]
val_patient_no = shuffled[NTRAIN:NTRAIN + NVAL]
test_patient_no = shuffled[-NTEST:]

# make train.txt, test.txt and val.txt with slice numbers
train_txt = []
val_txt = []
test_txt = []

for patient_no in train_patient_no:
    files = glob(f"data/images/{patient_no}_*.npy")
    for f in files:
        train_txt.append(os.path.basename(f).replace(".npy", ""))

for patient_no in val_patient_no:
    files = glob(f"data/images/{patient_no}_*.npy")
    for f in files:
        val_txt.append(os.path.basename(f).replace(".npy", ""))

for patient_no in test_patient_no:
    files = glob(f"data/images/{patient_no}_*.npy")
    for f in files:
        test_txt.append(os.path.basename(f).replace(".npy", ""))
    
# checks that union is 0
assert len(set(train_txt) & set(val_txt))==0 and len(set(train_txt) & set(test_txt))==0 and len(set(test_txt) & set(val_txt))==0

print("Overlap check completed")

np.savetxt("data/splits/train.txt", train_txt, fmt="%s")
np.savetxt("data/splits/val.txt",   val_txt,   fmt="%s")
np.savetxt("data/splits/test.txt",  test_txt,  fmt="%s")

print("Finished :)")