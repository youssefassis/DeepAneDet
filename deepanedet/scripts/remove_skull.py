import argparse, os, json
import numpy as np
import nibabel as ni
from deepanedet.volume import selection as vs

def remove_skull(patient_dir):
    config_file = os.path.join(patient_dir, 'config.json')
    with open(config_file, 'r') as f:
        d = json.load(f)
    ni_vol = ni.load(os.path.join(patient_dir, d['init volume']))
    vol = np.asarray(ni_vol.dataobj).astype(np.float32)

    mask = vs.removeSkullMask(vol)
    vol = mask * vol
    vol /= np.max(vol)

    outname = 'noskull.nii.gz'
    ni.Nifti1Image(vol, ni_vol.affine).to_filename(os.path.join(patient_dir, outname))
    d['noskull volume'] = outname
    with open(config_file, 'w') as f:
        json.dump(d, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Writes the brain volume (without skull) of each patient directory as "
                                                 "noskull.nii.gz, recorded in its config.json under 'noskull volume'.")
    parser.add_argument("patient_dirs", nargs="+", help="e.g. path/to/Data_dir/P* or path/to/Data_dir/sub-*/ses-*")
    for patient_dir in parser.parse_args().patient_dirs:
        print(f'Removing the skull of {patient_dir}')
        remove_skull(patient_dir)

if __name__ == '__main__':
    main()
