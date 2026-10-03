import argparse
from deepanedet.data import io as dio

def main():
    parser = argparse.ArgumentParser(description="Writes the candidate negative patch centers of each patient directory "
                                                 "(points.csv and points.fcsv), selected in its brain volume ('noskull "
                                                 "volume' of config.json, see deepanedet.scripts.remove_skull) away from its aneurysms.")
    parser.add_argument("patient_dirs", nargs="+", help="e.g. path/to/Data_dir/P* or path/to/Data_dir/sub-*/ses-*")
    parser.add_argument("--truth-file", help="aneurysm points file in each patient directory, e.g. aneurysms.csv "
                                             "(default: the 'pts aneurysm' entry of config.json)")
    args = parser.parse_args()
    for patient_dir in args.patient_dirs:
        dio.extractPointsFromPatient(patient_dir, truth_file=args.truth_file)

if __name__ == '__main__':
    main()
