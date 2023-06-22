# example use in command line
# python ADNI_MRI_parallel_processing.py \
# -p '/absolute/path/to/ADNI' \
# -t '/path/to/Data/tester_generator' \
# -r '/path/to/fsl/bin/flirt' \
# -m '/path/to/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz' \
# -n \
# -c 6

import os
import ants
import subprocess
import time
import concurrent.futures
import glob
import argparse
import logging

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def parse_args():
    args_dict = {
        'p': ('Path to ADNI class directory - AD or NC', None),
        't': ('New folder to store processed data', None),
        'r': ('Absolute path to where flirt is registered', None),
        'm': ('Absolute path of MNI52_T1_1mm_brain.nii.gz', None),
        'n': ('Remove the transition files after processing is complete for one file', 'store_true'),
        'c': ('Number of cores to process images, e.g., 6 cores process 6 images at one time', 8)
    }
    parser = argparse.ArgumentParser()
    for arg, (desc, default) in args_dict.items():
        kwargs = {'help': desc}
        if default is not None:
            kwargs['default'] = default
        if default == 'store_true':
            kwargs['action'] = 'store_true'
        else:
            kwargs['type'] = type(default) if default else str
        parser.add_argument(f'-{arg}', f'--{arg}', **kwargs)
    return parser.parse_args()

def preprocess(source, args):
    patient_id = source.split('/')[6]
    only_file_name = source.split('/')[-1].split('.')[0]

    create_dir_if_not_exists(args.target)
    
    target_patient_folder = os.path.join(args.target, patient_id) 
    create_dir_if_not_exists(target_patient_folder)
        
    target_patient_file = os.path.join(target_patient_folder, only_file_name)
    steps = ['_cropped.nii.gz', '_stripped.nii.gz', '_reoriented.nii.gz', '.nii.gz', '_n4bias.nii.gz']
    crop, skullstrip, reorient, final, n4bias = [target_patient_file + step for step in steps]

    if not os.path.exists(target_patient_file):
        start = time.time()
        logging.info(f' Processing {source}')
        subprocess.run(['robustfov', '-i', source, '-r', crop], check=True)
        
        new_img = ants.image_read(crop)
        bias = ants.n4_bias_field_correction(new_img)
        ants.image_write(bias, n4bias)
        
        subprocess.run(['bet', n4bias, skullstrip,  '-f', '0.5', '-g', '0'], check=True)
        subprocess.run(['fslreorient2std', skullstrip, reorient], check=True)
        
        subprocess.run([args.register,
                        '-in', skullstrip, 
                        '-ref', args.mni, 
                        '-out', final, 
                        '-bins', '256', 
                        '-cost', 'corratio', 
                        '-dof', '12',
                        '-interp', 'trilinear'], check=True)  

        images = [crop, reorient, skullstrip, n4bias]

        if args.nofiles:
            [os.remove(image) for image in images]

        end = time.time()
        logging.info(f'Process finished for {final}, took {end - start} seconds!')     

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    images_list = glob.glob(args.path + '/*/*/*/*/*')

    with concurrent.futures.ProcessPoolExecutor(args.cores) as executor:
        for img, result in zip(images_list, executor.map(preprocess, images_list, [args]*len(images_list))):
            logging.info(f'{img} is processed...')

if __name__ == "__main__":
    main()
