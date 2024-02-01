import argparse
import requests
import json
import os
import pandas as pd
from eye_ai import EyeAI

def main(hostname: str, catalog_id:str, configuration_rid: str):
    EA = EyeAI(hostname = hostname, catalog_id = catalog_id )
    angle2_rid = "2SK6"
# ============================================================
# Initiate an Execution
    configuration_records, cm = EA.execution_init(configuration_rid=configuration_rid)

# Clone github repo
    for proc in EA.configuration.process:
        repo_url = f'https://github.com/{proc.owner}/{proc.repo}.git'
        os.system(f'git clone {repo_url}')

# Data Preprocessing (Selecting image of angle 2 (Field 2) -- Image angle vocab 2SK6;)
    bag_path1 = configuration_records['bag_paths'][0]
    Dataset_Path = os.path.join(bag_path1, 'data/Image.csv')
    Dataset = pd.read_csv(Dataset_Path)
    Dataset_Field_2 = Dataset[Dataset['Image_Angle_Vocab'] == "2SK6"]
    file2_csv_path = f"{bag_path1}_Field_2.csv"
    Dataset_Field_2.to_csv(file2_csv_path, index=False)

# Execute Proecss algorithm (Cropping)
    os.chdir('/content/eye-ai-ml/Glaucoma Suspect or No Glaaucoma/')
    from Cleaned_Optic_Disc_Cropping_Algorithm_1_0_3_SVG import preprocess_and_crop
    with cm as exec:
        preprocess_and_crop(
            "/content/Dataset_2-5K2C/data/assets/Image/",
            file2_csv_path,
            '../../output/output.csv',
            'template.jpg', 
            '../../output/',
            '/content/'+configuration_records['model_paths'][0],
            configuration_records['process'][0],
            configuration_records['annotation_tag_rid'],
            EA.configuration.annotation_tag.name
            )
        os.chdir('/content')

# ============================================================
# ML result analysis
    import numpy as np
    output_csv_path = "output/output.csv"
    cropped_info = pd.read_csv(output_csv_path)[["Image RID", "Worked Image Cropping Function"]]

    cropped_info['Cropped'] = np.where(cropped_info['Worked Image Cropping Function'] == 'Raw Cropped to Eye', 'False', 'True')
    cropped_info = cropped_info[["Image RID", "Cropped"]]
    cropped_info.rename(columns={'Image RID': 'RID'}, inplace=True)

# Upload results
    # upload cropping bounding box
    EA.upload_assets(f'/content/output/{configuration_records["process"][0]}')
    # upload cropping metadata
    EA.update_image_table(cropped_info)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hostname', type = str, required=True)
    parser.add_argument('--catalog_id', type = str, required=True)
    parser.add_argument('--configuration_rid', type = str, required=True)
    args = parser.parse_args()
    main(args.hostname, args.catalog_id, args.configuration_rid)
