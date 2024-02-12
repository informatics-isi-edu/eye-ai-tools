import argparse
import os
import numpy as np
import pandas as pd
from eye_ai import EyeAI
from pathlib import PurePath
from eye_ai_ml.glaucoma.optic_disk_crop import preprocess_and_crop

def filter_angle2(bag_path: str) -> str:
    Dataset_Path = os.path.join(bag_path, 'data/Image.csv')
    Dataset = pd.read_csv(Dataset_Path)
    Dataset_Field_2 = Dataset[Dataset['Image_Angle_Vocab'] == "2SK6"]
    file2_csv_path = f"{bag_path}_Field_2.csv"
    Dataset_Field_2.to_csv(file2_csv_path, index=False)
    return file2_csv_path

def image_cropping_metadata(output_csv_path: str) -> pd.DataFrame:
    cropped_info = pd.read_csv(output_csv_path)[["Image RID", "Worked Image Cropping Function"]]
    cropped_info['Cropped'] = np.where(cropped_info['Worked Image Cropping Function'] == 'Raw Cropped to Eye', 'False', 'True')
    cropped_info = cropped_info[["Image RID", "Cropped"]]
    cropped_info.rename(columns={'Image RID': 'RID'}, inplace=True)
    return cropped_info

def main(hostname: str, catalog_id:str, configuration_rid: str):
    EA = EyeAI(hostname = hostname, catalog_id = catalog_id )
    # Initiate an Execution
    configuration_records = EA.execution_init(configuration_rid=configuration_rid)

    # Data Preprocessing (Selecting image of angle 2 (Field 2) -- Image angle vocab 2SK6;)
    file2_csv_path = filter_angle2(configuration_records['bag_paths'][0])

    # Execute Proecss algorithm (Cropping)
    with EA.start_execution(execution_rid=configuration_records['execution']) as exec:
        preprocess_and_crop(
            configuration_records['bag_paths'][0]+"/data/assets/Image/",
            file2_csv_path,
            './output/output.csv',
            'template.jpg',
            './output/',
            './'+configuration_records['model_paths'][0],
            configuration_records['execution'],
            configuration_records['Annotation_Tag'][0]["RID"],
            configuration_records['Annotation_Tag'][0]["name"],
            False
            )
    # upload cropping bounding box
    EA.upload_assets(f'./output/{configuration_records["execution"]}')

    # ML result analysis
    cropped_info = image_cropping_metadata("output/output.csv")
    EA.update_image_table(cropped_info)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hostname', type = str, required=True)
    parser.add_argument('--catalog_id', type = str, required=True)
    parser.add_argument('--configuration_rid', type = str, required=True)
    args = parser.parse_args()
    main(args.hostname, args.catalog_id, args.configuration_rid)
