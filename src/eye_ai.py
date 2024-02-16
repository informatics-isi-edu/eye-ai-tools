from typing import List, Callable
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from PIL import Image
from deriva_ml.deriva_ml_base import DerivaML, DerivaMLException
import re


class EyeAIException(DerivaMLException):
    def __init__(self, msg=""):
        super().__init__(msg=msg)


class EyeAI(DerivaML):
    """
    EyeAI is a class that extends DerivaML and provides additional routines for working with eye-ai
    catalogs using deriva-py.

    Attributes:
    - protocol (str): The protocol used to connect to the catalog (e.g., "https").
    - hostname (str): The hostname of the server where the catalog is located.
    - catalog_number (str): The catalog number or name.
    - credential (object): The credential object used for authentication.
    - catalog (ErmrestCatalog): The ErmrestCatalog object representing the catalog.
    - pb (PathBuilder): The PathBuilder object for constructing URL paths.

    Methods:
    - __init__(self, hostname: str = 'www.eye-ai.org', catalog_number: str = 'eye-ai'): Initializes the EyeAI object.
    - create_new_vocab(self, schema_name: str, table_name: str, name: str, description: str, synonyms: List[str] = [],
            exist_ok: bool = False) -> str: Creates a new controlled vocabulary in the catalog.
    - image_tall(self, dataset_rid: str, diagnosis_tag_rid: str): Retrieves tall-format image data based on provided
      diagnosis tag filters.
    - add_process(self, process_name: str, github_url: str = "", process_tag: str = "", description: str = "",
                    github_checksum: str = "", exists_ok: bool = False) -> str: Adds a new process to the Process table.
    - compute_diagnosis(self, df: pd.DataFrame, diag_func: Callable, cdr_func: Callable,
                          image_quality_func: Callable) -> List[dict]: Computes new diagnosis based on
                                                                       provided functions.
    - insert_new_diagnosis(self, entities: List[dict[str, dict]], diagTag_RID: str, process_rid: str): Batch inserts new
      diagnosis entities into the Diagnoisis table.

    Private Methods:
    - _find_latest_observation(df: pd.DataFrame): Finds the latest observations for each subject in the DataFrame.
    - _batch_insert(table: datapath._TableWrapper, entities: Sequence[dict[str, str]]): Batch inserts
       entities into a table.
    """

    def __init__(self, hostname: str = 'www.eye-ai.org', catalog_id: str = 'eye-ai', data_dir: str= './'):
        """
        Initializes the EyeAI object.

        Args:
        - hostname (str): The hostname of the server where the catalog is located.
        - catalog_number (str): The catalog number or name.
        """

        super().__init__(hostname, catalog_id, 'eye-ai', data_dir)

    @staticmethod
    def _find_latest_observation(df: pd.DataFrame):
        """
        Filter a DataFrame to retain only the rows representing the latest encounters for each subject.

        Args:
        - df (pd.DataFrame): Input DataFrame containing columns 'Subject_RID' and 'Date_of_Encounter'.

        Returns:
        - pd.DataFrame: DataFrame filtered to keep only the rows corresponding to the latest encounters
          for each subject.
        """
        latest_encounters = {}
        for index, row in df.iterrows():
            subject_rid = row['Subject_RID']
            date_of_encounter = row['Date_of_Encounter']
            if subject_rid not in latest_encounters or date_of_encounter > latest_encounters[subject_rid]:
                latest_encounters[subject_rid] = date_of_encounter
        for index, row in df.iterrows():
            if row['Date_of_Encounter'] != latest_encounters[row['Subject_RID']]:
                df.drop(index, inplace=True)
        return df

    def image_tall(self, dataset_rid: str, diagnosis_tag_rid: str):
        """
        Retrieve tall-format image data based on provided dataset and diagnosis tag filters.

        Args:
        - dataset_rid (str): RID of the dataset to filter images.
        - diagnosis_tag_rid (str): RID of the diagnosis tag used for further filtering.

        Returns:
        - pd.DataFrame: DataFrame containing tall-format image data from fist observation of the subject,
          based on the provided filters.
        """
        # Get references to tables to start path.
        subject_dataset = self.schema.Subject_Dataset
        subject = self.schema.Subject
        image = self.schema.Image
        observation = self.schema.Observation
        diagnosis = self.schema.Diagnosis
        path = subject_dataset.path

        results = path.filter(subject_dataset.Dataset == dataset_rid) \
            .link(subject, on=subject_dataset.Subject == subject.RID) \
            .link(observation, on=subject.RID == observation.Subject) \
            .link(image, on=observation.RID == image.Observation) \
            .filter(image.Image_Angle_Vocab == '2SK6') \
            .link(diagnosis, on=image.RID == diagnosis.Image) \
            .filter(diagnosis.Diagnosis_Tag == diagnosis_tag_rid)

        results = results.attributes(
            results.Subject.RID.alias("Subject_RID"),
            results.Observation.Date_of_Encounter,
            results.Diagnosis.RID.alias("Diagnosis_RID"),
            results.Diagnosis.RCB,
            results.Diagnosis.Image,
            results.Image.Image_Side_Vocab,
            results.Image.Filename,
            results.Diagnosis.Diagnosis_Vocab,
            results.Diagnosis.column_definitions['Cup/Disk_Ratio'],
            results.Diagnosis.Image_Quality_Vocab
        )
        image_frame = pd.DataFrame(results.fetch())

        # Select only the first observation which included in the grading app.
        image_frame = self._find_latest_observation(image_frame)

        # Show grader name
        grading_tags = ["2-35G0", "2-35RM", "2-4F74", "2-4F76"]
        diag_tag_vocab = self.list_vocabulary('Diagnosis_Tag')[["RID", "Name"]]
        if diagnosis_tag_rid in grading_tags:
            image_frame = pd.merge(image_frame, self.user_list(), how="left", left_on='RCB', right_on='ID')
        else:
            image_frame = image_frame.assign(
                Full_Name=diag_tag_vocab[diag_tag_vocab['RID'] == diagnosis_tag_rid]["Name"].item())

        # Now flatten out Diagnosis_Vocab, Image_quality_Vocab, Image_Side_Vocab
        diagnosis_vocab = self.list_vocabulary('Diagnosis_Image_Vocab')[["RID", "Name"]].rename(
            columns={"RID": 'Diagnosis_Vocab', "Name": "Diagnosis"})
        image_quality_vocab = self.list_vocabulary('Image_Quality_Vocab')[["RID", "Name"]].rename(
            columns={"RID": 'Image_Quality_Vocab', "Name": "Image_Quality"})
        image_side_vocab = self.list_vocabulary('Image_Side_Vocab')[["RID", "Name"]].rename(
            columns={"RID": 'Image_Side_Vocab', "Name": "Image_Side"})

        image_frame = pd.merge(image_frame, diagnosis_vocab, how="left", on='Diagnosis_Vocab')
        image_frame = pd.merge(image_frame, image_quality_vocab, how="left", on='Image_Quality_Vocab')
        image_frame = pd.merge(image_frame, image_side_vocab, how="left", on='Image_Side_Vocab')

        return image_frame[
            ['Subject_RID', 'Diagnosis_RID', 'Full_Name', 'Image', 'Image_Side', 'Diagnosis', 'Cup/Disk_Ratio',
             'Image_Quality']]

    def compute_diagnosis(self,
                          df: pd.DataFrame,
                          diag_func: Callable,
                          cdr_func: Callable,
                          image_quality_func: Callable) -> List[dict]:
        """
        Compute a new diagnosis based on provided functions.

        Args:
        - df (DataFrame): Input DataFrame containing relevant columns.
        - diag_func (Callable): Function to compute Diagnosis.
        - cdr_func (Callable): Function to compute Cup/Disk Ratio.
        - image_quality_func (Callable): Function to compute Image Quality.

        Returns:
        - List[Dict[str, Union[str, float]]]: List of dictionaries representing the generated Diagnosis.
          The Cup/Disk_Ratio is always round to 4 decimal places.
        """

        result = df.groupby("Image").agg({"Cup/Disk_Ratio": cdr_func,
                                          "Diagnosis": diag_func,
                                          "Image_Quality": image_quality_func})
        result = result.round({'Cup/Disk_Ratio': 4})
        result = result.fillna('NaN')
        result.reset_index('Image', inplace=True)

        image_quality_map = {e["Name"]: e["RID"] for e in self.schema.Image_Quality_Vocab.entities()}
        diagnosis_map = {e["Name"]: e["RID"] for e in self.schema.Diagnosis_Image_Vocab.entities()}
        result.replace({"Image_Quality": image_quality_map,
                        "Diagnosis": diagnosis_map}, inplace=True)
        result.rename({'Image_Quality': 'Image_Quality_Vocab', 'Diagnosis': 'Diagnosis_Vocab'}, axis=1, inplace=True)

        return result.to_dict(orient='records')

    def insert_new_diagnosis(self, entities: List[dict[str, dict]],
                             diagTag_rid: str,
                             process_rid: str):
        """
        Batch insert new diagnosis entities into the Diagnosis table.

        Args:
        - entities (List[dict[str, dict]]): List of diagnosis entities to be inserted.
        - diagTag_RID (str): RID of the diagnosis tag associated with the new entities.
        - process_rid (str): RID of the process associated with the new entities.
        """
        self._batch_insert(self.schema.Diagnosis,
                           [{'Process': process_rid, 'Diagnosis_Tag': diagTag_rid, **e} for e in entities])
    
    def insert_image_annotation(self, upload_result: str, metadata: pd.DataFrame):
        image_annot_entities = []
        for annotation in upload_result.values():
            if annotation["State"] == 0 and annotation["Result"] is not None:
                rid = annotation["Result"].get("RID")
                if rid is not None:
                    filename = annotation["Result"].get("Filename")
                    cur = metadata[metadata['Saved SVG Name'] == filename]
                    image_rid = cur['Image RID'].iloc[0]
                    annot_func = cur['Worked Image Cropping Function'].iloc[0]
                    annot_func_rid = self.lookup_term(table_name="Annotation_Function", term_name=annot_func)
                    annot_type_rid = self.lookup_term(table_name="Annotation_Type", term_name="Optic Nerve")
                    image_annot_entities.append({'Annotation_Function': annot_func_rid,
                                                 'Annotation_Type':annot_type_rid,
                                                 'Image': image_rid,
                                                 'Execution_Assets':rid})
        self._batch_insert(self.schema.Image_Annotation, image_annot_entities)

    def get_bounding_box(self, svg_path: str) -> tuple:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        rect = root.find(".//{http://www.w3.org/2000/svg}rect")
        x_min = int(rect.attrib['x'])
        y_min = int(rect.attrib['y'])
        width = int(rect.attrib['width'])
        height = int(rect.attrib['height'])
        bbox = (x_min, y_min, x_min + width, y_min + height)
        print("Bounding Box:", bbox)
        return bbox

    def get_crop_image(self, bag_path: str) -> tuple:
        svg_root_path = bag_path + '/data/assets/Image_Annotation/'
        image_root_path = bag_path + '/data/assets/Image/'
        cropped_path = Path(bag_path + "/data/assets/Image_cropped")
        cropped_path.mkdir(parents=True, exist_ok=True)
        image_annot_df = pd.read_csv(bag_path+'/data/Image_Annotation.csv')
        image_df = pd.read_csv(bag_path+'/data/Image.csv')

        for index, row in image_annot_df.iterrows():
            image_rid = row['Image']
            svg_path = svg_root_path + row['Filename']
            bbox = self.get_bounding_box(svg_path)
            # crop image save with nice name
            image_file_name = image_df[image_df['RID'] == image_rid]['Filename'].values[0]
            image_file_path = image_root_path + image_file_name
            print(image_file_path)
            image = Image.open(image_file_path)
            cropped_image = image.crop(bbox)
            cropped_image.save(str(cropped_path) + '/Cropped_' + image_file_name)
            image_df["Cropped Filename"] = 'Cropped_' + image_file_name
        output_csv = bag_path + "/data/Cropped_Image.csv"
        image_df.to_csv(output_csv)
        return cropped_path, output_csv 
