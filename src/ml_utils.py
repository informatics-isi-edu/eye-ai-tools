import deriva.core.utils.globus_auth_utils
from deriva.core import ErmrestCatalog, HatracStore, AttrDict, get_credential
import deriva.core.ermrest_model as ermrest_model
import deriva.core.datapath as datapath
import pandas as pd

from typing import List
import logging
from deriva.core import init_logging

class EyeAI():
    """
    CatalogHelper is a class that provides helper routines for manipulating a catalog using deriva-py.

    Attributes:
    - protocol (str): The protocol used to connect to the catalog (e.g., "https").
    - hostname (str): The hostname of the server where the catalog is located.
    - catalog_number (str): The catalog number or name.
    - credential (object): The credential object used for authentication.
    - catalog (ErmrestCatalog): The ErmrestCatalog object representing the catalog.
    - pb (PathBuilder): The PathBuilder object for constructing URL paths.

    Methods:
    - __init__(self, protocol: str, hostname: str, catalog_number: str): Initializes the HelperRoutines object.
    - create_new_vocab(self, schema_name: str, table_name: str, name: str, description: str, synonyms: List[str] = [], exist_ok: bool = False) -> str: Creates a new tag in the catalog.
    """
    def __init__(self, hostname: str = 'www.eye-ai.org', catalog_number: str = 'eye-ai'):
        """
        Initializes the HelperRoutines object.

        Args:
        - hostname (str): The hostname of the server where the catalog is located.
        - catalog_number (str): The catalog number or name.
        """

        self.credential = get_credential(hostname)
        self.catalog = ErmrestCatalog('https', hostname, catalog_number, self.credential)
        self.model = self.catalog.getCatalogModel()
        self.pb = self.catalog.getPathBuilder()
        self.eye_ai = self.pb.schemas['eye-ai']

    def _vocab_columns(self, table: ermrest_model.Table):
        vocab_columns = {'Name', 'URI', 'Synonyms', 'Description', 'ID'}
        def is_vocab(table: ermrest_model.Table):
            return vocab_columns.issubset({c.name for c in table.columns})

        return [fk.columns[0].name for fk in table.foreign_keys
                if len(fk.columns) == 1 and is_vocab(fk.pk_table)]
    @property
    def image(self) -> datapath._TableWrapper:
        return self.eye_ai.Image
    

    def _find_latest_observation(self, df: pd.DataFrame):
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
        # Get references to tables to start path.
        subject_dataset = self.eye_ai.Subject_Dataset
        subject = self.eye_ai.Subject
        image = self.eye_ai.Image
        observation = self.eye_ai.Observation
        diagnosis = self.eye_ai.Diagnosis
        path = subject_dataset.path

        results = path.filter(subject_dataset.Dataset == dataset_rid)\
        .link(subject, on=subject_dataset.Subject==subject.RID)\
        .link(observation, on=subject.RID==observation.Subject)\
        .link(image, on=observation.RID==image.Observation)\
        .filter(image.Image_Angle_Vocab == '2SK6')\
        .link(diagnosis, on=image.RID==diagnosis.Image)\
        .filter(diagnosis.Diagnosis_Tag==diagnosis_tag_rid)

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
        users = self.pb.schemas['public']
        EC = users.ERMrest_Client
        path = EC.path
        User = pd.DataFrame(path.entities().fetch())[['ID', 'Full_Name']]
        image_frame = pd.merge(image_frame, User, how="left", left_on='RCB', right_on='ID' )

        # Now flatten out Diagnosis_Vocab, Image_quality_Vocab, Image_Side_Vocab
        diagnosis_vocab = pd.DataFrame(self.eye_ai.Diagnosis_Image_Vocab.entities().fetch())[['RID', "Name"]]
        diagnosis_vocab.columns=['Diagnosis_Vocab', 'Diagnosis']
        image_quality_vocab = pd.DataFrame(self.eye_ai.Image_Quality_Vocab.entities().fetch())[['RID', "Name"]]
        image_quality_vocab.columns=['Image_Quality_Vocab', 'Image_Quality']
        image_side_vocab = pd.DataFrame(self.eye_ai.Image_Side_Vocab.entities().fetch())[['RID', "Name"]]
        image_side_vocab.columns=['Image_Side_Vocab', 'Image_Side']

        image_frame = pd.merge(image_frame, diagnosis_vocab, how="left", on='Diagnosis_Vocab')
        image_frame = pd.merge(image_frame, image_quality_vocab, how="left", on='Image_Quality_Vocab')
        image_frame = pd.merge(image_frame, image_side_vocab, how="left", on='Image_Side_Vocab')

        if diagnosis_tag_rid == "C1T4":
            image_frame['Full_Name'] = 'Initial Diagnosis'

        return image_frame[['Subject_RID', 'Diagnosis_RID', 'Full_Name', 'Image', 'Image_Side', 'Diagnosis', 'Cup/Disk_Ratio', 'Image_Quality']]
    

    def create_new_vocab(self, table_name: str, name: str, description: str, synonyms: List[str] = [], exist_ok: bool = False):
        """
        Creates a new control vocabulary in the control vocabulary table.

        Args:
        - schema_name (str): The name of the schema where the control vocabulary table is located.
        - table_name (str): The name of the control vocabulary table.
        - name (str): The name of the new control vocabulary.
        - description (str): The description of the new control vocabulary.
        - synonyms (List[str]): Optional list of synonyms for the new control vocabulary. Defaults to an empty list.
        - exist_ok (bool): Optional flag indicating whether to allow creation if the control vocabulary name already exists. Defaults to False.

        Returns:
        - str: The RID (Record ID) of the newly created control vocabulary.

        Raises:
        - Exception: If the control vocabulary name already exists and exist_ok is False.
        """

        init_logging()
        # Load all the vocab names
        try:
            vocab_table = self.pb.schemas['eye-ai'].tables[table_name]
        except:
            print("The schema or table doesn't exist.")
            return
        entities = vocab_table.path.entities()

        Name_list = [e['Name'] for e in entities]
            
        # Check if vocab name existed
        if exist_ok == False:
            if name in Name_list:
                idx = Name_list.index(name)
                logging.info(name + "existed")
                return entities[idx]['RID']
            else:
                logging.info("New vocab created")
                new_entity = {'Name': name, 'Description':description, 'Synonyms':synonyms}
                entities = vocab_table.insert([new_entity], defaults={'ID', 'URI'})
                return entities[0]['RID']
        else:
            if name in Name_list:
                logging.info("The control vocab already existed.")
            else:
                logging.info("New vocab created")
                new_entity = {'Name': name, 'Description':description, 'Synonyms':synonyms}
                entities = vocab_table.insert([new_entity], defaults={'ID', 'URI'})
                return entities[0]['RID']

    def insert_new_process(self, Metadata:str, Github_URL:str="", Process_Tag:str="", Description:str="", Github_Checksum:str=""):
        process_records = [{'Github_URL':Github_URL,
                            'Metadata':Metadata,
                            'Process_Tag': Process_Tag,
                            'Description': Description,
                            'Github_Checksum':Github_Checksum}]
        response = self.catalog.post("/entity/eye-ai:Process", json=process_records)
        response.raise_for_status()
        return response.json()[0]['RID']
    
    def generate_Diagnosis(self, df, Diag_rule, CDR_rule, ImageQuality_rule):
        result = df.groupby("Image").agg({"Cup/Disk_Ratio":CDR_rule,
                                        "Diagnosis": Diag_rule,
                                        "Image_Quality": ImageQuality_rule})
        result.reset_index('Image', inplace=True)
        return result.to_dict(orient='records')
    
    def _batchInsert(self, table, entities: List[str]):
        n = len(entities)
        batch_num = min(n//5, 5000)
        for i in range(n//batch_num):
            table.insert(entities[i*batch_num: (i+1)*batch_num])
            print(i*batch_num, (i+1)*batch_num)
        if (i+1)*batch_num < n:
            table.insert(entities[(i+1)*batch_num: n])
            print((i+1)*batch_num, n)

    def insert_new_diag(self, entities, diagTag_RID, Process_RID):
        n = len(entities)
        for e in entities:
            e["Process"] = Process_RID
            e["Diagnosis_Tag"] = diagTag_RID
        self._batchInsert(self.eye_ai.Diagnosis, entities)
        return entities