from deriva.core import ErmrestCatalog, get_credential
import deriva.core.ermrest_model as ermrest_model
import deriva.core.datapath as datapath
import pandas as pd

from itertools import islice
from typing import List, Sequence, Callable


class EyeAIException(Exception):
    def __init__(self, msg=""):
        self._msg = msg


class EyeAI:
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
    - create_new_vocab(self, schema_name: str, table_name: str, name: str, description: str, synonyms: List[str] = [],
            exist_ok: bool = False) -> str: Creates a new tag in the catalog.
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

    @staticmethod
    def _vocab_columns(table: ermrest_model.Table):
        vocab_columns = {'Name', 'URI', 'Synonyms', 'Description', 'ID'}

        def is_vocab(table: ermrest_model.Table):
            return vocab_columns.issubset({c.name for c in table.columns})

        return [fk.columns[0].name for fk in table.foreign_keys
                if len(fk.columns) == 1 and is_vocab(fk.pk_table)]

    @staticmethod
    def _find_latest_observation(df: pd.DataFrame):
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
        users = self.pb.schemas['public']
        path = users.ERMrest_Client.path
        user = pd.DataFrame(path.entities().fetch())[['ID', 'Full_Name']]
        image_frame = pd.merge(image_frame, user, how="left", left_on='RCB', right_on='ID')

        # Now flatten out Diagnosis_Vocab, Image_quality_Vocab, Image_Side_Vocab
        diagnosis_vocab = pd.DataFrame(self.eye_ai.Diagnosis_Image_Vocab.entities().fetch())[['RID', "Name"]]
        diagnosis_vocab.columns = ['Diagnosis_Vocab', 'Diagnosis']
        image_quality_vocab = pd.DataFrame(self.eye_ai.Image_Quality_Vocab.entities().fetch())[['RID', "Name"]]
        image_quality_vocab.columns = ['Image_Quality_Vocab', 'Image_Quality']
        image_side_vocab = pd.DataFrame(self.eye_ai.Image_Side_Vocab.entities().fetch())[['RID', "Name"]]
        image_side_vocab.columns = ['Image_Side_Vocab', 'Image_Side']

        image_frame = pd.merge(image_frame, diagnosis_vocab, how="left", on='Diagnosis_Vocab')
        image_frame = pd.merge(image_frame, image_quality_vocab, how="left", on='Image_Quality_Vocab')
        image_frame = pd.merge(image_frame, image_side_vocab, how="left", on='Image_Side_Vocab')

        if diagnosis_tag_rid == "C1T4":
            image_frame['Full_Name'] = 'Initial Diagnosis'

        return image_frame[
            ['Subject_RID', 'Diagnosis_RID', 'Full_Name', 'Image', 'Image_Side', 'Diagnosis', 'Cup/Disk_Ratio',
             'Image_Quality']]

    def insert_new_term(self, table_name: str,
                        name: str,
                        description: str,
                        synonyms: List[str] = [],
                        exist_ok: bool = False):
        """
        Creates a new control vocabulary in the control vocabulary table.

        Args:
        - table_name (str): The name of the control vocabulary table.
        - name (str): The name of the new control vocabulary.
        - description (str): The description of the new control vocabulary.
        - synonyms (List[str]): Optional list of synonyms for the new control vocabulary. Defaults to an empty list.
        - exist_ok (bool): Optional flag indicating whether to allow creation if the control vocabulary
        name already exists. Defaults to False.

        Returns:
        - str: The RID (Record ID) of the newly created control vocabulary.

        Raises:
        - Exception: If the control vocabulary name already exists and exist_ok is False.
        """
        try:
            vocab_table = self.pb.schemas['eye-ai'].tables[table_name]
        except KeyError:
            raise EyeAIException(f"The schema or table doesn't exist.")
        entities = vocab_table.path.entities()

        name_list = [e['Name'] for e in entities]

        if not exist_ok and name in name_list:
            idx = name_list.index(name)
            raise EyeAIException(f"{name} existed with RID {entities[idx]['RID']}")
        elif exist_ok and name in name_list:
            raise EyeAIException("The control vocab already existed.")
        else:
            entities = vocab_table.insert([{'Name': name, 'Description': description, 'Synonyms': synonyms}],
                                          defaults={'ID', 'URI'})
            return entities[0]['RID']

    def insert_new_process(self, metadata: str, github_url: str = "", process_tag: str = "", description: str = "",
                           github_checksum: str = ""):
        """

        :param metadata:
        :param github_url:
        :param process_tag:
        :param description:
        :param github_checksum:
        :return:
        """
        entities = self.eye_ai.Process.insert([{'Github_URL': github_url,
                                                'Metadata': metadata,
                                                'Process_Tag': process_tag,
                                                'Description': description,
                                                'Github_Checksum': github_checksum}])
        return entities[0]['RID']

    @staticmethod
    def compute_diagnosis(df: pd.DataFrame,
                          diag_func: Callable,
                          cdr_func: Callable,
                          imageQuality_func: Callable) -> List[dict]:
        """
        Compute a new diagnosis based on provided functions.

        Args:
        - df (DataFrame): Input DataFrame containing relevant columns.
        - Diag_func (Callable): Function to compute Diagnosis.
        - CDR_func (Callable): Function to compute Cup/Disk Ratio.
        - ImageQuality_func (Callable): Function to compute Image Quality.

        Returns:
        - List[Dict[str, Union[str, float]]]: List of dictionaries representing the generated Diagnosis.
        """
        result = df.groupby("Image").agg({"Cup/Disk_Ratio": cdr_func,
                                          "Diagnosis": diag_func,
                                          "Image_Quality": imageQuality_func})
        result.reset_index('Image', inplace=True)
        return result.to_dict(orient='records')

    @staticmethod
    def _batch_insert(table: datapath._TableWrapper, entities: Sequence[dict[str, str]]):
        while chunk := list(islice(iter(entities), 5000)):
            table.insert(chunk)

    def insert_new_diagnosis(self, entities: List[dict[str, dict]],
                             diagTag_RID: str,
                             process_RID: str):
        self._batch_insert(self.eye_ai.Diagnosis,
                           [{'Process': process_RID, 'Diagnosis_Tag': diagTag_RID, **e} for e in entities])
