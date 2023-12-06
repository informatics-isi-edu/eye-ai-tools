from deriva.core import ErmrestCatalog, get_credential
import deriva.core.ermrest_model as ermrest_model
import deriva.core.datapath as datapath
import pandas as pd

from itertools import islice
from typing import List, Sequence, Callable


class DerivaMLException(Exception):
    def __init__(self, msg=""):
        super().__init__(msg)
        self._msg = msg


class EyeAIException(DerivaMLException):
    def __init__(self, msg=""):
        super().__init__(msg=msg)


class DerivaML:
    def __init__(self, hostname: str, catalog_number: str, schema_name):
        self.credential = get_credential(hostname)
        self.catalog = ErmrestCatalog('https', hostname, catalog_number, self.credential)
        self.model = self.catalog.getCatalogModel()
        self.pb = self.catalog.getPathBuilder()
        self.host_name = hostname
        self.schema_name = schema_name
        self.schema = self.pb.schemas[schema_name]

    def is_vocabulary(self, table_name: str) -> bool:
        """

        :param table_name:
        :return:
        """
        vocab_columns = {'Name', 'URI', 'Synonyms', 'Description', 'ID'}
        try:
            table = self.model.schemas[self.schema_name].tables[table_name]
        except KeyError:
            raise DerivaMLException(f"The vocabulary table {table_name} doesn't exist.")
        return vocab_columns.issubset({c.name for c in table.columns})

    def _vocab_columns(self, table: ermrest_model.Table):
        """
        Return the list of columns in the table that are vocabulary terms.
        :param table:
        :return:
        """
        return [fk.columns[0].name for fk in table.foreign_keys
                if len(fk.columns) == 1 and self.is_vocabulary(fk.pk_table)]

    def add_term(self, table_name: str,
                 name: str,
                 description: str,
                 synonyms: List[str] = None,
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
        synonyms = synonyms or []

        try:
            if not self.is_vocabulary(table_name):
                raise EyeAIException(f"The table {table_name} is not a controlled vocabulary")
        except KeyError:
            raise EyeAIException(f"The schema or vocabulary table {table_name} doesn't exist.")

        try:
            entities = self.schema.tables[table_name].entities()
            name_list = [e['Name'] for e in entities]
            term_rid = entities[name_list.index(name)]['RID']
        except ValueError:
            # Name is not in list of current terms
            term_rid = self.schema.tables[table_name].insert(
                [{'Name': name, 'Description': description, 'Synonyms': synonyms}],
                defaults={'ID', 'URI'})[0]['RID']
        else:
            # Name is list of current terms.
            if not exist_ok:
                raise EyeAIException(f"{name} existed with RID {entities[name_list.index(name)]['RID']}")
        return term_rid

    def lookup_term(self, table_name: str, term_name: str) -> str:
        """
        Given a term name, return the RID of the associated term (or synanum).
        :param table_name:
        :param term_name:
        :return:
        """
        try:
            vocab_table = self.is_vocabulary(table_name)
        except KeyError:
            raise EyeAIException(f"The schema or vocabulary table {table_name} doesn't exist.")

        if not vocab_table:
            raise EyeAIException(f"The table {table_name} is not a controlled vocabulary")

        for term in self.schema.tables[table_name].entities():
            if term_name == term['Name'] or (term['Synonyms'] and term_name in term['Synonyms']):
                return term['RID']

        raise EyeAIException(f"Term {term_name} is not in vocabuary {table_name}")

    def list_vocabularies(self):
        """
        Return a list of all of the controlled vocabularies in the schema.
        :return:
        """
        return [t for t in self.schema.tables if self.is_vocabulary(t)]

    def list_vocabulary(self, table_name: str) -> pd.DataFrame:
        """
        Return the list of terms that are in a vocabulary table.
        :param table_name:
        :return:
        """
        try:
            vocab_table = self.is_vocabulary(table_name)
        except KeyError:
            raise EyeAIException(f"The schema or vocabulary table {table_name} doesn't exist.")

        if not vocab_table:
            raise EyeAIException(f"The table {table_name} is not a controlled vocabulary")

        return pd.DataFrame(self.schema.tables[table_name].entities().fetch())

    def user_list(self) -> pd.DataFrame:
        users = self.pb.schemas['public']
        path = users.ERMrest_Client.path
        return pd.DataFrame(path.entities().fetch())[['ID', 'Full_Name']]


class EyeAI(DerivaML):
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

        super().__init__(hostname, catalog_number, 'eye-ai')

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
        image_frame = pd.merge(image_frame, self.user_list(), how="left", left_on='RCB', right_on='ID')

        # Now flatten out Diagnosis_Vocab, Image_quality_Vocab, Image_Side_Vocab
        diagnosis_vocab = self.list_vocabulary('Diagnosis_Image_Vocab').columns = ['Diagnosis_Vocab', 'Diagnosis']
        diagnosis_vocab.columns = ['Diagnosis_Vocab', 'Diagnosis']
        image_quality_vocab = self.list_vocabulary('Image_Quality_Vocab')
        image_quality_vocab.columns = ['Image_Quality_Vocab', 'Image_Quality']
        image_side_vocab = self.list_vocabulary('Image_Side_Vocab')
        image_side_vocab.columns = ['Image_Side_Vocab', 'Image_Side']

        image_frame = pd.merge(image_frame, diagnosis_vocab, how="left", on='Diagnosis_Vocab')
        image_frame = pd.merge(image_frame, image_quality_vocab, how="left", on='Image_Quality_Vocab')
        image_frame = pd.merge(image_frame, image_side_vocab, how="left", on='Image_Side_Vocab')

        if diagnosis_tag_rid == "C1T4":
            image_frame['Full_Name'] = 'Initial Diagnosis'

        return image_frame[
            ['Subject_RID', 'Diagnosis_RID', 'Full_Name', 'Image', 'Image_Side', 'Diagnosis', 'Cup/Disk_Ratio',
             'Image_Quality']]

    def add_process(self, process_name: str, github_url: str = "", process_tag: str = "", description: str = "",
                    github_checksum: str = "",
                    exists_ok: bool = False):
        """

        :param process_name:
        :param github_url:
        :param process_tag:
        :param description:
        :param github_checksum:
        :param exists_ok:
        :return:
        """
        # TODO Should check to make sure process doesn't already exist?

        entities = self.schema.Process.insert([{'Github_URL': github_url,
                                                'Name': process_name,
                                                'Process_Tag': process_tag,
                                                'Description': description,
                                                'Github_Checksum': github_checksum}])
        return entities[0]['RID']

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

    @staticmethod
    def _batch_insert(table: datapath._TableWrapper, entities: Sequence[dict[str, str]]):
        it = iter(entities)
        while chunk := list(islice(it, 2000)):
            table.insert(chunk)

    def insert_new_diagnosis(self, entities: List[dict[str, dict]],
                             diagTag_RID: str,
                             process_rid: str):
        EyeAI._batch_insert(self.schema.Diagnosis,
                            [{'Process': process_rid, 'Diagnosis_Tag': diagTag_RID, **e} for e in entities])
