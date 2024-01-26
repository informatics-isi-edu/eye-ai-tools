from deriva.core import ErmrestCatalog, get_credential
import deriva.core.ermrest_model as ermrest_model
import deriva.core.datapath as datapath
import pandas as pd
import requests
import os
import json
from itertools import islice
from typing import List, Sequence, Callable
from datetime import datetime


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
        Check if a given table is a controlled vocabulary table.

        Args:
        - table_name (str): The name of the table.

        Returns:
        - bool: True if the table is a controlled vocabulary, False otherwise.

        """
        vocab_columns = {'Name', 'URI', 'Synonyms', 'Description', 'ID'}
        try:
            table = self.model.schemas[self.schema_name].tables[table_name]
        except KeyError:
            raise DerivaMLException(f"The vocabulary table {table_name} doesn't exist.")
        return vocab_columns.issubset({c.name for c in table.columns})

    def _vocab_columns(self, table: ermrest_model.Table):
        """
        Return the list of columns in the table that are control vocabulary terms.

        Args:
        - table (ermrest_model.Table): The table.

        Returns:
        - List[str]: List of column names that are control vocabulary terms.
        """
        return [fk.columns[0].name for fk in table.foreign_keys
                if len(fk.columns) == 1 and self.is_vocabulary(fk.pk_table)]

    def add_term(self, table_name: str,
                 name: str,
                 description: str,
                 synonyms: List[str] = None,
                 exist_ok: bool = False):
        """
        Creates a new control vocabulary term in the control vocabulary table.

        Args:
        - table_name (str): The name of the control vocabulary table.
        - name (str): The name of the new control vocabulary.
        - description (str): The description of the new control vocabulary.
        - synonyms (List[str]): Optional list of synonyms for the new control vocabulary. Defaults to an empty list.
        - exist_ok (bool): Optional flag indicating whether to allow creation if the control vocabulary name already exists. Defaults to False.

        Returns:
        - str: The RID of the newly created control vocabulary.

        Raises:
        - EyeAIException: If the control vocabulary name already exists and exist_ok is False.
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
        Given a term name, return the RID of the associated term (or synonym).

        Args:
        - table_name (str): The name of the controlled vocabulary table.
        - term_name (str): The name of the term to look up.

        Returns:
        - str: The RID of the associated term or synonym.

        Raises:
        - EyeAIException: If the schema or vocabulary table doesn't exist, or if the term is not found in the vocabulary.

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
        Return a list of all of the controlled vocabulary tables in the schema.

        Returns:
         - List[str]: A list of table names representing controlled vocabulary tables in the schema.

        """
        return [t for t in self.schema.tables if self.is_vocabulary(t)]

    def list_vocabulary(self, table_name: str) -> pd.DataFrame:
        """
        Return the dataframe of terms that are in a vocabulary table.

        Args:
        - table_name (str): The name of the controlled vocabulary table.

        Returns:
        - pd.DataFrame: A DataFrame containing the terms in the specified controlled vocabulary table.

        Raises:
        - EyeAIException: If the schema or vocabulary table doesn't exist, or if the table is not a controlled vocabulary.
        """
        try:
            vocab_table = self.is_vocabulary(table_name)
        except KeyError:
            raise EyeAIException(f"The schema or vocabulary table {table_name} doesn't exist.")

        if not vocab_table:
            raise EyeAIException(f"The table {table_name} is not a controlled vocabulary")

        return pd.DataFrame(self.schema.tables[table_name].entities().fetch())

    def user_list(self) -> pd.DataFrame:
        """
        Return a DataFrame containing user information.
        """
        users = self.pb.schemas['public']
        path = users.ERMrest_Client.path
        return pd.DataFrame(path.entities().fetch())[['ID', 'Full_Name']]


class EyeAI(DerivaML):
    """
    EyeAI is a class that extends DerivaML and provides additional routines for working with eye-ai catalogs using deriva-py.

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
    - image_tall(self, dataset_rid: str, diagnosis_tag_rid: str): Retrieves tall-format image data based on provided diagnosis tag filters.
    - add_process(self, process_name: str, github_url: str = "", process_tag: str = "", description: str = "",
                    github_checksum: str = "", exists_ok: bool = False) -> str: Adds a new process to the Process table.
    - compute_diagnosis(self, df: pd.DataFrame, diag_func: Callable, cdr_func: Callable,
                          image_quality_func: Callable) -> List[dict]: Computes new diagnosis based on provided functions.
    - insert_new_diagnosis(self, entities: List[dict[str, dict]], diagTag_RID: str, process_rid: str): Batch inserts new diagnosis entities into the Diagnoisis table.

    Private Methods:
    - _find_latest_observation(df: pd.DataFrame): Finds the latest observations for each subject in the DataFrame.
    - _batch_insert(table: datapath._TableWrapper, entities: Sequence[dict[str, str]]): Batch inserts entities into a table.
    """

    def __init__(self, hostname: str = 'www.eye-ai.org', catalog_number: str = 'eye-ai'):
        """
        Initializes the EyeAI object.

        Args:
        - hostname (str): The hostname of the server where the catalog is located.
        - catalog_number (str): The catalog number or name.
        """

        super().__init__(hostname, catalog_number, 'eye-ai')
        self.start_time = None 

    @staticmethod
    def _find_latest_observation(df: pd.DataFrame):
        """
        Filter a DataFrame to retain only the rows representing the latest encounters for each subject.

        Args:
        - df (pd.DataFrame): Input DataFrame containing columns 'Subject_RID' and 'Date_of_Encounter'.

        Returns:
        - pd.DataFrame: DataFrame filtered to keep only the rows corresponding to the latest encounters for each subject.
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
        - pd.DataFrame: DataFrame containing tall-format image data from fist observation of the subject, based on the provided filters.
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
        Grading_tags = ["2-35G0", "2-35RM", "2-4F74", "2-4F76"]
        diag_tag_vocab = self.list_vocabulary('Diagnosis_Tag')[["RID", "Name"]]
        if diagnosis_tag_rid in Grading_tags:
            image_frame = pd.merge(image_frame, self.user_list(), how="left", left_on='RCB', right_on='ID')
        else:
            image_frame = image_frame.assign(Full_Name=diag_tag_vocab[diag_tag_vocab['RID'] == diagnosis_tag_rid]["Name"].item())
        # image_frame = pd.merge(image_frame, self.user_list(), how="left", left_on='RCB', right_on='ID')

        # Now flatten out Diagnosis_Vocab, Image_quality_Vocab, Image_Side_Vocab
        diagnosis_vocab = self.list_vocabulary('Diagnosis_Image_Vocab')[["RID", "Name"]].rename(columns={"RID":'Diagnosis_Vocab', "Name":"Diagnosis"})
        image_quality_vocab = self.list_vocabulary('Image_Quality_Vocab')[["RID", "Name"]].rename(columns={"RID":'Image_Quality_Vocab', "Name":"Image_Quality"})
        image_side_vocab = self.list_vocabulary('Image_Side_Vocab')[["RID", "Name"]].rename(columns={"RID":'Image_Side_Vocab', "Name":"Image_Side"})

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
        - List[Dict[str, Union[str, float]]]: List of dictionaries representing the generated Diagnosis. The Cup/Disk_Ratio is always round to 4 decimal places.
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

    @staticmethod
    def _batch_update(table: datapath._TableWrapper, entities: Sequence[dict[str, str]], update_cols: List[datapath._ColumnWrapper]):
        it = iter(entities)
        while chunk := list(islice(it, 2000)):
            table.update(chunk, [table.RID], update_cols)

    @staticmethod
    def _github_metadata(owner: str, repo: str, file_path: str) -> dict[str:str]:
        response = requests.get(f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}")
        Github_metadata = response.json()
        return {"Github_Checksum": Github_metadata['sha'], "Github_URL": Github_metadata["html_url"]}

    def update_image_table(self, df: pd.DataFrame):
        """
        Batch update Cropped info (True/ False) into the Image table.

        Args:
        - df (pd.DataFrame): A dataframe of new Cropped info to be inserted. It contains two columns: RID, and Cropped('True'/'False')
        """

        Cropped_map = {e["Name"]: e["RID"] for e in self.schema.Cropped.entities()}
        df.replace({"Cropped": Cropped_map}, inplace=True)
        EyeAI._batch_update(self.schema.Image, df.to_dict(orient='records'), [self.schema.Image.Cropped])

    def insert_new_diagnosis(self, entities: List[dict[str, dict]],
                             diagTag_RID: str,
                             process_rid: str):
        """
        Batch insert new diagnosis entities into the Diagnosis table.

        Args:
        - entities (List[dict[str, dict]]): List of diagnosis entities to be inserted.
        - diagTag_RID (str): RID of the diagnosis tag associated with the new entities.
        - process_rid (str): RID of the process associated with the new entities.
        """
        self._batch_insert(self.schema.Diagnosis,
                            [{'Process': process_rid, 'Diagnosis_Tag': diagTag_RID, **e} for e in entities])
    
    def add_process(self, process_name: str, process_tag: str = "", description: str = "",
                    github_owner: str = "", github_repo: str = "", github_file_path: str = "",
                    exist_ok: bool = False) -> str:
        """
        Add a new process to the catalog.

        Args:
        - process_name (str): Name of the new process.
        - github_url (str, optional): GitHub URL associated with the process.
        - process_tag (str, optional): Tag for the process.
        - description (str, optional): Description of the process.
        - github_checksum (str, optional): Checksum of the GitHub repository.
        - exists_ok (bool, optional):  Optional flag indicating whether to allow creation if the control vocabulary name already exists. Defaults to False.

        Returns:
        - str: RID (Record ID) of the newly created process.

        Raises:
        - Exception: If the process already exists and exists_ok is False.
        """

        try:
            entities = self.schema.Process.entities()
            name_list = [e['Name'] for e in entities]
            term_rid = entities[name_list.index(process_name)]['RID']
        except ValueError:
            # Name is not in list of current terms
            github_metadata = self._github_metadata(github_owner, github_repo, github_file_path)
            term_rid = self.schema.Process.insert([{'Github_URL': github_metadata["Github_URL"],
                                                'Name': process_name,
                                                'Process_Tag': process_tag,
                                                'Description': description,
                                                'Github_Checksum': github_metadata["Github_Checksum"]}])[0]['RID']
        else:
            # Name is list of current terms.
            if not exist_ok:
                raise EyeAIException(f"{process_name} existed with RID {entities[name_list.index(process_name)]['RID']}")
        return term_rid

    def add_workflow(self, workflow_name: str, description: str = "",
                    github_owner: str = "", github_repo: str = "", github_file_path: str = "",
                    Process_list: List = [],
                    exist_ok: bool = False) -> str:
        try:
            entities = self.schema.Workflow.entities()
            name_list = [e['Name'] for e in entities]
            term_rid = entities[name_list.index(workflow_name)]['RID']
        except ValueError:
            # Name is not in list of current terms
            github_metadata = self._github_metadata(github_owner, github_repo, github_file_path)
            term_rid = self.schema.Workflow.insert([{'Github_URL': github_metadata["Github_URL"],
                                                'Name': workflow_name,
                                                'Description': description,
                                                'Github_Checksum': github_metadata["Github_Checksum"]}])[0]['RID']
            entities = [{"Process": p, "Workflow": term_rid} for p in Process_list]
            self._batch_insert(self.schema.Workflow_Process, entities)
        else:
            # Name is list of current terms.
            if not exist_ok:
                raise EyeAIException(f"{workflow_name} existed with RID {entities[name_list.index(workflow_name)]['RID']}")
        return term_rid

    def add_execution(self, Execution_name: str, workflow_RID: str, description: str = "", exist_ok: bool = False) -> str:
        try:
            entities = self.schema.Execution.entities()
            name_list = [e['Name'] for e in entities]
            term_rid = entities[name_list.index(Execution_name)]['RID']
        except ValueError:
            # Name is not in list of current terms
            term_rid = self.schema.Execution.insert([{
                'Name': Execution_name,
                'Description': description,
                'Workflow': workflow_RID}])[0]['RID']
        else:
            # Name is list of current terms.
            if not exist_ok:
                raise EyeAIException(f"{Execution_name} existed with RID {entities[name_list.index(Execution_name)]['RID']}")
        return term_rid

    def download_execution_asset(self, Asset_RID: str, Execution_RID, dest_dir: str=""):
            Asset_metadata = self.schema.Execution_Asset.filter(self.schema.Execution_Asset.RID == Asset_RID).entities()[0]
            Asset_URL = Asset_metadata['URL']
            file_name = Asset_metadata['Filename']
            os.system(f'deriva-hatrac-cli --host {self.host_name} get {Asset_URL} "{dest_dir+file_name}"')

            self._batch_insert(self.schema.Execution_Asset_Execution, [{"Execution_Asset": Asset_RID, "Execution": Execution_RID}])
            return dest_dir+file_name

    def upload_execution_assets(self, file_path: str, outputf_path: str, Execution_RID: str):
        os.system(f'deriva-upload-cli {self.host_name}  --catalog eye-ai {file_path} --output-file {outputf_path}')
        with open(outputf_path, 'r') as results:
            upload_results = json.load(results)
        entities = []
        for asset in upload_results.values():
            if asset["State"] == 0 and asset["Result"] is not None:
                rid = asset["Result"].get("RID")
                if rid is not None:
                    entities.append({"Execution_Asset": rid, "Execution": Execution_RID})
        print(entities)
        self._batch_insert(self.schema.Execution_Asset_Execution, entities)

    def execution_start(self, metadata: dict) -> dict:
        # Insert or return process
        Process = []
        for proc in metadata["Process"]:
            proc_RID = self.add_process(proc["Name"], proc["Process_Tag"], proc["Description"], 
                                         proc["owner"], proc["repo"], proc["file_path"], exist_ok = True)
            Process.append(proc_RID)
        # Insert or return Workflow
        workflow_RID = self.add_workflow(metadata["Workflow"]["Name"], metadata["Workflow"]["Description"],
                                         metadata["Workflow"]["owner"], metadata["Workflow"]["repo"],  metadata["Workflow"]["file_path"], 
                                         Process, exist_ok = True)
        # Insert or return Execution
        Execution_RID = self.add_execution(Execution_name = metadata["Execution"]["Name"], description = metadata["Execution"]["Description"]
                                            ,workflow_RID = workflow_RID)
        self._batch_insert(self.schema.Dataset_Execution, [{"Dataset": metadata['Dataset'], "Execution": Execution_RID }])

        self.start_time = datetime.now()
        return {"Execution": Execution_RID, "Workflow": workflow_RID, "Process": Process}
        

    def execution_end(self, Execution_RID: str, file_path: str, outputf_path: str):
        self.upload_execution_assets(file_path, outputf_path, Execution_RID)

        duration = datetime.now() - self.start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        Duration = f'{hours}H {minutes}min {seconds}sec'
        print(f"Execution duration: {Duration}")
        self._batch_update(self.schema.Execution, [{"RID": Execution_RID, "Duration": Duration}], [self.schema.Execution.Duration])
