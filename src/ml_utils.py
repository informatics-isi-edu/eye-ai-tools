import deriva.core.utils.globus_auth_utils
from deriva.core import ErmrestCatalog, HatracStore, AttrDict, get_credential
import deriva.core.datapath as datapath
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

    @property
    def image(self) -> datapath._TableWrapper:
        return self.eye_ai.Image