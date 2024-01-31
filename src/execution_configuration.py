from typing import List
from pydantic import BaseModel
import argparse
import json
from deriva_ml import DerivaML


class Process(BaseModel):
    name: str
    process_tag_name: str
    description: str
    owner: str
    repo: str
    file_path: str


class Workflow(BaseModel):
    name: str
    description: str
    owner: str
    repo: str
    file_path: str


class Execution(BaseModel):
    name: str
    description: str


class AnnotationTag(BaseModel):
    name: str
    description: str
    synonyms: List[str] = []


class DiagnosisTag(BaseModel):
    name: str
    description: str
    synonyms: List[str] = []


class ExecutionConfiguration(BaseModel):
    host: str
    catalog_id: str
    dataset_rid: List[str]
    bdbag_url: List[str]
    models: List[str]
    process: List[Process]
    workflow: Workflow
    execution: Execution
    annotation_tag: AnnotationTag
    diagnosis_Tag: DiagnosisTag


def upload_configuration(hostname: str, catalog_id: str):
    dml = DerivaML(hostname=hostname, catalog_id=catalog_id, schema_name=None)
    # Need to upload the config file here......


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='execution_config',
        description='Validate and upload a DerivaML execution configuration',
        epilog='The Deriva ML library')
    parser.add_argument('configuration_file')  # positional argument
    parser.add_argument('-h', '--hostname')
    parser.add_argument('')
    parser.add_argument('-u', '--upload',
                        help="Upload configuration file to host:catalog_id",
                        nargs=2)
    args = parser.parse_args()

    with open(args.configuration_file, 'r') as file:
        config = json.load(file)
        # check input metadata
    configuration = ExecutionConfiguration.parse_obj(config)
    if args.upload:
        upload_configuration(args.hostname, args.catalog_id)
