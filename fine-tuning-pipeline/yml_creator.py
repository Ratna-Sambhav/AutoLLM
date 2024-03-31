import subprocess
import os
from ruamel.yaml import YAML
import argparse
import json

def train_axolotl_model(action, yml_file_path, dataset_name, dataset_type, yml_updates, new_yml_file_name):
  yaml = YAML(typ='rt')
  with open(yml_file_path, 'r') as file:
    data = yaml.load(file)
  data['datasets'][0]['path'] = dataset_name
  if dataset_type:
    data['datasets'][0]['type'] = dataset_type
  else:
    data['datasets'][0]['type'] = {
      'system_prompt': yml_updates.get('system_prompt', ''),
      'field_system': yml_updates.get('field_system', ''),
      'field_instruction': yml_updates.get('field_instruction', ''),
      'field_output': yml_updates.get('field_output', ''),
      'format': yml_updates.get('format', yml_updates.get('field_instruction', '')),
      'no_input_format': yml_updates.get('no_input_format', yml_updates.get('format', ''))
      }
  # Other Updates
  for item in yml_updates.items():
    if item[0] not in ['system_prompt', 'field_system', 'field_instruction', 'field_output', 'format', 'no_input_format']:
      print(item)
      data[item[0]] = item[1]
  # Save the yml file with new name
  output_file = new_yml_file_name
  with open(output_file, 'w') as file:
      yaml.indent(mapping=2, sequence=4, offset=2)
      yaml.dump(data, file)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Axolotl model with parameters from JSON file")
    parser.add_argument("json_file", help="Path to the JSON file containing parameters")
    args = parser.parse_args()

    # Load parameters from JSON file
    with open(args.json_file, "r") as file:
        params = json.load(file)

    # Extract parameters
    action = params.get("action")
    yml_file_path = params.get("yml_file_path")
    dataset_name = params.get("dataset_name")
    dataset_type = params.get("dataset_type")
    yml_updates = params.get("yml_updates", {})
    new_yml_file_name = "new_yaml.yml"

    # Run train_axolotl_model function
    train_axolotl_model(action, yml_file_path, dataset_name, dataset_type, yml_updates, new_yml_file_name)

if __name__ == "__main__":
    main()
