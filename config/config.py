import os
import yaml

class ConfigNamespace:
    """
    A class to dynamically create nested namespaces.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = ConfigNamespace(**value)
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__dict__}"

def load_yaml_file(file_path):
    """
    Load a YAML file and return its contents as a dictionary.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def load_configs(config_directory=None, config_prefix='config_', default_config_file='config.yaml'):
    """
    Load all YAML configuration files in the given directory,
    add prefixes based on file names, and return a combined configuration object.
    The default configuration file is loaded without a prefix.

    :param config_directory: Directory containing YAML configuration files
    :param config_prefix: Prefix to identify configuration files
    :param default_config_file: Name of the default config file to load without a prefix
    :return: ConfigNamespace object with combined configurations
    """

    if config_directory is None:
        # Set config_directory to the directory where the current script is located
        config_directory = os.path.dirname(os.path.abspath(__file__))

    combined_config = {}

    # Load the default configuration file without a prefix
    default_config_path = os.path.join(config_directory, default_config_file)
    if os.path.exists(default_config_path):
        default_config = load_yaml_file(default_config_path)
        combined_config.update(default_config)

    # Load other configuration files with a prefix
    for file_name in os.listdir(config_directory):
        if file_name.startswith(config_prefix) and file_name.endswith('.yaml'):
            config_name = file_name[len(config_prefix):-5]  # Strip prefix and .yaml
            config_data = load_yaml_file(os.path.join(config_directory, file_name))
            combined_config[config_name] = config_data

    return ConfigNamespace(**combined_config)

# debug
if __name__ == "__main__":
    config = load_configs()
    print(config)
 