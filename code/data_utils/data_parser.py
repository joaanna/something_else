import os
import json

from collections import namedtuple

ListData = namedtuple('ListData', ['id', 'label', 'path'])


class DatasetBase(object):
    """
    To read json data and construct a list containing video sample `ids`,
    `label` and `path`
    """
    def __init__(self, json_path_input, json_path_labels, data_root,
                 extension, is_test=False):
        self.json_path_input = json_path_input
        self.json_path_labels = json_path_labels
        self.data_root = data_root
        self.extension = extension
        self.is_test = is_test

        # preparing data and class dictionary
        self.classes = self.read_json_labels()
        self.classes_dict = self.get_two_way_dict(self.classes)
        self.json_data = self.read_json_input()

    def read_json_input(self):
        json_data = []
        if not self.is_test:
            with open(self.json_path_input, 'r') as jsonfile:
                json_reader = json.load(jsonfile)
                for elem in json_reader:
                    label = self.clean_template(elem['template'])
                    if label not in self.classes:
                        raise ValueError("Label mismatch! Please correct")
                    item = ListData(elem['id'],
                                    label,
                                    os.path.join(self.data_root,
                                                 elem['id'] + self.extension)
                                    )
                    json_data.append(item)
        else:
            with open(self.json_path_input, 'r') as jsonfile:
                json_reader = json.load(jsonfile)
                for elem in json_reader:
                    # add a dummy label for all test samples
                    item = ListData(elem['id'],
                                    "Holding something",
                                    os.path.join(self.data_root,
                                                 elem['id'] + self.extension)
                                    )
                    json_data.append(item)
        return json_data

    def read_json_labels(self):
        classes = []
        with open(self.json_path_labels, 'r') as jsonfile:
            # print(jsonfile)
            json_reader = json.load(jsonfile)
            for elem in json_reader:
                classes.append(elem)
        return sorted(classes)

    def get_two_way_dict(self, classes):
        classes_dict = {}
        for i, item in enumerate(classes):
            classes_dict[item] = i
            classes_dict[i] = item
        return classes_dict

    def clean_template(self, template):
        """ Replaces instances of `[something]` --> `something`"""
        template = template.replace("[", "")
        template = template.replace("]", "")
        return template


class WebmDataset(DatasetBase):
    def __init__(self, json_path_input, json_path_labels, data_root,
                 is_test=False):
        EXTENSION = ".webm"
        super().__init__(json_path_input, json_path_labels, data_root,
                         EXTENSION, is_test)

