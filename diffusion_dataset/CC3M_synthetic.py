from .LAION_synthetic import LAIONSyntheticDataset, LAIONCategorySyntheticDataset
import os
import json
from tqdm import tqdm

class CC3MSyntheticDataset(LAIONSyntheticDataset):
    def load_dataset(self, synthetic_data_dir=None, force_regenerate_meta=False):
        if os.path.exists(self.json_path) and not force_regenerate_meta:
            with open(self.json_path, 'r') as f:
                self.data = json.load(f)
        else:
            print('Generating meta.json for CC3M...')
            assert synthetic_data_dir is not None, f'synthetic_data_dir must be provided for {self.__class__.__name__}'

            self.data = []
            for parent_dir, _, _ in tqdm(os.walk(synthetic_data_dir)):
                for entry in os.scandir(parent_dir):
                    if entry.is_file() and entry.name.endswith('.json'):
                        json_path = entry.path
                        # named after xxxx/label_[id number].json
                        image_id = json_path.split('/')[-1].split('.')[0].split('_')[-1]
                        if image_id.lstrip('0') == '':
                            image_id_int = 0
                        else:
                            image_id_int = int(image_id.lstrip('0')) # id number are padded with 0s, need to remove them, int(00xxx) is not allowed
                        parent_folder = image_id_int // 10000

                        image_path = os.path.join(
                            self.root, 
                            'cc3m-images',
                            f'{parent_folder:05d}', 
                            f'{image_id}.jpg'
                        )
                        if os.path.exists(image_path):
                            self.data.append((image_path, json_path))

            with open(self.json_path, 'w') as f:
                json.dump(self.data, f, indent=4)

class CC3MCategorySyntheticDataset(LAIONCategorySyntheticDataset):

    def load_dataset(self, synthetic_data_dir=None, force_regenerate_meta=False):
        if os.path.exists(self.json_path) and not force_regenerate_meta:
            with open(self.json_path, 'r') as f:
                self.data = json.load(f)
        else:
            print('Generating meta.json for CC3M...')
            assert synthetic_data_dir is not None, f'synthetic_data_dir must be provided for {self.__class__.__name__}'

            self.data = []
            for parent_dir, _, _ in tqdm(os.walk(synthetic_data_dir)):
                for entry in os.scandir(parent_dir):
                    if entry.is_file() and entry.name.endswith('.json'):
                        json_path = entry.path
                        # named after xxxx/label_[id number].json
                        image_id = json_path.split('/')[-1].split('.')[0].split('_')[-1]
                        if image_id.lstrip('0') == '':
                            image_id_int = 0
                        else:
                            image_id_int = int(image_id.lstrip('0')) # id number are padded with 0s, need to remove them, int(00xxx) is not allowed
                        parent_folder = image_id_int // 10000

                        image_path = os.path.join(
                            self.root, 
                            'cc3m-images',
                            f'{parent_folder:05d}', 
                            f'{image_id}.jpg'
                        )
                        if os.path.exists(image_path):
                            self.data.append((image_path, json_path))

            with open(self.json_path, 'w') as f:
                json.dump(self.data, f, indent=4)