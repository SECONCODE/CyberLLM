import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
#from skmultilearn.problem_transform import LabelPowerset
from sklearn.metrics import hamming_loss, precision_recall_fscore_support, label_ranking_loss
from textattack.augmentation import EasyDataAugmenter
from tqdm import tqdm

#'Updated ENISA EXTRACTED2.xlsx'
data = pd.read_csv('XXXX/Updated ENISA EXTRACTED2.csv')

# Extract the desired columns from the DataFrame
cve_ids = data['cve_id'].tolist()
descriptions = data['description'].tolist()
techniques_str = data['label'].astype(str).tolist()

# Augment the descriptions with tqdm for progress tracking
augmentor = EasyDataAugmenter()
augmented_descriptions = []
augmented_techniques_str = []  # Store augmented technique_ids
augmented_cve_ids = []  # Store augmented CVE IDs

for i, description in enumerate(tqdm(descriptions, desc="Augmentation Progress")):
    augmented_description = augmentor.augment(description)
    num_augmentations = len(augmented_description)
    
    # Replicate the corresponding technique_id for each augmented description
    technique_id = techniques_str[i]
    augmented_technique_ids = [technique_id] * num_augmentations
    
    # Replicate the corresponding CVE ID for each augmented description
    cve_id = cve_ids[i]
    augmented_cve_ids.extend([cve_id] * num_augmentations)
    
    augmented_descriptions.extend(augmented_description)
    augmented_techniques_str.extend(augmented_technique_ids)


print("Complete")

# Save the augmented data to a new Excel file
augmented_data = pd.DataFrame({'cve_id': augmented_cve_ids, 'description': augmented_descriptions, 'technique_id': augmented_techniques_str})
#augmented_data.to_excel('augmented_data.xlsx', index=False)
augmented_data.to_csv('augmented_data.csv', index=False)
