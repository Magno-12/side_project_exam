import numpy as np
import pandas as pd

def augment_data(data: pd.DataFrame, numeric_columns=[], augmentation_factor=5) -> pd.DataFrame:
    augmented_data = data.copy()
    for _ in range(augmentation_factor - 1):
        new_data = data[numeric_columns].apply(lambda x: x + np.random.normal(loc=0, scale=0.1 * x.std(), size=x.shape))
        augmented_data = pd.concat([augmented_data, new_data], ignore_index=True)
    return augmented_data
