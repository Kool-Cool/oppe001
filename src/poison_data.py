import pandas as pd
import numpy as np

# Load the clean dataset
data = pd.read_csv("data/v0/transactions_2022.csv")

# Define poisoning levels
poison_levels = [2, 8, 20]  # in percent

for level in poison_levels:
    poisoned = data.copy()
    
    # Identify class 0 indices
    class_0_idx = poisoned[poisoned['Class'] == 0].index
    
    # Randomly select percentage to flip
    n_to_flip = int(len(class_0_idx) * (level / 100))
    flip_idx = np.random.choice(class_0_idx, n_to_flip, replace=False)
    
    # Flip the class label
    poisoned.loc[flip_idx, 'Class'] = 1
    
    # Save poisoned dataset
    poisoned_file = f"data/v0/poisoned_{level}_percent.csv"
    poisoned.to_csv(poisoned_file, index=False)
    print(f"Saved {poisoned_file} ({n_to_flip} labels flipped)")
