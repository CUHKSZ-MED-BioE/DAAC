# DAAC
**To do @ywf**
## Overview
**To do @ywf
**
## Datasets
### Preprocessing
(1) AD:(link) ineternal dataset **To do @ywf**
    ADFD:(link) external dataset **To do @ywf**

### Processed data
Here we use AD dataset as an example.

#### AD (Internal) & ADFD (External) Datasets Preparation

This document describes how to prepare and organize the datasets used in our experiments.  
We use two datasets:

- **AD**: internal dataset  
- **ADFD**: external dataset

After preprocessing and alignment, both datasets share the same file format, directory structure, and feature layout.

> Note: The external dataset (ADFD) is aggregated from multiple sources.  
> Users must obtain the raw data themselves and perform preprocessing to match the unified format.  
> Preprocessing scripts for the **internal** dataset (AD) are provided in this repository.

---

#### 1. Directory Structure

After processing, the data should be organized as:

```text
datasets/
  ├── AD/
  │   ├── Feature/
  │   │   ├── feature_1.npy
  │   │   ├── feature_2.npy
  │   │   └── ...
  │   └── Label/
  │       └── label.npy
  └── ADFD/
      ├── Feature/
      │   ├── feature_1.npy
      │   ├── feature_2.npy
      │   └── ...
      └── Label/
          └── label.npy
````

* `AD/` is the **internal dataset**.
* `ADFD/` is the **external dataset**.
* Each dataset contains:

  * `Feature/`: per-patient feature files
  * `Label/`: one label file for all patients

---

#### 2. Processed Data Format

### 2.1 Feature Files

For both **AD** and **ADFD**, the per-patient feature files follow:

* Path pattern:
  `datasets/DATA_NAME/Feature/feature_ID.npy`
* `DATA_NAME` ∈ {`AD`, `ADFD`}
* `ID` is the **patient ID**.

Each `feature_ID.npy` is a 3-D NumPy array with shape:

```text
[N_r, T, C]
```

where:

* `N_r`: number of trials for this patient
* `T`: number of timestamps per trial
* `C`: feature dimensions

> Note: Different patients can have different numbers of trials (`N_r`).

---

##### 2.2 Label File

For both datasets, the label file is:

* Path:
  `datasets/DATA_NAME/Label/label.npy`

`label.npy` is a 2-D NumPy array with shape:

```text
[N_p, 2]
```

where:

* `N_p`: number of patients in the dataset
* Column 0: patient label (e.g., `0` = healthy, `1` = AD, etc.)
* Column 1: patient ID (integer from `1` to `N`)

The ID in the second column must match the `ID` used in `feature_ID.npy`.

---

##### 2.3 Consistency Between Internal and External Datasets

Both **AD (internal)** and **ADFD (external)** must share the **same data structure and feature layout** after preprocessing:

* The shape `[N_r, T, C]` has the same semantics:

  * dimension 0: trial index
  * dimension 1: time steps
  * dimension 2: feature/channel index

However, due to different acquisition schemes and preprocessing pipelines, the **feature types** and their **ordering** in the raw data can differ between the internal and external datasets.

Therefore, when processing each dataset, you must:

1. Carefully check the dataset description (e.g., channel list, feature definitions, ordering).
2. Adjust and reorder the features so that **the same feature index `c` (0 ≤ c < C) corresponds to the same physical/statistical meaning** in both datasets, such as:

   * the same EEG channel or sensor,
   * the same frequency band,
   * the same handcrafted/statistical feature, etc.
3. If required, convert units or normalize feature values so that scales are compatible.

After this alignment step, models can be trained and evaluated on AD and ADFD in a unified way.

---

#### 3. Data Preparation Workflow

##### 3.1 Internal Dataset (AD)

For the **internal dataset (AD)**:

1. Obtain the raw internal data according to your access policy.

2. Use the preprocessing scripts provided in the `data_processing/` directory of this repository.
   These scripts will:

   * read the raw internal data,
   * compute and organize features per trial,
   * ensure that feature types and ordering follow the unified definition,
   * save:

     * `Feature/feature_ID.npy` for each patient,
     * `Label/label.npy` for all patients.

3. After running the scripts, verify that the AD data follows the directory structure and formats described in Sections 1 and 2.

##### 3.2 External Dataset (ADFD)

For the **external dataset (ADFD)**:

1. Collect raw data from the intended external sources.
   You must comply with the corresponding licenses, terms of use, and ethical requirements of each source.

2. Design or implement your own preprocessing pipeline (e.g., scripts or notebooks) to:

   * parse the raw external data,
   * extract features with definitions compatible with the internal dataset,
   * align feature **types** and **ordering** with the internal dataset (see Section 2.3),
   * assign consistent patient IDs and labels.

3. Save the processed external data following the same structure:

   ```text
   datasets/ADFD/Feature/feature_ID.npy
   datasets/ADFD/Label/label.npy
   ```

4. Double-check that:

   * the feature index semantics are identical to the internal dataset,
   * labels and patient IDs are correctly matched.

> We do **not** distribute either raw or processed external data files in this repository.
> Users are responsible for obtaining the external data and performing all necessary preprocessing steps.


## Requirement

The recommended requirements are specified as follows:

Python 3.10
Jupyter Notebook
scikit-learn==1.2.1
torch==1.13.1+cu116
matplotlib==3.7.1
numpy==1.23.5
scipy==1.10.1
pandas==1.5.3
wfdb==4.1.0
neurokit2==0.2.4
The dependencies can be installed by:

`pip install -r requirements.txt`

## Run the code

We have integrated the original three-stage pipeline into a single script. You can run the entire workflow with:

```bash
python AD_MCRD_exp.py
```
After the script finishes, the experimental results (including logs and evaluation metrics) will be saved to:

`test_run/logs/AD/`


