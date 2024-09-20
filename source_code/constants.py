TASKS_IDX = {
    # "eyes_open": [1],
    # "eyes_open": [2],
    "right_or_left_fist":[6, 10, 14],
    "right_or_left_fist_mi": [3, 7, 11],
    "fists_or_feet": [4, 8, 12],
    "fists_or_feet_mi": [5, 9, 13],
}
ALL_TASKS = range(3, 15)
ALL_SUBJECTS = range(1, 110)

MY_BCI_ARGPARSE_DESCRIPTION = """
Progam to to train and test a classifier model on Physionet dataset.\n
- If no arguments are provided, will run on all the experiments of all the subjects.\n
- To train the model on a specific task and subject, provide their correspinding indices and "train" keyword.\n
- To train the model on a specific task and subject, provide their correspinding indices and "predict" keyword.\n
"""