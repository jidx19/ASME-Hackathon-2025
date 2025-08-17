1. File and Path Setup

The terminal's working directory must be the same as the Python file's location. This ensures the model can find the necessary training data and model parameters.

    file_path_train = "LHS625.csv"

    file_path_results = "universal_kriging_all_results_LHS625_Scalerv3.json"

2. Running the Prediction Model

To get the prediction results, run prediction-LHS625-v3-English.py in the terminal. Follow the prompts to input the required parameters.

3. Running the Optimization Model

To run the multi-start optimization, execute optimization_gridnetwork-v5-English.py. The script will run automatically and output the results upon completion.