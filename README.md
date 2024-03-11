1. Create a virtual environment using this command: python3 -m venv myen
2. Activate the virtual environment: source myenv/bin/activate
3. Create the requirements.txt file: pip freeze > requirements.txt
4. Install all the requirements: pip install -r requirements.txt
5. Create and train the model using: python3 train_mask_detection_model.py
6. Run the model to see the results: python3 detect_mask.py
