# Audio Fall Classification

## Installation:
`pip install -r requirements.txt`

## Usage
1. Record the audio file using 

    `$ python preprocess_data.py` 

2. Run the prediction using 

    `$ python predict_class.py -w <weights file name> file <audio file name>`

3. If the system detects a fall audio event, it will send an email to alert the caregiver 

