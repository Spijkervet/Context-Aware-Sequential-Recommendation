## Requirements

torch==0.4.0

numpy==1.14.3

## Data and preprocessing

Data files are stored in the ./data/
Provide data in the following format:

[ [ItemId, DayOfTheWeek, HourOfTheDay], [ItemId, DayOfTheWeek, HourOfTheDay, .... ]

where each row in the file represents a user sequence in chronological order.
An example of the data file is ./data/miniData.json

preprocessAWS.py preprocesses the Amazon Books data to the mentioned format.

## Training and Evaluation

To train and evaluate the model, run the following code (with cuda):

$ python main.py cuda miniData STAR

To train and evaluate the model, run the following code (without cuda):

$ python main.py noCuda miniData STAR

where python main.py (cuda) (data name) (model)

The result log and the state of the training+model are saved under ./output/ folder


## Credits

Code was partially adapted from the code provided by the author of CONTEXT-AWARE RECURRENT NEURAL NETWORK (CARNN)

Link to (CARNN) paper: https://arxiv.org/abs/1609.05787
