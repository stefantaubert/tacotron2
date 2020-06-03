
from sklearn.model_selection import train_test_split
import pandas as pd

dest_filename = "/tmp/preprocessed.csv"
data = pd.read_csv(dest_filename, header=None, sep="|")
print(data)

train, test = train_test_split(data, test_size=500, random_state=1234)
train, val = train_test_split(train, test_size=100, random_state=1234)

print(len(train))
print(len(test))
print(len(val))

test_path = "filelist/ljs_audio_text_test_filelist.csv"
train_path = "filelist/ljs_audio_text_train_filelist.csv"
val_path = "filelist/ljs_audio_text_val_filelist.csv"

train.to_csv(train_path, header=None, index=None, sep="|")
test.to_csv(test_path, header=None, index=None, sep="|")
val.to_csv(val_path, header=None, index=None, sep="|")
