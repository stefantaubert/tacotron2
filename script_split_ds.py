if __name__ == "__main__":
  from sklearn.model_selection import train_test_split
  import pandas as pd
  from script_ds_pre import csv_separator

  dest_filename = "/tmp/preprocessed.csv"
  data = pd.read_csv(dest_filename, header=None, sep=csv_separator)
  print(data)

  train, test = train_test_split(data, test_size=500, random_state=1234)
  train, val = train_test_split(train, test_size=100, random_state=1234)

  #print(len(train))
  #print(len(test))
  #print(len(val))

  test_path = "filelist/ljs_audio_text_test_filelist.csv"
  train_path = "filelist/ljs_audio_text_train_filelist.csv"
  val_path = "filelist/ljs_audio_text_val_filelist.csv"

  train.to_csv(train_path, header=None, index=None, sep=csv_separator)
  test.to_csv(test_path, header=None, index=None, sep=csv_separator)
  val.to_csv(val_path, header=None, index=None, sep=csv_separator)
  print("Dataset is splitted in train-, val- and test-set.")