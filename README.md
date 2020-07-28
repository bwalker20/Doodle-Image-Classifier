# Doodle-Image-Classifier
Python program that uses Google's hand drawn image database to create a predictive model and a Tkinter interface to allow a user to draw an image and have the model guess what the user drew.

draw_model contains a predictive model with 15 classifications which is very accurate.
final_model contains a predictive model with 40 classifications which is not very accurate.

paint.py will run the draw_model predictive model.
paint_final will run the final_model predictive model.

To use train to produce the 40 classification model, download Googles hand drawn image numpy library which can be found here: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pli=1
place these files in a folder one directory up from these files and name it google_data. Then run the merge_data.py file to create the trainable numpy files. Now you can run train.py
