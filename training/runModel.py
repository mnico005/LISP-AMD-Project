import os
import sys
import csv
import pandas as pd
import numpy as np
import math

csv.field_size_limit(sys.maxsize)
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_nlp

import keras
#import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow_datasets as tfds
import time
print(keras_nlp.__version__)

import onnxruntime as ort
import numpy as np


keras.mixed_precision.set_global_policy("mixed_float16")


def generate_text(model, input_text, max_length=200):
    start = time.time()

    output = model.generate(input_text, max_length=max_length)
    print("\nOutput:")
    print(output)

    end = time.time()
    print(f"Total Time Elapsed: {end - start:.2f}s")

new_model = tf.keras.models.load_model('lisp.keras')

num = 0
data = []
with open('10mil/FFT.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        #generate_text(gpt2_lm, row['stream'], max_length=MAX_GENERATION_LENGTH)
        generate_text(lora_model, row['stream'], max_length=MAX_GENERATION_LENGTH)
        print("Actual:",row['cpi'])

