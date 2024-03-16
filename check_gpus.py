# import tensorflow as tf

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# if tf.test.gpu_device_name():
#     print('\nDefault GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("\nPlease install GPU version of TF\n")

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))