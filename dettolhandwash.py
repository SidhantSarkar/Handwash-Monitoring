import tensorflow as tf
import sys
import time
import cv2
import numpy as np
import os

# cap = cv2.VideoCapture(0)
#
# ret, frame = cap.read()
#
# # img2 = cv2.imread(image_path)
# cv2.imwrite("photo.jpg",frame)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # change this as you see fit
# # image_path = "photo.jpg"
#
# # Read in the image_data
# image_data = tf.gfile.FastGFile("photo.jpg", 'rb').read()
#
# # cap = cv2.VideoCapture(0)
#
#
# #
# # ret, frame = cap.read()
# # img2 = cv2.imread(image_path)
# # cv2.imwrite("photo.jpg",img2)
# # img2= cv2.resize(img2,dsize=(299,299), interpolation = cv2.INTER_CUBIC)
# # np_image_data = np.asarray(img2)
# # np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
# # np_final = np.expand_dims(np_image_data,axis=0)
#
# t0 = time.time()
#
# # Loads label file, strips off carriage return
# label_lines = [line.rstrip() for line
#                    in tf.gfile.GFile("tf_files/retrained_labels.txt")]
# # Unpersists graph from file
# with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     _ = tf.import_graph_def(graph_def, name='')
#
# print(_)
# t1 = time.time()
#
# with tf.Session() as sess:
#     # Feed the image_data as input to the graph and get first prediction
#     softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
#
#     predictions = sess.run(softmax_tensor, \
#              {'DecodeJpeg/contents:0': image_data})
#
#     # Sort to show labels of first prediction in order of confidence
#     top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
#
#     for node_id in top_k:
#         human_string = label_lines[node_id]
#         score = predictions[0][node_id]
#         print('%s (score = %.5f)' % (human_string, score))
# t2 = time.time()
#
# print(t2-t1)
#
# cap.release()
# cv2.destroyAllWindows()



import tensorflow as tf
import cv2

#initialise Camera
cap = cv2.VideoCapture(0)

# Main loop body
while(True):
    #read and write
    ret,frame = cap.read()
    cv2.imwrite("photo.jpg",frame)

    # Read in the image_data
    image_data = tf.gfile.FastGFile("photo.jpg", 'rb').read()




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy Camera instances
cap.release()
cv2.destroyAllWindows()
