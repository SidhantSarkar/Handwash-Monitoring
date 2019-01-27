import tensorflow as tf
import cv2

#initialise Camera
cap = cv2.VideoCapture(0)

# Loads training models
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# Main loop body
while(True):
    #read and write
    ret,frame = cap.read()
    cv2.imwrite("photo.jpg",frame)

    # Read in the image_data
    image_data = tf.gfile.FastGFile("photo.jpg", 'rb').read()

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy Camera instances
cap.release()
cv2.destroyAllWindows()
