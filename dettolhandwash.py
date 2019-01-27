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

# To keep a check on overall process
steps = [["Step 2",0],["Step 3",0],["Step 4",0],["Step 5",0],["Step 6",0],["Step 7",0]]

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

    # Top prediction
    result = label_lines[top_k[0]]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elif (result != "No Hands"):
        currentStep = 0

        # Detects step to be completed
        for currentStep in range(len(steps)):
            if (steps[currentStep][1] != 5):
                break

        # Checks if prediction matches with step to be done
        if (result == steps[currentStep][0]):
            steps[currentStep][1]+=1
            # Flag for 5 seconds step completion
            if (steps[currentStep][1] == 5):
                print(result + "completed successfully.")
                print("Move onto step " + currentStep+3)

        # Tells user to complete previous step
        else:
            print("Complete step " + currentStep + " for " + 5-steps[currentStep][1] + " seconds.")

    # Tells user to place hands under cameraT
    else:
        print("Please place your hands under the camera.")

# Destroy Camera instances
cap.release()
cv2.destroyAllWindows()
