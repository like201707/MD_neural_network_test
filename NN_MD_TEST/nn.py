import numpy as np
import pylab as pl
import XYZReader
import tensorflow as tf
import sys

def readEnergy(filename):
    """
    Read energy only
    """

    with open(filename, 'r') as inFile:

        E = []
        inFile.readline()
        for line in inFile:
            words = line.split()
            E.append([float(words[-2])])

    return E

def create_placeholders(numAtomsPerFrame, n_x, n_y, n_f):
    """
    creat placeholder for input X and Y
    n_x is the number of features for input
    n_y usually is 1
    """
    X = tf.placeholder(tf.float32, shape = (numAtomsPerFrame, n_x), name="X")
    Y = tf.placeholder(tf.float32, shape = (n_y), name="Y")
    F = tf.placeholder(tf.float32, shape = (numAtomsPerFrame, n_f), name="F")
    F_P = tf.placeholder(tf.float32, shape = (numAtomsPerFrame, n_f), name="F")
    return X, Y, F, F_P

def initialize_parameters():
    """
    initialize papameters for the neural network
    """
    W1 = tf.get_variable("W1", [2, 5], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [1, 5], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [5, 5], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [1, 5], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [5, 5], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [1, 5], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [5, 5], initializer = tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [1, 5], initializer = tf.zeros_initializer())
    W5 = tf.get_variable("W5", [5, 5], initializer = tf.contrib.layers.xavier_initializer())
    b5 = tf.get_variable("b5", [1, 5], initializer = tf.zeros_initializer())
    W6 = tf.get_variable("W6", [5, 1], initializer = tf.contrib.layers.xavier_initializer())
    b6 = tf.get_variable("b6", [1, 1], initializer = tf.zeros_initializer())
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3,\
                  "b3": b3, "W4": W4, "b4": b4, "W5": W5, "b5": b5, "W6": W6, "b6": b6}

    return parameters

def forward_propagation(X, parameters):
    """
    inplement forward forward_propagation
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    W6 = parameters['W6']
    b6 = parameters['b6']
    Z1 = tf.add(tf.matmul(X, W1), b1)
    A1 = tf.nn.tanh(Z1)
    Z2 = tf.add(tf.matmul(A1, W2), b2)
    A2 = tf.nn.tanh(Z2)
    Z3 = tf.add(tf.matmul(A2, W3), b3)
    A3 = tf.nn.tanh(Z3)
    Z4 = tf.add(tf.matmul(A3, W4), b4)
    A4 = tf.nn.tanh(Z4)
    Z5 = tf.add(tf.matmul(A4, W5), b5)
    A5 = tf.nn.tanh(Z5)
    Z6 = tf.add(tf.matmul(A5, W6), b6)
    A6 = Z6
    return Z6

def compute_cost(Z6, Y, numAtomsPerFrame, F_prediction, F, alpha = 0.01):
    """
    compute the cost for the nueral network
    using mean square error
    """
    prediction = tf.reduce_sum(Z6, axis=0)
    cost = tf.reduce_mean(tf.square((prediction - Y)/numAtomsPerFrame) + alpha/(3*numAtomsPerFrame)*tf.reduce_sum(tf.square(F_prediction-F)))
    return cost


def model(X_train, Y_train, X_test, Y_test, F_train, num_frames_train, numAtomsPerFrame, \
          plotCost = False, plotPrediction = False, learning_rate = 0.001):
    """
    traing the nueral network
    """
    n_x = 2
    n_y = 1
    n_f = 3
    costs = []
#    model_path = "/Users/keli/Desktop/git repo"
    dG2 = np.load('dG2.npy')
    dG4 = np.load('dG4.npy')
    X, Y, F, F_P = create_placeholders(numAtomsPerFrame, n_x, n_y, n_f)
    parameters = initialize_parameters()
    Z6 = forward_propagation(X, parameters)
    cost = compute_cost(Z6, Y, numAtomsPerFrame, F_P, F, alpha = 0.1)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    dE_dG = tf.gradients(Z6, X)
    predictions = np.zeros((num_frames_test,1))
#    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for frame in range(num_frames_train):
            batch_y_train = Y_train[frame].reshape(n_y)
            batch_x_train = X_train[frame].reshape(numAtomsPerFrame, n_x)
            batch_f_train = F_train[frame].reshape(numAtomsPerFrame, n_f)

            dE_dG2 = np.array(sess.run(dE_dG, feed_dict={X:batch_x_train, Y: batch_y_train}))[:, :, 0].reshape(numAtomsPerFrame, 1)
            dE_dG4 = np.array(sess.run(dE_dG, feed_dict={X:batch_x_train, Y: batch_y_train}))[:, :, 1].reshape(numAtomsPerFrame, 1)
            batch_fp_train = - (dG2[frame] * dE_dG2 + dG4[frame] * dE_dG4)
            frame_cost = sess.run([cost], feed_dict={X: batch_x_train, Y: batch_y_train, F_P: batch_fp_train, F: batch_f_train})
            sess.run(optimizer, feed_dict={X: batch_x_train, Y: batch_y_train})
            costs.append(frame_cost)
#            prediction_train = np.sum(sess.run(Z6, feed_dict={X: batch_x_train, Y: batch_y_train}))
#            print prediction_train
            #show progress
            sys.stdout.write("\r%2d %% train complete" % ((float(frame)/num_frames_train)*100))
            sys.stdout.flush()
        print("Optimization Finished!")
#        save_path = saver.save(sess, model_path)
#        print("Model saved in file: %s" % save_path)

        for frame in range(num_frames_test):
            batch_x_test = X_test[frame].reshape(numAtomsPerFrame, n_x)
            batch_y_test = Y_test[frame].reshape(n_y)
            prediction = np.sum(sess.run(Z6, feed_dict={X: batch_x_test}))
            predictions[frame] = prediction
            sys.stdout.write("\r%2d %% test complete" % ((float(frame)/num_frames_test)*100))
            sys.stdout.flush()
        print("Test Finished!")

    if plotPrediction == True:
        fig2 = pl.figure()
        X2 = np.arange(0, num_frames_test, 1)
        Y2 = np.array(predictions)
        Y3 = np.array(Y_test[:num_frames_test])
        pl.plot(X2, Y2, 'ro', X2, Y3, 'bo')
#        pl.plot(Y2, Y3, 'ro')
        fig2.savefig('Prediction.png', dpi=fig2.dpi)

    if plotCost == True:
        X1 = np.arange(0, num_frames_train, 1)
        Y1 = np.array(costs)
        fig = pl.figure()
        pl.plot(X1, Y1, 'b-', linewidth=2)
        pl.xlabel("train times", size=10)
        pl.ylabel("Cost", size=10)
        pl.xticks(size=10)
        pl.yticks(size=10)
        fig.savefig('Cost.png', dpi=fig.dpi)


if __name__ == '__main__':
    X_train = XYZReader.XYZReader("Si.lammpstrjInput.dat").atomC[:, : , :2]
    X_test = XYZReader.XYZReader("Si.lammpstrjInput.dat").atomC[:, : , :2]
    Y_train = np.array(readEnergy("Eng.dat"))
    Y_test = np.array(readEnergy("Eng.dat"))
    num_frames_train = X_train.shape[0]
    num_frames_test = X_test.shape[0]
    F_train = np.load('F.npy')[:num_frames_train]
    parameters = model(X_train, Y_train, X_test, Y_test, F_train, num_frames_train, numAtomsPerFrame = 512, plotCost = True, plotPrediction = True)
