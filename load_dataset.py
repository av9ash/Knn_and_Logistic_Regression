import os
import struct
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot
from Log_Reg import LogisticRegressionOVA
from KNN import KNN
import time


def read(data_set="training", path="."):
    if data_set is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif data_set is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("data set must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        _, _ = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows * cols)

    print(img.shape, lbl.shape)
    return lbl, img


def show(image):
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


def log_reg_driver(images_train, labels_train, images_test, labels_test):
    iterations = [10]
    # iterations = [10, 25, 50, 100]
    # iterations = [10, 30, 50, 70, 100]
    # iterations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    scores = []
    for n_iter in iterations:
        print("Number of iterations: ", n_iter)
        logi = LogisticRegressionOVA(0.00001, n_iter)
        logi.fit(images_train, labels_train)
        # logi = LogisticRegressionOVA(n_iter).fit(images_train[:30000:], labels_train[:30000:])
        scores.append(logi.score(images_test, labels_test))

    print(scores)
    pyplot.plot(iterations, scores, 'r--')
    pyplot.savefig('log_reg.png')
    pyplot.show()


def knn_driver(images_train, labels_train, images_test, labels_test):
    knn = KNN(images_train, labels_train)
    # created batches to avoid Memory Error
    batch_size = 200
    batches = len(images_test) // batch_size

    dists = []
    for i in range(batches):
        print(str(i + 1) + "/" + str(len(images_test) // batch_size))
        dists.extend(knn.compute_distances(images_test[i * batch_size:(i + 1) * batch_size]))

    dists = np.array(dists)

    ks = [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]
    # ks = [1, 10, 50, 70, 90, 100]
    # ks = [100]
    accuracy = []
    predictions = []

    for k in ks:
        print("For K = ", k)
        prediction = knn.predict(dists, k)
        predictions = predictions + list(prediction)

        print("Completed predicting the test data.")

        score = 0
        for i in range(len(labels_test)):
            if int(labels_test[i]) == int(predictions[i]):
                score += 1
        precision = score / len(labels_test)
        accuracy.append(precision)

        print("accuracy: ", precision)
        predictions = []


    # out_file = open("predictions.csv", "w")
    # out_file.write("ImageId,Label\n")
    # for i in range(len(predictions)):
    #     out_file.write(str(int(labels_test[i])) + "," + str(int(predictions[i])) + "\n")
    # out_file.close()

    pyplot.plot(ks, accuracy, 'r--')
    pyplot.savefig('knn.png')
    # pyplot.show()


def main():
    labels_train, images_train = read('training', 'Data/')
    labels_test, images_test = read('testing', 'Data/')

    print('Done loading data!')

    # print('Starting KNN')
    # start_time = time.time()
    # knn_driver(images_train, labels_train, images_test, labels_test)
    # print('Done KNN!')
    # end_time = time.time()
    # print("Elapsed time was %g minutes" % ((end_time - start_time)/60))

    print('Starting Regression..')
    start_time = time.time()
    log_reg_driver(images_train, labels_train, images_test, labels_test)
    print('Done Regression!')
    end_time = time.time()
    print("Elapsed time was %g minutes" % ((end_time - start_time)/60))


if __name__ == '__main__':
    main()
