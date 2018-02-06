import lstm
import time
import json
import numpy
import matplotlib.pyplot as plt

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def write_json_doc(filename, data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, cls=MyEncoder)

#Main Run Thread
if __name__=='__main__':
    global_start_time = time.time()
    epochs  = 1
    seq_len = 50

    print('> Loading data... ')

    X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', seq_len, True)

    print('> Data Loaded. Compiling...')

    model = lstm.build_model([1, 50, 100, 1])

    model.fit(
            X_train,
            y_train,
            batch_size=512,
            nb_epoch=epochs,
            validation_split=0.05)

    predictions = lstm.predict_point_by_point(model, X_test)

    for i, prediction in enumerate(predictions):
        write_json_doc(
                "predictions/{}.json".format(i),
                { 'input': X_test[i], 'output': prediction })
