import csv
import json
import time
import threading
import numpy as np
import plotly.graph_objs as go

from flask_socketio import SocketIO
from flask import Flask, render_template, request, redirect, url_for, jsonify

from Neural_network import NeuralNetwork

app = Flask(__name__)
socketio = SocketIO(app)

# Global variables to control training
training = False
pause_training = False
current_epoch = 0
total_epochs = 0
training_results = {}
epoch_history = []
inputs = []
targets = []
learning_rate = 0.1
nn = None  # Global variable of the neural network


def read_csv_and_extract_data(filename):
    """ Function to read csv file containing inputs and targets

    :param filename: Path to file
    :return: Lists containing inputs and targets
    """
    global inputs, targets

    gate_inputs = []
    gate_target = []

    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            gate_inputs.append([int(row[0]), int(row[1])])
            gate_target.append([int(row[2])])

    inputs = np.array(gate_inputs)
    targets = np.array(gate_target)


def load_config():
    """ Function to load configuration selected by user
    
    :return: Configuration file .JSON
    """
    with open('configuration.json', 'r') as file:
        config = json.load(file)
    return config


def create_neural_network_graph(config):
    """ Creates the neural network graph based on the values from config

    :param config: Config JSON file
    :return: returns the neural network graph
    """
    n_inputs = config['inputs']
    hidden_layers = config['hidden_layers']
    n_neurons = config['nodes']
    outputs = config['outputs']

    if len(n_neurons) != hidden_layers:
        raise ValueError("Len of 'nodes' should match with 'hidden_layers'")

    nodes = ['Inputs'] + [f'Hidden layer {i + 1}' for i in range(hidden_layers)] + ['Outputs']
    connections = [('Inputs', 'Hidden layer 1')] + [(f'Hidden layer {i}', f'Hidden layer {i + 1}') for i in
                                                    range(1, hidden_layers)] + [
                      (f'Hidden layer {hidden_layers}', 'Outputs')]

    node_traces = []
    connection_traces = []
    y_positions = [0] * (hidden_layers + 2)
    colors = ['#FFD700', '#87CEEB', '#98FB98', '#FFB6C1', '#FFD700', '#87CEEB', '#98FB98', '#FFB6C1']
    node_size = 20

    for i, node in enumerate(nodes):
        if i == 0:
            n_nodes = n_inputs
            color = colors[0]
        elif i == len(nodes) - 1:
            n_nodes = outputs
            color = colors[-1]
        else:
            n_nodes = n_neurons[i - 1]
            color = colors[i - 1]

        for j in range(n_nodes):
            node_traces.append(go.Scatter(
                x=[i],
                y=[j],
                mode='markers+text',
                name=f'Layer {i} Node {j}',
                text=[0],
                marker=dict(symbol='circle', size=node_size, color=color, line=dict(color='black', width=2)),
                textposition="top center"
            ))

    layout = go.Layout(
        title='Neural Network',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        shapes=[]
    )

    shapes_list = []
    for start, end in connections:
        start_idx = nodes.index(start)
        end_idx = nodes.index(end)
        start_n_nodes = n_inputs if start == 'Inputs' else n_neurons[start_idx - 1]
        end_n_nodes = n_neurons[end_idx - 1] if end != 'Outputs' else outputs

        for start_y in range(start_n_nodes):
            for end_y in range(end_n_nodes):
                connection_name = f'Layer {start_idx} Node {start_y} to Layer {end_idx} Node {end_y}'
                shape = {
                    'type': 'line',
                    'x0': start_idx,
                    'y0': start_y,
                    'x1': end_idx,
                    'y1': end_y,
                    'line': {
                        'color': 'black',
                        'width': 2,
                    },
                    'name': connection_name
                }
                shapes_list.append(shape)

                connection_traces.append(go.Scatter(
                    x=[(start_idx + end_idx) / 2],
                    y=[(start_y + end_y) / 2],
                    mode='text',
                    text=[0],
                    name=f'Text {connection_name}',
                    textposition="middle center"
                ))

    layout['shapes'] = shapes_list

    fig = go.Figure(data=node_traces + connection_traces, layout=layout)
    return fig


@app.route('/')
def index():
    """ Method for index page

    :return: Template for index page
    """
    load_config()
    json_graph = {
        "data": [],
        "layout": {
            "title": "Red Neuronal",
            "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
            "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
            "showlegend": False
        }
    }
    return render_template('index.html', graph_json=json_graph)


@app.route('/configuration', methods=['GET', 'POST'])
def configuration():
    """ Obtains data from the form and saves the info in a JSON file

    :return: Template for configuration
    """

    global total_epochs

    if request.method == 'POST':
        n_inputs = int(request.form['n_inputs'])
        hidden_layers = int(request.form['n_hidden_layers'])
        nodes = list(map(int, request.form['neurons_per_layer'].split(',')))
        outputs = int(request.form['n_outputs'])
        total_epochs = int(request.form['n_epochs'])

        if len(nodes) != hidden_layers:
            return "Error: Len of nodes 'nodes' should match with 'hidden_layers'"

        config = {
            'inputs': n_inputs,
            'hidden_layers': hidden_layers,
            'nodes': nodes,
            'outputs': outputs,
            'epochs': total_epochs
        }

        with open('configuration.json', 'w') as file:
            json.dump(config, file, indent=4)

        return redirect(url_for('render_graph'))

    return render_template('configuration.html')


@app.route('/render_graph')
def render_graph():
    """ Creates the neural network graph

    :return: Template from graph page
    """
    config = load_config()
    graph = create_neural_network_graph(config)

    # Convert Plotly figure to JSONizable
    graph_json = graph.to_json()

    return render_template('graph.html', graph_json=graph_json)


def train_neural_network():
    """ Function to train the network

    :return: None
    """
    global training, pause_training, current_epoch, training_results, nn, epoch_history

    config = load_config()

    input_size = config['inputs']
    hidden_layer_sizes = config['nodes']
    output_size = config['outputs']

    read_csv_and_extract_data('file.csv')
    nn = NeuralNetwork(input_size, hidden_layer_sizes, output_size)

    for epoch in range(current_epoch, total_epochs):
        if not training:
            break
        while pause_training:
            time.sleep(0.1)

        nn.forward(inputs)
        error = nn.mean_squared_error(targets)
        nn.backward(targets, learning_rate)

        current_epoch = epoch + 1
        training_results = {
            "epoch": current_epoch,
            "inputs": inputs.tolist(),
            "predictions": [nn.forward(x).tolist() for x in inputs],
            "targets": targets.tolist(),
            "error": error,
            "weights": nn.get_weights(),
            "activations": nn.get_activations()
        }
        epoch_history.append(training_results.copy())

        if current_epoch % 1000 == 0:
            print(f"Epoch {current_epoch}: Error = {error}")

        socketio.emit('training_results', training_results)
        time.sleep(.1)

    training = False
    socketio.emit('training_results', training_results)


@app.route('/control_training', methods=['POST'])
def control_training():
    """ Function to control the flow of the training

    :return: Returns status of the request
    """
    global training, pause_training, current_epoch, epoch_history, nn, training_results

    try:
        action = request.json.get('action')
        if action == 'play':
            # Restart training only if its finished
            if not training and current_epoch >= total_epochs:
                training = True
                pause_training = False
                current_epoch = 0
                epoch_history = []
                training_results = {}
                threading.Thread(target=train_neural_network).start()
            # Continue training if its paused
            elif not training and current_epoch < total_epochs:
                training = True
                pause_training = False
                threading.Thread(target=train_neural_network).start()
            elif pause_training:
                pause_training = False
        elif action == 'pause':
            pause_training = True
        elif action == 'forward':
            if pause_training:
                run_one_epoch()
        elif action == 'backward':
            if pause_training and current_epoch > 1:
                current_epoch -= 1
                epoch_history.pop()
                training_results = epoch_history[-1] if epoch_history else {}
                socketio.emit('training_results', training_results)
        elif action == 'stop':
            training = False
            pause_training = False
            current_epoch = 0
            epoch_history = []
            training_results = {}
            nn = None
            socketio.emit('training_results', training_results)

        # Returns current values
        return jsonify({"status": "success", "action": action, "results": training_results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


def run_one_epoch():
    """ Function to run only one epoch at a time

    :return: None
    """
    global current_epoch, training_results, nn, epoch_history

    config = load_config()

    input_size = config['inputs']
    hidden_layer_sizes = config['nodes']
    output_size = config['outputs']

    if nn is None:
        nn = NeuralNetwork(input_size, hidden_layer_sizes, output_size)

    nn.forward(inputs)
    error = nn.mean_squared_error(targets)
    nn.backward(targets, learning_rate)

    current_epoch += 1
    training_results = {
        "epoch": current_epoch,
        "inputs": inputs.tolist(),
        "predictions": [nn.forward(x).tolist() for x in inputs],
        "targets": targets.tolist(),
        "error": error,
        "weights": nn.get_weights(),
        "activations": nn.get_activations()
    }
    epoch_history.append(training_results.copy())

    print(f"Epoch {current_epoch}: Error = {error}")
    socketio.emit('training_results', training_results)


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
