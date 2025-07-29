#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>

using namespace std;

double normalize(double value, double min_val, double max_val) {
    if (max_val == min_val) return 0.5;
    return (value - min_val) / (max_val - min_val);
}

double denormalize(double value, double min_val, double max_val) {
    return value * (max_val - min_val) + min_val;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double dot_product(const vector<double>& vec1, const vector<double>& vec2) {
    double result = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) result += vec1[i] * vec2[i];
    return result;
}

class NeuralNetwork {
public:
    int input_nodes, hidden1_nodes, hidden2_nodes, output_nodes;
    double learning_rate;
    vector<vector<double>> weights_inh1, weights_h1h2, weights_h2o;
    vector<double> bias_h1, bias_h2, bias_o;
    vector<double> hidden1_inputs_sum, hidden1_outputs;
    vector<double> hidden2_inputs_sum, hidden2_outputs;
    vector<double> final_inputs_sum, final_outputs;
    
    NeuralNetwork(int in, int h1, int h2, int out, double lr) {
        input_nodes = in;
        hidden1_nodes = h1;
        hidden2_nodes = h2;
        output_nodes = out;
        learning_rate = lr;
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-0.5, 0.5);

        weights_inh1.resize(hidden1_nodes, vector<double>(input_nodes));
        for (auto& row : weights_inh1) for (auto& w : row) w = dis(gen);
        bias_h1.resize(hidden1_nodes); for (auto& b : bias_h1) b = dis(gen);

        weights_h1h2.resize(hidden2_nodes, vector<double>(hidden1_nodes));
        for (auto& row : weights_h1h2) for (auto& w : row) w = dis(gen);
        bias_h2.resize(hidden2_nodes); for (auto& b : bias_h2) b = dis(gen);

        weights_h2o.resize(output_nodes, vector<double>(hidden2_nodes));
        for (auto& row : weights_h2o) for (auto& w : row) w = dis(gen);
        bias_o.resize(output_nodes); for (auto& b : bias_o) b = dis(gen);
    }

    vector<double> feedforward(const vector<double>& input_list) {
        hidden1_inputs_sum.resize(hidden1_nodes);
        hidden1_outputs.resize(hidden1_nodes);
        for (int i = 0; i < hidden1_nodes; ++i) {
            hidden1_inputs_sum[i] = dot_product(input_list, weights_inh1[i]) + bias_h1[i];
            hidden1_outputs[i] = sigmoid(hidden1_inputs_sum[i]);
        }

        hidden2_inputs_sum.resize(hidden2_nodes);
        hidden2_outputs.resize(hidden2_nodes);
        for (int i = 0; i < hidden2_nodes; ++i) {
            hidden2_inputs_sum[i] = dot_product(hidden1_outputs, weights_h1h2[i]) + bias_h2[i];
            hidden2_outputs[i] = sigmoid(hidden2_inputs_sum[i]);
        }

        final_inputs_sum.resize(output_nodes);
        final_outputs.resize(output_nodes);
        for (int i = 0; i < output_nodes; ++i) {
            final_inputs_sum[i] = dot_product(hidden2_outputs, weights_h2o[i]) + bias_o[i];
            final_outputs[i] = sigmoid(final_inputs_sum[i]);
        }
        return final_outputs;
    }

    void train(const vector<double>& input_list, const vector<double>& target_list) {
        vector<double> outputs = feedforward(input_list);

        vector<double> output_errors(output_nodes);
        for (int i = 0; i < output_nodes; ++i)
            output_errors[i] = target_list[i] - outputs[i];

        vector<double> output_gradients(output_nodes);
        for (int i = 0; i < output_nodes; ++i)
            output_gradients[i] = output_errors[i] * sigmoid_derivative(final_inputs_sum[i]) * learning_rate;

        vector<double> hidden2_errors(hidden2_nodes, 0.0);
        for (int i = 0; i < output_nodes; ++i)
            for (int j = 0; j < hidden2_nodes; ++j)
                hidden2_errors[j] += (output_gradients[i] / learning_rate) * weights_h2o[i][j];

        vector<double> hidden2_gradients(hidden2_nodes);
        for (int i = 0; i < hidden2_nodes; ++i)
            hidden2_gradients[i] = hidden2_errors[i] * sigmoid_derivative(hidden2_inputs_sum[i]) * learning_rate;

        vector<double> hidden1_errors(hidden1_nodes, 0.0);
        for (int i = 0; i < hidden2_nodes; ++i)
            for (int j = 0; j < hidden1_nodes; ++j)
                hidden1_errors[j] += (hidden2_gradients[i] / learning_rate) * weights_h1h2[i][j];

        vector<double> hidden1_gradients(hidden1_nodes);
        for (int i = 0; i < hidden1_nodes; ++i)
            hidden1_gradients[i] = hidden1_errors[i] * sigmoid_derivative(hidden1_inputs_sum[i]) * learning_rate;

        for (int i = 0; i < output_nodes; ++i)
            for (int j = 0; j < hidden2_nodes; ++j)
                weights_h2o[i][j] += output_gradients[i] * hidden2_outputs[j];
        for (int i = 0; i < output_nodes; ++i)
            bias_o[i] += output_gradients[i];

        for (int i = 0; i < hidden2_nodes; ++i)
            for (int j = 0; j < hidden1_nodes; ++j)
                weights_h1h2[i][j] += hidden2_gradients[i] * hidden1_outputs[j];
        for (int i = 0; i < hidden2_nodes; ++i)
            bias_h2[i] += hidden2_gradients[i];

        for (int i = 0; i < hidden1_nodes; ++i)
            for (int j = 0; j < input_nodes; ++j)
                weights_inh1[i][j] += hidden1_gradients[i] * input_list[j];
        for (int i = 0; i < hidden1_nodes; ++i)
            bias_h1[i] += hidden1_gradients[i];
    }
};

struct DataRow {
    vector<double> inputs;
    vector<double> targets;
};

int main() {
    vector<DataRow> parsed_data;
    ifstream file("Flood_data.txt");
    string line;
    double MIN_VAL, MAX_VAL;

    if (!file.is_open()) {
        cerr << "Error: Flood_data.txt not found." << endl;
        return 1;
    }

    getline(file, line); // Skip line 1
    getline(file, line); // Skip line 2
    int line_num = 3;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> values;
        string cell;
        while (getline(ss, cell, '\t')) {
            if (!cell.empty()) values.push_back(stod(cell));
        }
        if (values.size() == 9) {
            parsed_data.push_back({vector<double>(values.begin(), values.begin()+8), {values[8]}});
        } else {
            cout << "line " << line_num << ": skip" << endl;
        }
        line_num++;
    }
    file.close();

    if (parsed_data.empty()) {
        cerr << "No valid data found." << endl;
        return 1;
    }

    vector<double> all_values;
    for (const auto& row : parsed_data) {
        all_values.insert(all_values.end(), row.inputs.begin(), row.inputs.end());
        all_values.insert(all_values.end(), row.targets.begin(), row.targets.end());
    }
    if (all_values.empty()) return 1;

    auto [min_it, max_it] = minmax_element(all_values.begin(), all_values.end());
    MIN_VAL = *min_it;
    MAX_VAL = *max_it;
    if (MIN_VAL == MAX_VAL) MAX_VAL += 1e-6;

    vector<DataRow> normalized_data;
    for (const auto& row : parsed_data) {
        vector<double> norm_inputs, norm_targets;
        for (double v : row.inputs) norm_inputs.push_back(normalize(v, MIN_VAL, MAX_VAL));
        for (double v : row.targets) norm_targets.push_back(normalize(v, MIN_VAL, MAX_VAL));
        normalized_data.push_back({norm_inputs, norm_targets});
    }

    shuffle(normalized_data.begin(), normalized_data.end(), mt19937(random_device()()));

    int input_nodes = 8, hidden1_nodes = 10, hidden2_nodes = 10, output_nodes = 1;
    double learning_rate = 0.08;
    int epochs = 100;

    NeuralNetwork nn(input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate);

    for (int i = 0; i < epochs; ++i) {
        for (const auto& row : normalized_data)
            nn.train(row.inputs, row.targets);
    }

    if (!parsed_data.empty()) {
        vector<double> test_input_norm;
        for (double v : parsed_data[0].inputs)
            test_input_norm.push_back(normalize(v, MIN_VAL, MAX_VAL));

        vector<double> result = nn.feedforward(test_input_norm);
        double predicted_raw = denormalize(result[0], MIN_VAL, MAX_VAL);

        cout << "\n--- Prediction Test ---" << endl;
        cout << "Predicted H(t+7): " << predicted_raw << endl;
    }

    return 0;
}
