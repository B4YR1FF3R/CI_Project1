// โปรแกรมนี้สร้างและฝึกสอนโครงข่ายประสาทเทียมแบบ 2 ชั้นซ่อน สำหรับทำนายค่าระดับน้ำท่วม
// พร้อมปรับจำนวนโหนด hidden layer, learning rate, momentum ได้

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;

const int INPUT_SIZE = 8;
const int OUTPUT_SIZE = 1;
const int EPOCHS = 300;
const double TARGET_MARGIN = 10.0;

// Activation functions
inline double tanh_act(double x) { return tanh(x); }
inline double tanh_derivative(double x) { return 1.0 - x * x; }

// Normalize / Denormalize
inline double normalize(double x, double min_val, double max_val) {
    return (max_val - min_val == 0) ? 0.0 : (x - min_val) / (max_val - min_val);
}

inline double denormalize(double x, double min_val, double max_val) {
    return x * (max_val - min_val) + min_val;
}

struct DataPoint {
    vector<double> inputs;
    double target;
};

vector<DataPoint> load_dataset(const string &filename, double &min_val, double &max_val) {
    vector<DataPoint> dataset;
    ifstream file(filename);
    string line;
    getline(file, line); // skip header

    vector<double> all_values;
    while (getline(file, line)) {
        replace(line.begin(), line.end(), '\t', ' ');
        stringstream ss(line);
        vector<double> values;
        double val;
        while (ss >> val) values.push_back(val);
        if (values.size() == INPUT_SIZE + 1) {
            for (double v : values) all_values.push_back(v);
            DataPoint dp;
            dp.inputs = vector<double>(values.begin(), values.begin() + INPUT_SIZE);
            dp.target = values.back();
            dataset.push_back(dp);
        }
    }
    min_val = *min_element(all_values.begin(), all_values.end());
    max_val = *max_element(all_values.begin(), all_values.end());
    return dataset;
}

struct NeuralNetwork {
    int hidden1_size, hidden2_size;
    double learning_rate, momentum;

    vector<vector<double>> w1, w2;
    vector<double> w3;

    vector<double> a1, a2;
    vector<double> delta1_prev, delta2_prev, delta3_prev;

    mt19937 gen{random_device{}()};
    uniform_real_distribution<> dis{-0.5, 0.5};

    NeuralNetwork(int h1, int h2, double lr, double m) :
        hidden1_size(h1), hidden2_size(h2), learning_rate(lr), momentum(m) {
        w1.resize(hidden1_size, vector<double>(INPUT_SIZE));
        w2.resize(hidden2_size, vector<double>(hidden1_size));
        w3.resize(hidden2_size);

        a1.resize(hidden1_size);
        a2.resize(hidden2_size);

        delta1_prev.resize(hidden1_size * INPUT_SIZE);
        delta2_prev.resize(hidden2_size * hidden1_size);
        delta3_prev.resize(hidden2_size);

        randomize_weights();
    }

    void randomize_weights() {
        for (auto &row : w1) for (double &w : row) w = dis(gen);
        for (auto &row : w2) for (double &w : row) w = dis(gen);
        for (double &w : w3) w = dis(gen);
    }

    double forward(const vector<double> &input) {
        for (int i = 0; i < hidden1_size; ++i) {
            double sum = 0.0;
            for (int j = 0; j < INPUT_SIZE; ++j)
                sum += w1[i][j] * input[j];
            a1[i] = tanh_act(sum);
        }
        for (int i = 0; i < hidden2_size; ++i) {
            double sum = 0.0;
            for (int j = 0; j < hidden1_size; ++j)
                sum += w2[i][j] * a1[j];
            a2[i] = tanh_act(sum);
        }
        double out = 0.0;
        for (int i = 0; i < hidden2_size; ++i)
            out += w3[i] * a2[i];
        return out;
    }

    void train(const vector<DataPoint> &data, double min_val, double max_val) {
        for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
            double total_error = 0.0;
            for (const auto &dp : data) {
                vector<double> input(INPUT_SIZE);
                for (int i = 0; i < INPUT_SIZE; ++i)
                    input[i] = normalize(dp.inputs[i], min_val, max_val);
                double target = normalize(dp.target, min_val, max_val);
                double output = forward(input);
                double error = target - output;
                total_error += error * error;

                // update w3
                for (int i = 0; i < hidden2_size; ++i) {
                    double delta = error * a2[i];
                    w3[i] += learning_rate * delta + momentum * delta3_prev[i];
                    delta3_prev[i] = learning_rate * delta;
                }

                // update w2
                for (int i = 0; i < hidden2_size; ++i) {
                    double err = error * w3[i] * tanh_derivative(a2[i]);
                    for (int j = 0; j < hidden1_size; ++j) {
                        double delta = err * a1[j];
                        int idx = i * hidden1_size + j;
                        w2[i][j] += learning_rate * delta + momentum * delta2_prev[idx];
                        delta2_prev[idx] = learning_rate * delta;
                    }
                }

                // update w1
                for (int i = 0; i < hidden1_size; ++i) {
                    double sum_err = 0.0;
                    for (int k = 0; k < hidden2_size; ++k)
                        sum_err += error * w3[k] * tanh_derivative(a2[k]) * w2[k][i];
                    double err = sum_err * tanh_derivative(a1[i]);
                    for (int j = 0; j < INPUT_SIZE; ++j) {
                        int idx = i * INPUT_SIZE + j;
                        double delta = err * input[j];
                        w1[i][j] += learning_rate * delta + momentum * delta1_prev[idx];
                        delta1_prev[idx] = learning_rate * delta;
                    }
                }
            }
            if (epoch % 10 == 0)
                cout << "Epoch " << epoch << "/" << EPOCHS << ", Error = " << total_error << endl;
        }
    }

    double predict(const vector<double> &input_raw, double min_val, double max_val) {
        vector<double> input(INPUT_SIZE);
        for (int i = 0; i < INPUT_SIZE; ++i)
            input[i] = normalize(input_raw[i], min_val, max_val);
        double output = forward(input);
        return denormalize(output, min_val, max_val);
    }
};

int main() {
    int hidden1 = 16;
    int hidden2 = 12;
    double lr = 0.01;
    double momentum = 0.9;

    double min_val, max_val;
    auto data = load_dataset("Flood_data.txt", min_val, max_val);
    if (data.empty()) {
        cerr << "Error: Failed to load data." << endl;
        return 1;
    }

    NeuralNetwork net(hidden1, hidden2, lr, momentum);
    net.train(data, min_val, max_val);

    vector<double> sample = {95, 95, 95, 95, 148, 149, 150, 150};
    double pred = net.predict(sample, min_val, max_val);

    cout << "\nSample Prediction Result:\nInput: ";
    for (double v : sample) cout << v << " ";
    cout << "\nPredicted Value: " << pred << endl;
    return 0;
}
