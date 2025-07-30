// โปรแกรมนี้สร้างและฝึกสอนโครงข่ายประสาทเทียมแบบง่ายสำหรับทำนายค่าระดับน้ำท่วม //
// พร้อมทำ cross-validation เพื่อตรวจสอบความแม่นยำ

#include <iostream>      // สำหรับแสดงผลลัพธ์ทางหน้าจอ
#include <vector>        // ใช้งาน vector แทน array แบบยืดหยุ่น
#include <cmath>         // ฟังก์ชันทางคณิตศาสตร์ เช่น tanh()
#include <algorithm>     // ฟังก์ชันเช่น min_element, max_element
#include <random>        // ใช้สุ่มค่าน้ำหนักของโครงข่าย
#include <fstream>       // สำหรับอ่านไฟล์ข้อมูล
#include <sstream>       // แยกสตริงออกเป็นตัวเลข
#include <numeric>       // ใช้สำหรับ accumulate (ถ้ามี)
#include <iomanip>       // จัดตกแต่งตาราง

using namespace std;

// กำหนดค่าคงที่สำหรับโครงข่ายประสาทเทียม
const int INPUT_SIZE = 8;               // จำนวน input features
const int HIDDEN_SIZE = 16;             // จำนวนโหนดใน hidden layer
const double LEARNING_RATE = 0.01;      // อัตราการเรียนรู้
const double MOMENTUM = 0.9;            // ค่าความเฉื่อยของการอัปเดตน้ำหนัก
const int EPOCHS = 300;                 // จำนวนรอบการฝึก
const double TARGET_MARGIN = 10.0;      // ค่าความคลาดเคลื่อนที่ยอมรับได้ (±10)

// ฟังก์ชัน Activation
// ใช้ tanh เป็น activation function
double tanh_act(double x) {
    return tanh(x);
}

// ฟังก์ชันอนุพันธ์ของ tanh สำหรับ backpropagation
double tanh_derivative(double x) {
    return 1.0 - x * x;
}

// ฟังก์ชัน Normalize สำหรับปรับค่าข้อมูลให้อยู่ในช่วง [0,1]
double normalize(double x, double min_val, double max_val) {
    if (max_val - min_val == 0) return 0.0;
    return (x - min_val) / (max_val - min_val);
}

// ฟังก์ชัน Denormalize เพื่อแปลงค่ากลับเป็นค่าเดิม
double denormalize(double x, double min_val, double max_val) {
    return x * (max_val - min_val) + min_val;
}

// โครงสร้างข้อมูล 1 จุดข้อมูล ประกอบด้วย input และ target
struct DataPoint {
    vector<double> inputs;
    double target;
};

// โหลดข้อมูลจากไฟล์ พร้อม normalize
vector<DataPoint> load_dataset(const string &filename, double &min_val, double &max_val) {
    vector<DataPoint> dataset;
    ifstream file(filename);
    string line;

    getline(file, line); // ข้าม header บรรทัดแรก

    vector<double> all_values;
    while (getline(file, line)) {
        replace(line.begin(), line.end(), '\t', ' ');  // แทน tab ด้วย space
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

    // หาค่าต่ำสุดและสูงสุดจากข้อมูลทั้งหมด
    if (!all_values.empty()) {
        min_val = *min_element(all_values.begin(), all_values.end());
        max_val = *max_element(all_values.begin(), all_values.end());
    } else {
        min_val = 0;
        max_val = 1;
    }

    return dataset;
}

// โครงสร้างโครงข่ายประสาทเทียม 1 ชั้นซ่อน
struct NeuralNetwork {
    vector<vector<double>> weights_input_hidden;       // น้ำหนักจาก input → hidden
    vector<double> weights_hidden_output;              // น้ำหนักจาก hidden → output
    vector<double> hidden_output;                      // เก็บค่าผลลัพธ์ของ hidden layer
    vector<double> input_hidden_delta_prev;            // สำหรับ momentum ของ input → hidden
    vector<double> hidden_output_delta_prev;           // สำหรับ momentum ของ hidden → output

    NeuralNetwork() {
        // สุ่มค่าเริ่มต้นให้น้ำหนักทุกเส้นทาง
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-0.5, 0.5);

        weights_input_hidden = vector<vector<double>>(HIDDEN_SIZE, vector<double>(INPUT_SIZE));
        for (auto &row : weights_input_hidden)
            for (double &w : row)
                w = dis(gen);

        weights_hidden_output = vector<double>(HIDDEN_SIZE);
        for (double &w : weights_hidden_output)
            w = dis(gen);

        hidden_output = vector<double>(HIDDEN_SIZE);
        input_hidden_delta_prev = vector<double>(HIDDEN_SIZE * INPUT_SIZE, 0.0);
        hidden_output_delta_prev = vector<double>(HIDDEN_SIZE, 0.0);
    }

    // ฟังก์ชัน forward คำนวณ output ของโครงข่าย
    double forward(const vector<double> &input) {
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            double sum = 0.0;
            for (int j = 0; j < INPUT_SIZE; ++j)
                sum += weights_input_hidden[i][j] * input[j];
            hidden_output[i] = tanh_act(sum);
        }

        double output = 0.0;
        for (int i = 0; i < HIDDEN_SIZE; ++i)
            output += weights_hidden_output[i] * hidden_output[i];

        return output;
    }

    // ฟังก์ชัน train ด้วย backpropagation
    void train(const vector<DataPoint> &dataset, double min_val, double max_val) {
        for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
            double total_error = 0.0;

            for (const auto &dp : dataset) {
                // ทำ normalize ข้อมูลก่อนส่งเข้าโครงข่าย
                vector<double> input(INPUT_SIZE);
                for (int i = 0; i < INPUT_SIZE; ++i)
                    input[i] = normalize(dp.inputs[i], min_val, max_val);

                double target = normalize(dp.target, min_val, max_val);
                double output = forward(input);
                double error = target - output;
                total_error += error * error;

                // อัปเดตน้ำหนักจาก hidden → output
                for (int i = 0; i < HIDDEN_SIZE; ++i) {
                    double delta_output = error * hidden_output[i];
                    weights_hidden_output[i] += LEARNING_RATE * delta_output + MOMENTUM * hidden_output_delta_prev[i];
                    hidden_output_delta_prev[i] = LEARNING_RATE * delta_output;
                }

                // อัปเดตน้ำหนักจาก input → hidden
                for (int i = 0; i < HIDDEN_SIZE; ++i) {
                    double error_hidden = error * weights_hidden_output[i] * tanh_derivative(hidden_output[i]);
                    for (int j = 0; j < INPUT_SIZE; ++j) {
                        double delta = error_hidden * input[j];
                        weights_input_hidden[i][j] += LEARNING_RATE * delta + MOMENTUM * input_hidden_delta_prev[i * INPUT_SIZE + j];
                        input_hidden_delta_prev[i * INPUT_SIZE + j] = LEARNING_RATE * delta;
                    }
                }
            }

            // แสดงผลความคลาดเคลื่อนทุก 10 epoch
            if (epoch % 10 == 0)
                cout << "Epoch " << epoch << "/" << EPOCHS << ", Total Squared Error: " << total_error << endl;
        }
    }

    // ฟังก์ชันทำนายค่าจาก input ที่ยังไม่ normalize
    double predict(const vector<double> &input_raw, double min_val, double max_val) {
        vector<double> input(INPUT_SIZE);
        for (int i = 0; i < INPUT_SIZE; ++i)
            input[i] = normalize(input_raw[i], min_val, max_val);
        double output = forward(input);
        return denormalize(output, min_val, max_val);
    }
};

// ฟังก์ชัน cross-validation เพื่อวัดประสิทธิภาพโมเดล
void cross_validate(vector<DataPoint> dataset, double min_val, double max_val, int folds = 10) {
    random_device rd;
    mt19937 g(rd());
    shuffle(dataset.begin(), dataset.end(), g);

    int fold_size = dataset.size() / folds;
    double total_accuracy = 0.0;
    int total_correct_within_margin = 0;
    int total_under_prediction = 0;
    int total_over_prediction = 0;
    int total_test_samples = 0;

    for (int f = 0; f < folds; ++f) {
        vector<DataPoint> train_set, test_set;
        for (int i = 0; i < dataset.size(); ++i) {
            if (i / fold_size == f)
                test_set.push_back(dataset[i]);
            else
                train_set.push_back(dataset[i]);
        }

        NeuralNetwork net;
        net.train(train_set, min_val, max_val);

        int correct = 0, under = 0, over = 0;

        for (const auto &dp : test_set) {
            double predicted = net.predict(dp.inputs, min_val, max_val);
            double error = predicted - dp.target;

            if (abs(error) <= TARGET_MARGIN)
                correct++;
            else if (error < -TARGET_MARGIN)
                under++;
            else
                over++;
        }

        int total = test_set.size();
        double accuracy = total > 0 ? double(correct) / total : 0.0;
        total_accuracy += accuracy;

        total_correct_within_margin += correct;
        total_under_prediction += under;
        total_over_prediction += over;
        total_test_samples += total;

        cout << "--- Fold " << f + 1 << "/" << folds << " ---\n";
        cout << "Accuracy: " << accuracy << " (" << correct << "/" << total << " within margin)\n";
        cout << "Under: " << under << ", Over: " << over << endl;
        cout << "\n";
    }

    cout << "!!! Average Accuracy across folds: " << total_accuracy / folds << " !!!" << endl;

    // พิมพ์ Confusion Matrix แบบจัดตำแหน่งตรง
    double pc_ttin = (double(total_correct_within_margin) / total_test_samples) * 100.0;
    double pc_ttun = (double(total_under_prediction) / total_test_samples) * 100.0;
    double pc_ttov = (double(total_over_prediction) / total_test_samples) * 100.0;

    cout << "\n                                  --- Confusion Matrix ---\n";
    cout << left
         << setw(26) << " " << "|"
         << setw(20) << "  Count" << "|"
         << setw(20) << "  % of all Samples" << "|"
         << "  Interpretation\n";
    cout << string(90, '-') << "\n";

    cout << left
         << setw(26) << " Correct within margin" << "|"
         << setw(20) << total_correct_within_margin << "|"
         << setw(19) << fixed << setprecision(4) << pc_ttin << "%" << "|"
         << "  The value is within acceptable limits.\n";

    cout << left
         << setw(26) << " Under predictions" << "|"
         << setw(20) << total_under_prediction << "|"
         << setw(19) << fixed << setprecision(4) << pc_ttun << "%" << "|"
         << "  The value is lower than actual value.\n";

    cout << left
         << setw(26) << " Over predictions" << "|"
         << setw(20) << total_over_prediction << "|"
         << setw(19) << fixed << setprecision(4) << pc_ttov << "%" << "|"
         << "  The value is higher than actual value.\n";

    cout << string(90, '-') << "\n";
    cout << left
         << setw(26) << " Total Test Samples" << "|"
         << setw(20) << total_test_samples << "|"
         << setw(20) << "100%" << "|" << endl;
}

void evaluate_cross_file(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filename << endl;
        return;
    }

    int correct = 0, under = 0, over = 0;
    int total = 0;
    string line;

    while (getline(file, line)) { // p0, p1, ...
        // predicted
        getline(file, line);
        stringstream ss_pred(line);
        double pred0, pred1;
        ss_pred >> pred0 >> pred1;

        // actual
        getline(file, line);
        stringstream ss_act(line);
        int act0, act1;
        ss_act >> act0 >> act1;

        int pred_class = (pred0 > pred1) ? 0 : 1;
        int actual_class = (act0 == 1) ? 0 : 1;

        if (pred_class == actual_class)
            correct++;
        else if (pred_class < actual_class)
            under++;
        else
            over++;

        total++;
    }

    double pc_correct = (double(correct) / total) * 100.0;
    double pc_under = (double(under) / total) * 100.0;
    double pc_over = (double(over) / total) * 100.0;

    // สร้างตารางให้ตรงทุกช่อง
    cout << "\n                              --- Confusion Matrix (from cross.txt) ---\n";
    cout << left
         << setw(26) << "Category"
         << "| " << setw(19) << "Count"
         << "| " << setw(20) << "% of Samples"
         << "| " << "Interpretation" << endl;

    cout << string(26, '-') << "|"
         << string(21, '-') << "|"
         << string(22, '-') << "|"
         << string(70, '-') << endl;

    cout << setw(26) << "Correct within margin"
         << "| " << setw(19) << correct
         << "| " << setw(20) << fixed << setprecision(2) << pc_correct
         << "| " << "The value is within acceptable limits." << endl;

    cout << setw(26) << "Under predictions"
         << "| " << setw(19) << under
         << "| " << setw(20) << fixed << setprecision(2) << pc_under
         << "| " << "The value is lower than actual class." << endl;

    cout << setw(26) << "Over predictions"
         << "| " << setw(19) << over
         << "| " << setw(20) << fixed << setprecision(2) << pc_over
         << "| " << "The value is higher than actual class." << endl;

    cout << string(26, '-') << "|"
         << string(21, '-') << "|"
         << string(22, '-') << "|"
         << string(70, '-') << endl;

    cout << setw(26) << "Total Test Samples"
         << "| " << setw(19) << total
         << "| " << setw(20) << "100.00"
         << "| " << endl;
}

int main() {
    double min_val, max_val;
    vector<DataPoint> data = load_dataset("Flood_data.txt", min_val, max_val);  // โหลดข้อมูลและ normalize

    cout << "Min: " << min_val << ", Max: " << max_val << endl;

    if (data.empty()) {
        cerr << "Error: No valid data loaded from file." << endl;  // ถ้าโหลดไม่ได้ให้แจ้งเตือน
        return 1;
    }

    NeuralNetwork model;
    model.train(data, min_val, max_val);  // ฝึกโมเดลกับข้อมูลทั้งหมด

    // ทดสอบการทำนายด้วยตัวอย่าง
    cout << "\n--- Sample Prediction ---\n";
    vector<double> input_sample = {95, 95, 95, 95, 148, 149, 150, 150};
    double pred = model.predict(input_sample, min_val, max_val);
    cout << "Input: ";
    for (double v : input_sample) cout << v << " ";
    cout << "\nPredicted: " << pred << endl;

    // ทำ 10-fold cross-validation
    cout << "\n--- Cross-Validation ---\n";
    cout << "\n";
    cross_validate(data, min_val, max_val);

    evaluate_cross_file("cross.txt");
    return 0;

}
