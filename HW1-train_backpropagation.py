import math
import random

def normalize(value, min_val, max_val): # ทำให้ค่าอยู่ระหว่าง 0 ถึง 1 (เพื่อป้องกันการครอบงำและเพิ่มประสิทธิภาพของ Algorithm)
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)

def denormalize(value, min_val, max_val): # ทำให้ค่าในช่วง 0 ถึง 1 ให้กลับไปอยู่ในช่วงค่าเดิม (ใช้เพื่อการตีความผลลัพธ์กลับเป็นค่าที่เราใช้งานได้)
    return value * (max_val - min_val) + min_val

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def dot_product(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2))

class NeuralNetwork:
    def __init__(self, input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden1_nodes = hidden1_nodes
        self.hidden2_nodes = hidden2_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights_inh1 = [[random.uniform(-0.5, 0.5) for _ in range(input_nodes)] for _ in range(hidden1_nodes)]
        self.bias_h1 = [random.uniform(-0.5, 0.5) for _ in range(hidden1_nodes)]

        self.weights_h1h2 = [[random.uniform(-0.5, 0.5) for _ in range(hidden1_nodes)] for _ in range(hidden2_nodes)]
        self.bias_h2 = [random.uniform(-0.5, 0.5) for _ in range(hidden2_nodes)]

        self.weights_h2o = [[random.uniform(-0.5, 0.5) for _ in range(hidden2_nodes)] for _ in range(output_nodes)]
        self.bias_o = [random.uniform(-0.5, 0.5) for _ in range(output_nodes)]

    def feedforward(self, input_list):
        inputs = list(input_list)

        # Calculate Hidden Layer 1 outputs
        self.hidden1_inputs_sum = []
        self.hidden1_outputs = []
        for i in range(self.hidden1_nodes):
            sum_val = dot_product(inputs, self.weights_inh1[i]) + self.bias_h1[i]
            self.hidden1_inputs_sum.append(sum_val)
            self.hidden1_outputs.append(sigmoid(sum_val))

        # Calculate Hidden Layer 2 outputs
        self.hidden2_inputs_sum = []
        self.hidden2_outputs = []
        for i in range(self.hidden2_nodes):
            sum_val = dot_product(self.hidden1_outputs, self.weights_h1h2[i]) + self.bias_h2[i]
            self.hidden2_inputs_sum.append(sum_val)
            self.hidden2_outputs.append(sigmoid(sum_val))

        # Calculate Output Layer outputs
        self.final_inputs_sum = []
        self.final_outputs = []
        for i in range(self.output_nodes):
            sum_val = dot_product(self.hidden2_outputs, self.weights_h2o[i]) + self.bias_o[i]
            self.final_inputs_sum.append(sum_val)
            self.final_outputs.append(sigmoid(sum_val))

        return self.final_outputs

    def train(self, input_list, target_list):
        outputs = self.feedforward(input_list)
        targets = list(target_list)

        # --- Calculate Output Layer Error ---
        output_errors = []
        for i in range(self.output_nodes):
            error = targets[i] - outputs[i]
            output_errors.append(error)

        output_gradients = []
        for i in range(self.output_nodes):
            gradient = output_errors[i] * sigmoid_derivative(self.final_inputs_sum[i])
            output_gradients.append(gradient * self.learning_rate)

        # --- Calculate Hidden Layer 2 Error (Propagate from Output) ---
        hidden2_errors = [0] * self.hidden2_nodes
        for i in range(self.output_nodes):
            for j in range(self.hidden2_nodes):
                hidden2_errors[j] += (output_gradients[i] / self.learning_rate) * self.weights_h2o[i][j]

        hidden2_gradients = []
        for i in range(self.hidden2_nodes):
            gradient = hidden2_errors[i] * sigmoid_derivative(self.hidden2_inputs_sum[i])
            hidden2_gradients.append(gradient * self.learning_rate)

        # --- Calculate Hidden Layer 1 Error (Propagate from Hidden Layer 2) ---
        hidden1_errors = [0] * self.hidden1_nodes
        for i in range(self.hidden2_nodes): # Iterate through hidden2 nodes
            for j in range(self.hidden1_nodes): # Iterate through hidden1 nodes
                hidden1_errors[j] += (hidden2_gradients[i] / self.learning_rate) * self.weights_h1h2[i][j]

        hidden1_gradients = []
        for i in range(self.hidden1_nodes):
            gradient = hidden1_errors[i] * sigmoid_derivative(self.hidden1_inputs_sum[i])
            hidden1_gradients.append(gradient * self.learning_rate)

        # --- Update Weights ---

        # Update weights_h2o (hidden2 to output)
        for i in range(self.output_nodes):
            for j in range(self.hidden2_nodes):
                delta_weight = output_gradients[i] * self.hidden2_outputs[j]
                self.weights_h2o[i][j] += delta_weight
        for i in range(self.output_nodes):
            self.bias_o[i] += output_gradients[i]

        # Update weights_h1h2 (hidden1 to hidden2)
        for i in range(self.hidden2_nodes):
            for j in range(self.hidden1_nodes):
                delta_weight = hidden2_gradients[i] * self.hidden1_outputs[j]
                self.weights_h1h2[i][j] += delta_weight
        for i in range(self.hidden2_nodes):
            self.bias_h2[i] += hidden2_gradients[i]

        # Update weights_ih1 (input to hidden1)
        inputs = list(input_list)
        for i in range(self.hidden1_nodes):
            for j in range(self.input_nodes):
                delta_weight = hidden1_gradients[i] * inputs[j]
                self.weights_inh1[i][j] += delta_weight
        for i in range(self.hidden1_nodes):
            self.bias_h1[i] += hidden1_gradients[i]

# --- Data Preparation ----

# Parse raw data into a list of dictionaries
parsed_data = [] # จะเก็บข้อมูลในรูปแบบ list
try:
    with open('Flood_data.txt', 'r') as file: # เปิดไฟล์ Flood_data.txt
        for _ in range(2): # ข้าม 2 บรรทัดแรก
            next(file)
        for line_num, line in enumerate(file, start=3): # อ่านข้อมูลจากบรรทัดที่ 3 เป็นต้นไป
            values = [float(x) for x in line.split('\t') if x.strip()] # ลบช่องว่างนำหน้า/ต่อท้าย และแยกข้อมูลด้วย whitespace
            if len(values) == 9:
                parsed_data.append({
                    "inputs": values[:8],
                    "targets": [values[8]]
            })
            else:
                print(f"line {line_num}: skip") # หากบรรทัดมีข้อมูลไม่ใช่ 9 ตัว ให้ข้ามบรรทัดนั้น
                continue
except FileNotFoundError:
    print("Error: Flood_data.txt not found. Please make sure the file is in the same directory.")
    exit() # Exit if file not found
except Exception as e:
    print(f"Error reading file: {e}")
    exit() # Exit on other file errors

if not parsed_data:
    print("No valid numeric data found in the file.") # ไม่พบข้อมูลตัวเลขที่ถูกต้องในไฟล์
    exit()

all_values = []
for data_row in parsed_data: # หา min/max สำหรับ normalization
    all_values.extend(data_row["inputs"])
    all_values.extend(data_row["targets"])

# Handle case where all_values might be empty or contain only one unique value (if all inputs/targets are identical)
if not all_values:
    print("No values found for normalization. Exiting.")
    exit()
if len(set(all_values)) < 2: # Check if there's enough variation for meaningful normalization
    MIN_VAL = all_values[0]
    MAX_VAL = all_values[0] + 1e-6 # Add a tiny value to prevent division by zero
    print("Warning: All data points are identical or very similar. Normalization might be less effective.")
else:
    MIN_VAL = min(all_values)
    MAX_VAL = max(all_values)


print(f"Min value in dataset: {MIN_VAL}")
print(f"Max value in dataset: {MAX_VAL}")

normalized_training_data = []
for data in parsed_data:
    normalized_inputs = [normalize(x, MIN_VAL, MAX_VAL) for x in data["inputs"]]
    normalized_targets = [normalize(x, MIN_VAL, MAX_VAL) for x in data["targets"]]
    normalized_training_data.append({"inputs": normalized_inputs, "targets": normalized_targets}) # Normalize data

random.shuffle(normalized_training_data) # Shuffle the training data

input_nodes = 8
hidden1_nodes = 10 # Number of nodes in the first hidden layer
hidden2_nodes = 10  # Number of nodes in the second hidden layer
output_nodes = 1
learning_rate = 0.08
epochs = 100 # รอบการฝึก (เพิ่มขึ้นเพื่อให้โอกาสในการลู่เข้ามากขึ้น)

nn = NeuralNetwork(input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate)

print("\n--- Initial Training (before Cross-Validation) ---\n")
for i in range(epochs):
    for data in normalized_training_data:
        nn.train(data["inputs"], data["targets"])
    if (i + 1) % (epochs // 10) == 0:
        total_error = 0
        for data in normalized_training_data:
            output = nn.feedforward(data["inputs"])
            total_error += sum([(data["targets"][j] - output[j])**2 for j in range(len(output))])
        print(f"Epoch {i+1}/{epochs}, Total Squared Error: {total_error:.6f}")
print("\n--- Initial Training Complete ---")

# --- Testing ---

# Example Test Case (using the first row from your data)
if parsed_data: # Ensure there's data before testing
    test_input_raw = parsed_data[0]["inputs"]
    actual_target_raw = parsed_data[0]["targets"][0]

    test_input_normalized = [normalize(x, MIN_VAL, MAX_VAL) for x in test_input_raw]
    predicted_output_normalized = nn.feedforward(test_input_normalized)
    predicted_output_raw = denormalize(predicted_output_normalized[0], MIN_VAL, MAX_VAL)

    print(f"\n--- Prediction Test ---")
    print(f"Input: {test_input_raw}")
    print(f"Actual H(t+7): {actual_target_raw}")
    print(f"Predicted H(t+7) (Raw): {predicted_output_raw:.2f}")

    # Test with another example (last row)
    last_data_point = parsed_data[-1]
    test_input_raw_last = last_data_point["inputs"]
    actual_target_raw_last = last_data_point["targets"][0]
    test_input_normalized_last = [normalize(x, MIN_VAL, MAX_VAL) for x in test_input_raw_last]
    predicted_output_normalized_last = nn.feedforward(test_input_normalized_last)
    predicted_output_raw_last = denormalize(predicted_output_normalized_last[0], MIN_VAL, MAX_VAL)

    print(f"\n--- Another Prediction Test (from training data) ---")
    print(f"Input: {test_input_raw_last}")
    print(f"Actual H(t+7): {actual_target_raw_last}")
    print(f"Predicted H(t+7) (Raw): {predicted_output_raw_last:.2f}")
else:
    print("\nSkipping prediction tests as no data was parsed.")

# ฟังก์ชันสำหรับแบ่งข้อมูลเป็น folds
def create_folds(data, num_folds):
    shuffled_data = list(data) # Make a copy to shuffle
    random.shuffle(shuffled_data)
    fold_size = len(shuffled_data) // num_folds
    folds = []
    for i in range(num_folds):
        start = i * fold_size
        end = start + fold_size
        folds.append(shuffled_data[start:end])
    # Add any remaining data to the last fold if division is not exact
    if len(shuffled_data) % num_folds != 0:
        folds[-1].extend(shuffled_data[num_folds * fold_size:])
    return folds

# --- Main Cross-Validation Loop ---
num_folds = 10
all_folds = create_folds(normalized_training_data, num_folds) # ใช้ข้อมูลที่ Normalize แล้ว

all_predictions = [] # สำหรับเก็บผลทำนาย
all_actuals = []     # สำหรับเก็บค่าจริง
all_accuracies = []  # สำหรับเก็บ Accuracy ของแต่ละ fold
convergence_epochs_per_fold = [] # สำหรับบันทึกความเร็วในการ converge

# Variables for the custom "Confusion Matrix"
total_correct_within_margin = 0
total_under_prediction = 0
total_over_prediction = 0
total_test_samples = 0

print("\n--- Starting 10-Fold Cross-Validation ---")

for i in range(num_folds):
    print(f"\n--- Fold {i+1}/{num_folds} ---")
    test_set = all_folds[i]
    training_set = []
    for j, fold in enumerate(all_folds):
        if i != j:
            training_set.extend(fold)

    if not training_set or not test_set:
        print(f"Skipping Fold {i+1} due to insufficient training or test data.")
        continue

    # Re-initialize the network for each fold to ensure fresh weights
    nn = NeuralNetwork(input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate)

    # --- Training for this fold ---
    min_error = float('inf')
    epochs_no_improve = 0
    patience = 500 # Stop training if error doesn't improve for this many epochs (adjusted for potentially faster convergence)

    for epoch in range(epochs): # epochs ที่กำหนดไว้ทั้งหมด
        random.shuffle(training_set) # Shuffle training set for each epoch
        for data in training_set:
            nn.train(data["inputs"], data["targets"])

        # Calculate error for convergence check
        current_train_error = 0
        for data in training_set:
            output = nn.feedforward(data["inputs"])
            current_train_error += sum([(data["targets"][j] - output[j])**2 for j in range(len(output))])

        # Early stopping check
        if current_train_error < min_error:
            min_error = current_train_error
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and epoch > 100: # Ensure training runs for a minimum before stopping
            print(f"Early stopping at epoch {epoch+1} due to no improvement.")
            convergence_epochs_per_fold.append(epoch + 1)
            break
    else: # If loop completes without early stopping
        convergence_epochs_per_fold.append(epochs) # All epochs were used

    # --- Evaluation for this fold ---
    fold_predictions = []
    fold_actuals = []
    correct_predictions = 0

    ACCEPTABLE_ERROR_MARGIN_RAW = 10.0 # ยอมรับคลาดเคลื่อนไม่เกิน 5 หน่วย
    ACCEPTABLE_ERROR_MARGIN_NORMALIZED = ACCEPTABLE_ERROR_MARGIN_RAW / (MAX_VAL - MIN_VAL)

    # Fold-specific counters for simplified "confusion matrix"
    fold_correct_within_margin = 0
    fold_under_prediction = 0
    fold_over_prediction = 0


    for data in test_set:
        input_norm = data["inputs"]
        target_norm = data["targets"][0] # Get the single target value

        predicted_norm = nn.feedforward(input_norm)[0] # Get the single predicted value

        actual_raw = denormalize(target_norm, MIN_VAL, MAX_VAL)
        predicted_raw = denormalize(predicted_norm, MIN_VAL, MAX_VAL)

        fold_actuals.append(actual_raw)
        fold_predictions.append(predicted_raw)

        # Categorize for simplified "Confusion Matrix"
        if abs(predicted_raw - actual_raw) <= ACCEPTABLE_ERROR_MARGIN_RAW:
            fold_correct_within_margin += 1
        elif predicted_raw < actual_raw:
            fold_under_prediction += 1
        else: # predicted_raw > actual_raw
            fold_over_prediction += 1

    fold_accuracy = fold_correct_within_margin / len(test_set) if len(test_set) > 0 else 0
    all_accuracies.append(fold_accuracy)

    total_test_samples += len(test_set)
    total_correct_within_margin += fold_correct_within_margin
    total_under_prediction += fold_under_prediction
    total_over_prediction += fold_over_prediction

    print(f"Fold {i+1} Accuracy (within {ACCEPTABLE_ERROR_MARGIN_RAW} raw units error): {fold_accuracy:.4f}")
    print(f"  Correct within margin: {fold_correct_within_margin}")
    print(f"  Under predictions: {fold_under_prediction}")
    print(f"  Over predictions: {fold_over_prediction}")
    print(f"  Total test samples in fold: {len(test_set)}")

# --- Overall Results ---
print("\n--- Cross-Validation Results ---")
print(f"Average Convergence Epochs: {sum(convergence_epochs_per_fold) / num_folds:.2f}")
print(f"Average Accuracy across {num_folds} folds (within {ACCEPTABLE_ERROR_MARGIN_RAW} raw units error): {sum(all_accuracies) / num_folds:.4f}")

print("\n--- Overall Prediction Summary (Simplified Confusion Matrix for Regression) ---")
print(f"Total Test Samples: {total_test_samples}")
print(f"Predictions within +/- {ACCEPTABLE_ERROR_MARGIN_RAW} raw units of actual: {total_correct_within_margin} ({total_correct_within_margin/total_test_samples:.2%} of total)")
print(f"Under-predictions (Predicted < Actual - {ACCEPTABLE_ERROR_MARGIN_RAW}): {total_under_prediction} ({total_under_prediction/total_test_samples:.2%} of total)")
print(f"Over-predictions (Predicted > Actual + {ACCEPTABLE_ERROR_MARGIN_RAW}): {total_over_prediction} ({total_over_prediction/total_test_samples:.2%} of total)")

print("\n                                  --- Confusion Matrix ---\n")
print("                        |        Count        |  %.of all Samples  |      Interpretation")
print("                        |                     |                    |")
print("------------------------|---------------------|--------------------|---------------------------------")
print("                        |                     |                    |")
print(f"  Correct within margin |         {total_correct_within_margin}          |       {total_correct_within_margin/total_test_samples:.2%}       |  The model's predictions were very close to the actual values.")
print("                        |                     |                    |")
print("------------------------|---------------------|--------------------|---------------------------------")
print("                        |                     |                    |")
print(f"   Under predictions    |         {total_under_prediction}          |       {total_under_prediction/total_test_samples:.2%}       |  The model's consistently predicted lower than the actual values.")
print("                        |                     |                    |")
print("------------------------|---------------------|--------------------|---------------------------------")
print("                        |                     |                    |")
print(f"   Over predictions     |         {total_over_prediction}         |       {total_over_prediction/total_test_samples:.2%}       |  The model consistently predicted higher than the actual values.")
print("                        |                     |                    |")
print("------------------------|---------------------|--------------------|---------------------------------")
print("                        |                     |                    |")
print(f"   Total Test Samples   |         {total_test_samples}         |         100        |")
print("                        |                     |                    |")
