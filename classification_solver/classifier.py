import numpy as np
import random
import kociemba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Do NOT change these
U_perm = [
    6,3,0, 7,4,1, 8,5,2,
    45,46,47,12,13,14,15,16,17,
    9,10,11,21,22,23,24,25,26,
    27,28,29,30,31,32,33,34,35,
    18,19,20,39,40,41,42,43,44,
    36,37,38,48,49,50,51,52,53
]

R_perm = [
    0,1,20, 3,4,23, 6,7,26,
    15,12,9,16,13,10,17,14,11,
    18,19,29,21,22,32,24,25,35,
    27,28,51,30,31,48,33,34,45,
    36,37,38,39,40,41,42,43,44,
    8,46,47,5,49,50,2,52,53
]

F_perm = [
    0,1,2, 3,4,5, 44,41,38,
    6,10,11, 7,13,14, 8,16,17,
    24,21,18,25,22,19,26,23,20,
    15,12,9,30,31,32,33,34,35,
    36,37,27,39,40,28,42,43,29,
    45,46,47,48,49,50,51,52,53
]

D_perm = [
    0,1,2,3,4,5,6,7,8,
    9,10,11,12,13,14,24,25,26,
    18,19,20,21,22,23,42,43,44,
    33,30,27,34,31,28,35,32,29,
    36,37,38,39,40,41,51,52,53,
    45,46,47,48,49,50,15,16,17
]

L_perm = [
    53,1,2,50,4,5,47,7,8,
    9,10,11,12,13,14,15,16,17,
    0,19,20,3,22,23,6,25,26,
    18,28,29,21,31,32,24,34,35,
    42,39,36,43,40,37,44,41,38,
    45,46,33,48,49,30,51,52,27
]

B_perm = [
    11,14,17,3,4,5,6,7,8,
    9,10,35,12,13,34,15,16,33,
    18,19,20,21,22,23,24,25,26,
    27,28,29,30,31,32,36,39,42,
    2,37,38,1,40,41,0,43,44,
    51,48,45,52,49,46,53,50,47
]

ALL_MOVES = ["U","U'","U2","R","R'","R2","F","F'","F2","D","D'","D2","L","L'","L2","B","B'","B2"]
color_map = {'U':0,'R':1,'F':2,'D':3,'L':4,'B':5}

def inverse_perm(perm):
    inv = [0]*54
    for i,p in enumerate(perm):
        inv[p] = i
    return inv

def double_perm(perm):
    return [perm[perm[i]] for i in range(54)]

MOVE_PERM = {
    'U': U_perm,
    'R': R_perm,
    'F': F_perm,
    'D': D_perm,
    'L': L_perm,
    'B': B_perm
}

MOVE_PERM["U'"] = inverse_perm(MOVE_PERM["U"])
MOVE_PERM["U2"] = double_perm(MOVE_PERM["U"])
MOVE_PERM["R'"] = inverse_perm(MOVE_PERM["R"])
MOVE_PERM["R2"] = double_perm(MOVE_PERM["R"])
MOVE_PERM["F'"] = inverse_perm(MOVE_PERM["F"])
MOVE_PERM["F2"] = double_perm(MOVE_PERM["F"])
MOVE_PERM["D'"] = inverse_perm(MOVE_PERM["D"])
MOVE_PERM["D2"] = double_perm(MOVE_PERM["D"])
MOVE_PERM["L'"] = inverse_perm(MOVE_PERM["L"])
MOVE_PERM["L2"] = double_perm(MOVE_PERM["L"])
MOVE_PERM["B'"] = inverse_perm(MOVE_PERM["B"])
MOVE_PERM["B2"] = double_perm(MOVE_PERM["B"])

class Cube:
    def __init__(self):
        self.state = list("UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB")

    def from_string(self, s):
        self.state = list(s)

    def to_string(self):
        return "".join(self.state)

    def apply_move(self, move):
        perm = MOVE_PERM[move]
        new_state = [None]*54
        for i in range(54):
            new_state[i] = self.state[perm[i]]
        self.state = new_state

    def apply_moves(self, moves_str):
        for m in moves_str.split():
            self.apply_move(m)

    def scramble(self, length):
        seq = []
        for _ in range(length):
            m = random.choice(ALL_MOVES)
            seq.append(m)
            self.apply_move(m)
        return " ".join(seq)

    def scramble_specific(self, moves):
        for move in moves:
            self.apply_move(move)

    def is_solved(self):
        return self.to_string() == "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

def state_to_numeric(s):
    return np.array([color_map[c] for c in s], dtype=np.int8)

import numpy as np
import random
import kociemba

def generate_data(num_samples=1000, scramble_min=1, scramble_max=15, output_file="training_data.npz"):
    move_to_idx = {m: i for i, m in enumerate(ALL_MOVES)}
    states_list = []
    moves_list = []

    cube = Cube()

    # Try loading the existing data
    try:
        existing_data = np.load(output_file)
        existing_states = existing_data['states']
        existing_moves = existing_data['moves']
        print(f"Loaded existing data from {output_file}. Current data size: {len(existing_states)} samples.")
    except FileNotFoundError:
        existing_states = np.array([])
        existing_moves = np.array([])
        print(f"No existing data found. Generating new data from scratch.")

    # Generate new samples
    for i in range(num_samples):
        cube.from_string("UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB")
        length = random.randint(scramble_min, scramble_max)
        cube.scramble(length)
        scrambled_state = cube.to_string()

        try:
            solution = kociemba.solve(scrambled_state)
            solution_moves = solution.split()
            if len(solution_moves) > 0:
                first_move = solution_moves[0]
                states_list.append(scrambled_state)
                moves_list.append(first_move)
        except:
            pass

    if len(states_list) == 0:
        raise ValueError("No valid training samples generated. Check permutations or cube setup.")

    # Convert new states and moves to numeric representations
    states_array = np.stack([state_to_numeric(s) for s in states_list])
    moves_array = np.array([move_to_idx[m] for m in moves_list], dtype=np.int64)

    # Append the new data to the existing data
    all_states = np.vstack([existing_states, states_array]) if existing_states.size > 0 else states_array
    all_moves = np.hstack([existing_moves, moves_array]) if existing_moves.size > 0 else moves_array

    print(f"New data added: {states_array.shape[0]} samples. Total data size: {all_states.shape[0]} samples.")

    # Save the updated data
    np.savez(output_file, states=all_states, moves=all_moves)
    print(f"Training data saved to {output_file} with {all_states.shape[0]} samples.")


def generate_test_data(num_samples=500, scramble_min=1, scramble_max=15, output_file="test_data.npz"):
    print('Generating Test data')
    states_list = []
    lengths_list = []
    cube = Cube()

    for i in range(num_samples):
        cube.from_string("UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB")
        length = random.randint(scramble_min, scramble_max)
        cube.scramble(length)
        scrambled_state = cube.to_string()
        states_list.append(scrambled_state)
        lengths_list.append(length)

    states_array = np.stack([state_to_numeric(s) for s in states_list])
    lengths_array = np.array(lengths_list, dtype=np.int64)

    np.savez(output_file, states=states_array, lengths=lengths_array)
    print(f"Test data saved to {output_file} with {len(states_list)} samples.")

class ImitationModel(nn.Module):
    def __init__(self, input_dim=54, hidden_dim=128, output_dim=len(ALL_MOVES)):
        super(ImitationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(train_file="training_data.npz", model_path="imitation_model_best.pth", epochs=20, batch_size=64, lr=1e-3):
    print('Beginning Training')
    data = np.load(train_file)
    states = data["states"]
    moves = data["moves"]

    states_tensor = torch.tensor(states, dtype=torch.long)
    moves_tensor = torch.tensor(moves, dtype=torch.long)
    dataset = TensorDataset(states_tensor, moves_tensor)

    val_ratio = 0.1
    val_size = int(len(dataset)*val_ratio)
    train_size = len(dataset)-val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ImitationModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        # Use tqdm for training loop
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}") as pbar:
            for X,y in train_loader:
                X,y = X, y
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs,y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()*X.size(0)
                running_correct += (outputs.argmax(dim=1)==y).sum().item()
                running_total += X.size(0)

                pbar.update(1)

        train_loss = running_loss/running_total
        train_acc = running_correct/running_total

        model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        val_running_total = 0
        with torch.no_grad():
            for X_val,y_val in val_loader:
                X_val,y_val = X_val, y_val
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs,y_val)
                val_running_loss += val_loss.item()*X_val.size(0)
                val_running_correct += (val_outputs.argmax(dim=1)==y_val).sum().item()
                val_running_total += X_val.size(0)

        val_loss_avg = val_running_loss/val_running_total
        val_acc = val_running_correct/val_running_total

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc*100:.2f}%")

        if val_acc>best_val_acc:
            best_val_acc=val_acc
            torch.save(model.state_dict(), model_path)
            print("Model improved and saved.")

def test_model(model_path="imitation_model_best.pth", test_data_file="test_data.npz", max_moves=200):
    # Testing the model
    data = np.load(test_data_file)
    states = data["states"]
    lengths = data["lengths"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImitationModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    idx_to_move = {i:m for i,m in enumerate(ALL_MOVES)}
    inv_color_map = {v:k for k,v in color_map.items()}

    # Track statistics
    length_success = {l:{"success":0,"total":0,"move_counts":[] } for l in range(1,16)}

    for i in range(len(states)):
        length = lengths[i]
        length_success[length]["total"] += 1

        facelets = [inv_color_map[v] for v in states[i]]
        cube = Cube()
        cube.from_string("".join(facelets))

        solved = False
        current_state_num = states[i]
        for step in range(max_moves):
            s_tensor = torch.tensor(current_state_num, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(s_tensor)
            pred_move_idx = torch.argmax(logits, dim=1).item()
            pred_move = idx_to_move[pred_move_idx]

            cube.apply_move(pred_move)
            if cube.is_solved():
                solved = True
                length_success[length]["success"] += 1
                length_success[length]["move_counts"].append(step+1)
                break
            current_state_num = state_to_numeric(cube.to_string())

    # Print summary
    scramble_lengths = []
    success_rates = []
    avg_moves = []

    for l in range(1,16):
        total = length_success[l]["total"]
        success = length_success[l]["success"]
        sr = 0.0
        am = 0.0
        if total > 0:
            sr = (success/total)*100.0
        # Compute average moves only for successes
        if success > 0:
            am = np.mean(length_success[l]["move_counts"])
        scramble_lengths.append(l)
        success_rates.append(sr)
        avg_moves.append(am)

        if total>0:
            print(f"Scramble Length {l}: {success}/{total} solved ({sr:.2f}%), Avg Moves (solves only): {am:.2f}")
        else:
            print(f"Scramble Length {l}: No test samples")

    # Plot success rate vs. scramble length
    plt.figure(figsize=(10,4), dpi=300)
    plt.subplot(1,2,1)
    plt.plot(scramble_lengths, success_rates, marker='o')
    plt.title("Success Rate vs Scramble Length")
    plt.xlabel("Scramble Length")
    plt.ylabel("Success Rate (%)")
    plt.grid(True)

    # Plot average move count vs. scramble length (for successful solves only)
    plt.subplot(1,2,2)
    # Some scramble lengths might have 0 successful solves, resulting in avg_moves=0. Filter those out.
    scramble_lengths_filtered = [l for l,am in zip(scramble_lengths, avg_moves) if am>0]
    avg_moves_filtered = [am for am in avg_moves if am>0]
    plt.plot(scramble_lengths_filtered, avg_moves_filtered, marker='o', color='orange')
    plt.title("Average Moves (Solves Only) vs Scramble Length")
    plt.xlabel("Scramble Length")
    plt.ylabel("Average Moves")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Summary
    for l in range(1,16):
        total = length_success[l]["total"]
        success = length_success[l]["success"]
        if total>0:
            rate = (success/total)*100.0
            print(f"Scramble Length {l}: {success}/{total} solved ({rate:.2f}%)")
        else:
            print(f"Scramble Length {l}: No test samples")

if __name__ == "__main__":
    # Generate data with scrambles between 1 and 15
    # generate_data(num_samples=1000000, scramble_min=1, scramble_max=15, output_file="training_data.npz")
    # generate_test_data(num_samples=1500, scramble_min=1, scramble_max=15, output_file="test_data.npz")

    # Train on GPU with tqdm progress updates
    # train_model(train_file="training_data.npz", model_path="imitation_model_best.pth", epochs=20, batch_size=64, lr=1e-3)

    # Test with up to 200 moves
    # test_model(model_path="imitation_model_best.pth", test_data_file="test_data.npz", max_moves=200)
    pass