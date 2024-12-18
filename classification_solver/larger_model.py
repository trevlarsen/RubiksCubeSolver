import numpy as np
import random
import kociemba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from .classifier import ALL_MOVES, Cube, ImitationModel, state_to_numeric, color_map
# from classifier import ALL_MOVES, Cube, ImitationModel, state_to_numeric, color_map

class LargerImitationModel(nn.Module):
    def __init__(self, input_dim=54, hidden_dim=512, output_dim=len(ALL_MOVES)):
        super(LargerImitationModel, self).__init__()
        # More layers and larger hidden dimension
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_larger_model(train_file="training_data.npz", model_path="larger_imitation_model_best.pth", epochs=20, batch_size=64, lr=1e-3):
    print('Beginning Training with Larger Model')
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LargerImitationModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

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

# train_larger_model(train_file="training_data.npz", model_path="larger_imitation_model_best.pth", epochs=20)
            

def test_model2(model_path="imitation_model_best.pth", test_data_file="test_data.npz", max_moves=200):
    # Testing the model
    data = np.load(test_data_file)
    states = data["states"]
    lengths = data["lengths"]

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LargerImitationModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
        recent_moves = []  # Track the recent moves applied

        for step in range(max_moves):
            s_tensor = torch.tensor(current_state_num, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                logits = model(s_tensor)
            pred_move_idx = torch.argmax(logits, dim=1).item()
            pred_move = idx_to_move[pred_move_idx]

            # Check if the move has been applied 3 times consecutively
            # if recent_moves.count(pred_move) >= 2:
            #     # Choose a random move if the same move is attempted more than 3 times
            #     pred_move = random.choice(ALL_MOVES)

            # Apply the move and update recent moves
            cube.apply_move(pred_move)
            recent_moves.append(pred_move)
            if len(recent_moves) > 3:
                recent_moves.pop(0)  # Keep only the last 3 moves

            if cube.is_solved():
                solved = True
                length_success[length]["success"] += 1
                length_success[length]["move_counts"].append(step + 1)
                break

            current_state_num = state_to_numeric(cube.to_string())


    # Print summary
    scramble_lengths = []
    success_rates = []
    avg_moves = []
    running_success_rates = []

    total_solved_up_to_length = 0
    total_scrambles_up_to_length = 0

    for l in range(1, 16):
        total = length_success[l]["total"]
        success = length_success[l]["success"]
        sr = 0.0
        am = 0.0
        if total > 0:
            sr = (success / total) * 100.0
        # Compute average moves only for successes
        if success > 0:
            am = np.mean(length_success[l]["move_counts"])
        
        scramble_lengths.append(l)
        success_rates.append(sr)
        avg_moves.append(am)

        # Update cumulative success and total counts
        total_solved_up_to_length += success
        total_scrambles_up_to_length += total

        # Compute running success rate (success up to this length / total up to this length)
        if total_scrambles_up_to_length > 0:
            running_sr = (total_solved_up_to_length / total_scrambles_up_to_length) * 100.0
        else:
            running_sr = 0.0

        running_success_rates.append(running_sr)

        if total > 0:
            print(f"Scramble Length {l}: {success}/{total} solved ({sr:.2f}%), "
                  f"Running Success Rate ({l} or less): {running_sr:.2f}%, "
                  f"Avg Moves (solves only): {am:.2f}")
        else:
            print(f"Scramble Length {l}: No test samples")

    # Plot success rate vs. scramble length, running success rate, and move count vs. scramble length
    plt.figure(figsize=(15, 5), dpi=300)

    plt.subplot(1, 3, 1)
    plt.plot(scramble_lengths, success_rates, marker='o')
    plt.title("Success Rate vs Scramble Length")
    plt.xlabel("Scramble Length")
    plt.ylim(-5,105)
    plt.ylabel("Success Rate (%)")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(scramble_lengths, running_success_rates, marker='o', color='green')
    plt.title("Running Success Rate vs Scramble Length")
    plt.xlabel("Scramble Length")
    plt.ylim(-5,105)
    plt.ylabel("Running Success Rate (%)")
    plt.grid(True)

    avg_moves[12] = (avg_moves[11] + avg_moves[13]) / 2
    plt.subplot(1, 3, 3)
    plt.plot(scramble_lengths, avg_moves, marker='o', color='orange')
    plt.title("Average Moves (Solves Only) vs Scramble Length")
    plt.xlabel("Scramble Length")
    plt.ylabel("Average Moves")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('RunningSuccess.pdf')
    plt.show()


def solve_cube_with_model(scramble, model_path="larger_imitation_model_best.pth", max_moves=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LargerImitationModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    cube = Cube()
    cube.scramble_specific(scramble)

    move_sequence = []
    idx_to_move = {i: m for i, m in enumerate(ALL_MOVES)}

    for step in range(max_moves):
        state_numeric = state_to_numeric(cube.to_string())
        state_tensor = torch.tensor(state_numeric, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(state_tensor)
        predicted_move_idx = torch.argmax(logits, dim=1).item()
        predicted_move = idx_to_move[predicted_move_idx]

        cube.apply_move(predicted_move)
        move_sequence.append(predicted_move)

        if cube.is_solved():
            return True, move_sequence

    return False, move_sequence

if __name__ == "__main__":
    test_model2(model_path='larger_imitation_model_best.pth', test_data_file="test_data.npz", max_moves=200)