import numpy as np
import random
import kociemba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from classification_solver import ALL_MOVES, Cube, ImitationModel, state_to_numeric

def generate_more_data(num_samples=1000, scramble_min=4, scramble_max=15, output_file="training_data.npz"):
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

# generate_more_data(num_samples=1999800, scramble_min=4, scramble_max=15, output_file="training_data.npz")
    

def continue_training(train_file="training_data.npz",
                      model_path="imitation_model_best.pth",
                      new_model_path="imitation_model_best_continued.pth",
                      epochs=20, batch_size=64, lr=1e-3):
    print('Continuing Training')
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
    model = ImitationModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load previous model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)  # If you saved just state_dict previously
    # If you had a checkpoint with optimizer and epoch info, you'd do:
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch'] + 1
    # best_val_acc = checkpoint['best_val_acc']
    # For simplicity, assume we just have the model weights and start fresh epochs.

    best_val_acc = 0.0  # You can set this if you tracked previously.

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}") as pbar:
            for X,y in train_loader:
                X,y = X.to(device), y.to(device)
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
                X_val,y_val = X_val.to(device), y_val.to(device)
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
            torch.save(model.state_dict(), new_model_path)
            print("Model improved and saved.")

# continue_training(train_file="training_data.npz", model_path="imitation_model_best.pth", epochs=20)