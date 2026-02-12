import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
def load_data(filepath="labeled_data.csv"):
    df = pd.read_csv(filepath)
    
    # Separate Features (X) and Target (y)
    # We drop the label columns and metadata
    X = df.drop(columns=['cluster', 'math_target', 'final_prediction', 'cluster_aligned'], errors='ignore')
    y = df['final_prediction']
    
    return X, y

# Load data
print("Loading data...")
X, y = load_data()

# Split into Train and Test (80% Train, 20% Test)
# Stratify ensures we have equal balance of Walking/Stairs in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the data (Crucial for Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for the App later
joblib.dump(scaler, 'scaler.pkl')

# ==========================================
# 2. SCIKIT-LEARN MODEL (Random Forest)
# ==========================================
print("\n--- Training Scikit-Learn Random Forest ---")

# Initialize and Train
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
y_pred_rf = rf_model.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {acc_rf*100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Save the Sklearn model
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Random Forest model saved as 'random_forest_model.pkl'")

# ==========================================
# 3. PYTORCH NEURAL NETWORK
# ==========================================
print("\n--- Training PyTorch Neural Network ---")

# Convert numpy arrays to PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test.values)

# We will Define the Neural Network Architecture
class ActivityNet(nn.Module):
    def __init__(self, input_size):
        super(ActivityNet, self).__init__()
        # Layer 1: Input -> Hidden (64 neurons)
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        # Layer 2: Hidden -> Output (2 classes: Walking vs Stairs)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize Model
input_dim = X_train.shape[1]
model = ActivityNet(input_dim)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
epochs = 200
print(f"Training for {epochs} epochs...")

for epoch in range(epochs):
    optimizer.zero_grad()           # Clear gradients
    outputs = model(X_train_tensor) # Forward pass
    loss = criterion(outputs, y_train_tensor) # Calculate loss
    loss.backward()                 # Backpropagation
    optimizer.step()                # Update weights
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval() # Set to evaluation mode
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f"Neural Network Accuracy: {accuracy*100:.2f}%")

# Save the PyTorch model
torch.save(model.state_dict(), 'activity_net.pth')
print("PyTorch model saved as 'activity_net.pth'")

print("\nAll Steps Complete. Ready for App Deployment!")