# Set the Hyperparameters
IMG_SIZE = 224
EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MODEL = 'EfficientNet' #'ResNet' #'VGG'

# Loading and Preprocessing the Dataset
# Import the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchsummary import summary
from sklearn.model_selection import GroupShuffleSplit
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

seed_value = 42

torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deteministic = True
torch.backends.cudnn.benchmark = False

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load and preprocess the data information file
df = pd.read_csv('./Dataset/data_information.csv')
# df = df[df['Class']==df['Label']]

categories = {'NORMAL':0,'DRUSEN':1,'CNV':2}
# df['Label'] = df['Label'].map(categories)

# Spilit dataframe into train, validation, and test
def spilit(df, test_size=0.2):
    # Create a GroupShuffleSplit object
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed_value)

    # Split the dataset
    train_index, test_index = next(gss.split(df, groups=df['Patient ID']))

    # Create separate DataFrames for the train and test sets
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    
    return train_df, test_df

train_val_df, test_df = spilit(df, test_size=0.2)
train_df, val_df = spilit(train_val_df, test_size=0.2)

val_df = val_df[val_df['Class']==val_df['Label']]
test_df = test_df[test_df['Class']==test_df['Label']]

train_df['Label'] = train_df['Label'].map(categories)
val_df['Label'] = val_df['Label'].map(categories)
test_df['Label'] = test_df['Label'].map(categories)

# Define the dataset class
class OCTDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join('./Dataset/', self.data.iloc[idx]['Directory'])
        label = self.data.iloc[idx]['Label']
        patient_id = self.data.iloc[idx]['Patient ID']
        
        # Load the image from file
        img = Image.open(img_path).convert('RGB')
                
        # Apply transformations
        if self.transform:
            img = self.transform(img)
            
        # Convert label to one-hot encoding
        if label == 0:
            label = torch.tensor([1, 0, 0])
        elif label == 1:
            label = torch.tensor([0, 1, 0])
        else:
            label = torch.tensor([0, 0, 1])
        
        return img, label, patient_id
    
# Set up the data loaders and transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, scale=(0.8, 1.2))], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = OCTDataset(train_df, transform=train_transform)
val_dataset = OCTDataset(val_df, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Defining and Training the Model
# Define the Convolutional Neural Network models
if MODEL == 'VGG':
    model = models.vgg16(weights=True)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, 3)
    model = model.to('cuda')
    
elif MODEL == 'ResNet':
    model = models.resnet34(weights=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model = model.to('cuda')

elif MODEL == 'EfficientNet':
    model = models.efficientnet_v2_s(weights=True)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, 3)
    model = model.to('cuda')
    
# Set up the model and optimizer
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()

# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Count the number of trainable parameters in the model
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of trainable parameters: %d' % num_params)

# Print the model summary
summary(model, (3, 224, 224))

# Define the training and validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(val_loader)
    with torch.no_grad():
        for i, (inputs, labels, patient_ids) in enumerate(val_loader):
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    start_time = time.time()
    for i, (inputs, labels, patient_ids) in enumerate(train_loader):
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Print progress
        progress = (i+1) / total_batches * 100
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"\rTrainig progress: {progress:.2f}% ({i+1}/{total_batches}) Time taken for epoch: {epoch_time:.2f} seconds", end="")
    
    return running_loss / len(train_loader), epoch_time

# Train the model
PATH = MODEL + '_best_model.pth'
best_val_acc = 0.0
early_stopping_patience = PATIENCE

epochs_since_last_improvement = 0

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    train_loss, epoch_time = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Check if the current model has the best validation accuracy so far, and save it if it does
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), PATH)
        print('\nSaved best model with validation accuracy %.2f%%' % (best_val_acc))
        epochs_since_last_improvement = 0
    else:
        epochs_since_last_improvement += 1
        if epochs_since_last_improvement >= early_stopping_patience:
            print('\nStopping early because validation accuracy has not improved in %d epochs' % (early_stopping_patience))
            break
    
    print('\nEpoch %d: train loss = %.4f, val loss = %.4f, val acc = %.2f%%' % (epoch+1, train_loss, val_loss, val_acc))
    print('..........................................................')
    
# Plot the train loss and validation loss
def plot_loss(train_losses, val_losses, window_size=10):
    train_losses_smooth = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
    val_losses_smooth = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)  # Increase DPI for higher quality plot
    
    # Set color palette for the plot
    colors = sns.color_palette('Set2', n_colors=2)
    
    ax.plot(train_losses_smooth, label='Training Loss', color=colors[0])
    ax.plot(val_losses_smooth, label='Validation Loss', color=colors[1])
    ax.legend()
    ax.set_xlabel('Epoch', fontsize=12)  # Increase font size of axis labels
    ax.set_ylabel('Loss', fontsize=12)  # Increase font size of axis labels
    ax.set_ylim([0, 1])  # Set y-axis limit to between 0 and 1
    ax.set_title('Training and Validation Loss', fontsize=14)  # Increase font size of title
    
    # Add grid lines
    ax.grid(True, which='both', linestyle='--')
    ax.xaxis.grid(False)  # Remove x-axis grid lines
    
    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    plt.tight_layout()  # Add padding to prevent labels from getting cut off
    
    plt.show()
    
plot_loss(train_losses, val_losses, window_size=4)

model.load_state_dict(torch.load(PATH))

test_data = OCTDataset(test_df, transform=val_transform)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

model.eval()
correct = 0
total = 0
predictions = []
with torch.no_grad():
    for data in test_loader:
        images, labels, patient_ids = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(labels, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions += predicted.cpu().numpy().tolist()
        
test_df['Prediction'] = predictions

report = metrics.classification_report(test_df['Label'], test_df['Prediction'])
print(report)

result = {'Model': MODEL,
          'Epoch Runtime (sec)': [int(epoch_time)],
          '# Param (mil)': [round(int(num_params) / 1000000, 2)],
          'Accuracy (%)': ['{:.1f}'.format(metrics.accuracy_score(test_df['Label'], test_df['Prediction']) * 100)],
          'Precision   (%)': ['{:.1f}'.format(metrics.precision_score(test_df['Label'], test_df['Prediction'], average='macro') * 100)],
          'Recall   (%)': ['{:.1f}'.format(metrics.recall_score(test_df['Label'], test_df['Prediction'], average='macro') * 100)],
          'F1 Score   (%)': ['{:.1f}'.format(metrics.f1_score(test_df['Label'], test_df['Prediction'], average='macro') * 100)],
         }
pd.DataFrame(result).set_index('Model')

cm = confusion_matrix(test_df['Label'], test_df['Prediction'])
print(cm)


classes = categories.keys()
cm = confusion_matrix(test_df['Label'], test_df['Prediction'])

# plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm, cmap='cool')
ax.grid(False)
ax.set_xlabel('Predicted Labels', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual Labels', fontsize=12, fontweight='bold')
ax.xaxis.set(ticks=np.arange(len(classes)), ticklabels=classes)
ax.yaxis.set(ticks=np.arange(len(classes)), ticklabels=classes)

# annotate the confusion matrix with the counts in each cell
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontsize=20)

plt.show()