#! /usr/bin/env python 

import warnings

# Suppress the specific UserWarning
warnings.filterwarnings("ignore", message="Failed to load image Python extension*")

import os
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import monai
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,accuracy_score,balanced_accuracy_score
from monai.data import DataLoader
from monai.data.utils import pad_list_data_collate
from monai.utils import set_determinism
from scipy.stats import ttest_rel,wilcoxon
from monai.transforms import (
    Resize,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate90,
    ScaleIntensity,
)

#Set up the depthwise and pointwise operator layers in the bottleneck residual blocks of the MobileNetV2 architecture
class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv3D, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

#Build out the MobileNetV2 architecture, using ReLu instead of ReLu6 and 13 instead of 17 bottleneck layers 
class MobileNetV2_3D(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(MobileNetV2_3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),nn.BatchNorm3d(32),nn.ReLU(inplace=True),
            DepthwiseSeparableConv3D(32, 64, stride=1),nn.BatchNorm3d(64),nn.ReLU(inplace=True),
            DepthwiseSeparableConv3D(64, 128, stride=2),nn.BatchNorm3d(128),nn.ReLU(inplace=True),
            DepthwiseSeparableConv3D(128, 128, stride=1),nn.BatchNorm3d(128),nn.ReLU(inplace=True),
            DepthwiseSeparableConv3D(128, 256, stride=2),nn.BatchNorm3d(256),nn.ReLU(inplace=True),
            DepthwiseSeparableConv3D(256, 256, stride=1),nn.BatchNorm3d(256),nn.ReLU(inplace=True),
            DepthwiseSeparableConv3D(256, 512, stride=2),nn.BatchNorm3d(512),nn.ReLU(inplace=True),
            DepthwiseSeparableConv3D(512, 512, stride=1),nn.BatchNorm3d(512),nn.ReLU(inplace=True),
            DepthwiseSeparableConv3D(512, 512, stride=1),nn.BatchNorm3d(512),nn.ReLU(inplace=True),
            DepthwiseSeparableConv3D(512, 512, stride=1),nn.BatchNorm3d(512),nn.ReLU(inplace=True),
            DepthwiseSeparableConv3D(512, 512, stride=1),nn.BatchNorm3d(512),nn.ReLU(inplace=True),
            DepthwiseSeparableConv3D(512, 512, stride=2),nn.BatchNorm3d(512),nn.ReLU(inplace=True),
            DepthwiseSeparableConv3D(512, 512, stride=1),nn.BatchNorm3d(512),nn.ReLU(inplace=True),
            DepthwiseSeparableConv3D(512, 1024, stride=2),nn.BatchNorm3d(1024),nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def set_random_seed(seed_value):
    set_determinism(seed=seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Define a function that will count the number of patients in each class
def extract_filenames(root_dir, class_names):
    image_files = [[os.path.join(root_dir, class_names[i], x) for x in os.listdir(os.path.join(root_dir, class_names[i]))] for i in range(len(class_names))]
    num_each = [len(files) for files in image_files]
    image_files_list = [file for files in image_files for file in files]
    image_class_list = [i for i, num in enumerate(num_each) for _ in range(num)]
    return image_files_list, image_class_list, num_each

#Define a function to call for cross-validation
def split_data_values(data, fold_number, total_folds):
    # Calculate the size of each fold
    fold_size = len(data) // total_folds

    # Calculate the start and end indices for the validation fold
    val_start = (fold_number - 1) * fold_size
    val_end = fold_number * fold_size

    # Split the data into training and validation sets
    val_set = data[val_start:val_end]
    train_set = np.concatenate((data[:val_start],data[val_end:]))
    
    return train_set, val_set

#Check if GPU is useable
if torch.cuda.is_available():
    print("CUDA is available.")

#Define directory to save and pull data
root_dir = "/home/fletcher.barrett/Sex_Classifier/"
directories = {
    "development": os.path.join(root_dir, "Images_cv/Development"),
    "test": os.path.join(root_dir, "Images_cv/Test")
}

class_names = sorted(x for x in os.listdir(directories["development"]) if os.path.isdir(os.path.join(directories["development"], x)))

#Create dictionary of development and testing data (shuffle indices of each group so batches will be relatively stratified)
data = {}
image_dimensions = []
for split, directory in directories.items():
    image_files_list, image_class_list, num_each = extract_filenames(directory, class_names)
    data[f"{split}_image_files_list"] = image_files_list
    data[f"{split}_image_class_list"] = image_class_list
    data[f"{split}_num_each"] = num_each
    for image_file in image_files_list:   
      nifti_img = nib.load(image_file)
      image_data = nifti_img.get_fdata()
      image_dimensions.append(image_data.shape)
      
      #If you want to view the central slice of each MRI, uncomment the following lines. These lines will save each image to the root directory
      
      #slice_index = image_data.shape[0]/2
      #selected_slice = image_data[int(slice_index),:,:]
      #plt.imshow(selected_slice, cmap='gray')
      #plt.axis('off')
      #plt.colorbar()
      #image_title = image_file.split('/')[-1][:-7]
      #plt.imsave(root_dir+f'{image_data.shape}_{image_title}.png', selected_slice, cmap='gray')
      #plt.close()
      
    num_total = len(image_files_list)
    print(f"Total image count in {split}: {num_total}")
    print(f"Label counts in {split}: {num_each}")
    print('\n')
    indices = np.arange(num_total)
    np.random.shuffle(indices)
    data[f"{split}_x"] = [image_files_list[i] for i in indices]
    data[f"{split}_y"] = [image_class_list[i] for i in indices]

dimension_counts = {}

# Iterate over the list of image dimensions
for dimension in image_dimensions:
    # If the dimension already exists in the dictionary, increment its count
    if dimension in dimension_counts:
        dimension_counts[dimension] += 1
    # If the dimension is new, initialize its count to 1
    else:
        dimension_counts[dimension] = 1

# Print the counts of each unique dimension
for dimension, count in dimension_counts.items():
    print(f"{dimension}: {count}")
    
#Print the new image size for user
new_img_size = (96, 128, 128)
print(f"Images resized to: {new_img_size}")
print(f"Label names: {class_names}")

#Create the transforms applied to training data and testing&validation data separately
train_transforms = Compose(
    [
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensity(), 
        Resize(new_img_size), 
        RandFlip(spatial_axis=0, prob=0.5),
        RandRotate90()
    ]
)

val_transforms = Compose(
    [
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensity(),
        Resize(new_img_size)
    ]
)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img = self.transforms(self.image_files[index])
        return img, self.labels[index]

#Define the batch size according to the image sizes used and the GPU RAM available
batch_size = 15
print(f"Batch size: {batch_size}")

#Define the number of runs through model development that will be performed, the number of folds in cross-validation, the types of architectures, and the directory to write results to
total_runs = 3
val_split = 5
architectures = ['DenseNet121','MobileNetV2','ResNet18']
results_dir = root_dir+'Results'

#Loop through the types of architectures
for architecture in architectures:
  os.makedirs(results_dir+f'/{architecture}', exist_ok=True)
  
  #Loop through the number of model development runs
  for run in np.arange(1,total_runs+1):   
      set_random_seed(run)   
      os.makedirs(results_dir+f'/{architecture}/Run{run}', exist_ok=True)
      
      #Loop through the folds
      for fold in np.arange(1,val_split+1):
          torch.cuda.empty_cache()
          root_dir = results_dir+f'/{architecture}/Run{run}/Fold{fold}'  
          os.makedirs(root_dir, exist_ok=True)
          
          #Define the training and testing data using cross validation
          data['train_x'],data['val_x'] = split_data_values(data['development_x'], fold, val_split)
          data['train_y'],data['val_y'] = split_data_values(data['development_y'], fold, val_split)
  
          # Define nifti dataset, data loader
          data['train_y'] = [int(x) for x in data['train_y']]
          train_labels = torch.nn.functional.one_hot(torch.as_tensor(data['train_y'])).float()
          val_labels = torch.nn.functional.one_hot(torch.as_tensor(data['val_y'])).float()
          
          # create a training data loader
          train_ds = Dataset(image_files=data['train_x'], labels=train_labels, transforms=train_transforms)
          train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available(),collate_fn=pad_list_data_collate)
          
          # create a validation data loader
          val_ds = Dataset(image_files=data['val_x'], labels=val_labels, transforms=val_transforms)
          val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available(),collate_fn=pad_list_data_collate)
          
          #Create the model instance
          device = torch.device("cuda")
          
          if architecture == 'MobileNetV2':
              model = MobileNetV2_3D(in_channels=1).to(device) 
          #ResNet18 architecture with original channels (64,128,256,512) is too complex for memory availability  
          if architecture == 'ResNet18':
              model = monai.networks.nets.ResNet('basic',layers=[2, 2, 2, 2],block_inplanes=[16, 32, 64, 128], spatial_dims=3, n_input_channels=1, conv1_t_size=7, conv1_t_stride=1, no_max_pool=False, shortcut_type='B', widen_factor=1.0, num_classes=2, feed_forward=True, bias_downsample=True).to(device) 
          if architecture == 'DenseNet121':
              model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)   
          
          #Classes are in equal proportion so no weights are needed
          loss_function = torch.nn.CrossEntropyLoss()
          
          optimizer = torch.optim.Adam(model.parameters(), 1e-4)
          
          #Start a typical PyTorch training
          val_interval = 1
          best_metric = -1
          best_metric_epoch = -1
          epoch_loss_values = []
          metric_values = []
          writer = SummaryWriter()
          max_epochs = 30
          patience = 10
          
          results_list = []
          
          #Loop through the number of epochs allowed for each fold
          for epoch in range(max_epochs):
              print("-" * 10)
              print(f"Architecture {architecture}, run {run}, fold {fold}, epoch {epoch + 1}/{max_epochs}")
              model.train()
              epoch_loss = 0
              step = 0
              
              #Train the model
              for batch_data in train_loader:
                  step += 1
                  inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                  optimizer.zero_grad()
                  outputs = model(inputs)
                  loss = loss_function(outputs, labels)
                  loss.backward()
                  optimizer.step()
                  epoch_loss += loss.item()
                  epoch_len = len(train_ds) // train_loader.batch_size
                  print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                  writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
              
              epoch_loss /= step
              epoch_loss_values.append(epoch_loss)
              print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
              
              #Evaluate the model on the training data and the validation data
              if (epoch + 1) % val_interval == 0:
                  model.eval()
                  train_y_true = []
                  train_y_pred = []
                  val_y_true = []
                  val_y_pred = []
                  
                  with torch.no_grad():
                      for batch_data in train_loader:
                          inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                          outputs = model(inputs)
                          pred = outputs.argmax(dim=1)
                          
                          for i in range(len(pred)):
                              train_y_true.append(torch.argmax(labels[i]).item())
                              train_y_pred.append(pred[i].item())
          
                  with torch.no_grad():
                      for val_data in val_loader:
                          val_images, val_labels = (val_data[0].to(device), val_data[1].to(device))
                          outputs = model(val_images)
                          pred = outputs.argmax(dim=1)
                          
                          for i in range(len(pred)):
                              val_y_true.append(torch.argmax(val_labels[i]).item())
                              val_y_pred.append(pred[i].item())
                
                  # Calculate balanced accuracy (only really necessary for imbalanced data)
                  train_metric = balanced_accuracy_score(train_y_true, train_y_pred)
                  val_metric = balanced_accuracy_score(val_y_true, val_y_pred)
                  metric_values.append(val_metric)
              
                  results_list.append([epoch,train_metric,val_metric])
            
              if val_metric > best_metric:
                  counter = 0
                  best_metric = val_metric
                  best_metric_epoch = epoch + 1
                  torch.save(model.state_dict(), root_dir + "/best_metric_model_classification3d_array.pth")
                  print("saved new best metric model")
                  
              else:
                  counter += 1
      
              print(f"Current epoch: {epoch+1} current training/validation balanced accuracy: {train_metric:.4f} / {val_metric:.4f} ")
              print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
              writer.add_scalar("val_accuracy", val_metric, epoch + 1)
              
              if counter == patience:
                  break
          
          print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
          writer.close()
          
          model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_classification3d_array.pth")))
          
          test_labels = torch.nn.functional.one_hot(torch.as_tensor(data['test_y'])).float()
          # create a testing data loader
          test_ds = Dataset(image_files=data['test_x'], labels=test_labels, transforms=val_transforms)
          test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available(),collate_fn=pad_list_data_collate)
          
          itera = iter(test_loader)
          
          def get_next_im():
              test_data = next(itera)
              return test_data[0][0].unsqueeze(0).to(device), test_data[1][0].unsqueeze(0).to(device)
          
          img, label = get_next_im()
          
          # Get the occlusion sensitivity map
          occ_sens = monai.visualize.OcclusionSensitivity(nn_module=model, mask_size=8, n_batch=1)
          
          # Only get a single slice to save time.
          depth_slice = img.shape[2] // 2
          occ_sens_b_box = [depth_slice - 1, depth_slice, -1, -1, -1, -1]
          
          occ_result, _ = occ_sens(x=img, b_box=occ_sens_b_box)
          occ_result = occ_result[0, label.argmax().item()][None]
          
          fig, axes = plt.subplots(1, 2, figsize=(25, 10), facecolor="white")
          
          # Plot original image
          ax1 = axes[0]
          im_show1 = ax1.imshow(np.squeeze(img[0, 0, depth_slice, ...].detach().cpu()), cmap="gray")
          ax1.axis("off")
          fig.colorbar(im_show1, ax=ax1)
          ax1.set_aspect('auto')  
          
          # Plot occlusion sensitivity map
          ax2 = axes[1]
          im_show2 = ax2.imshow(np.squeeze(occ_result[0].detach().cpu()), cmap="jet")
          ax2.axis("off")
          fig.colorbar(im_show2, ax=ax2)
          ax2.set_aspect('auto') 
          
          # Save the plot
          plt.savefig(os.path.join(root_dir, "occlusion.png"))
          plt.close()
          
          #Define a function to plot some examples from testing with their predicted and true labels
          def plot_images(root_dir, images, true_labels, pred_labels,image_number,depth_slice):
              plt.figure(figsize=(12, 6))
              rows = int(image_number//np.sqrt(image_number))
              cols = int(rows+image_number%np.sqrt(image_number))
              for i in range(len(images)):
                  plt.subplot(rows, cols, i + 1)
                  image_slice = images[i][0][depth_slice, :, :]
                  plt.imshow(image_slice, cmap='gray')
                  plt.title(f"Predicted Class: {class_names[pred_labels[i]]}\nTrue Class: {class_names[true_labels[i]]}", fontsize=10)
                  plt.axis('off')
              plt.tight_layout()
              plt.savefig(os.path.join(root_dir, "example_predictions.png"))
              plt.close()
          
          model.eval()
          y_true = []
          y_pred = []
          with torch.no_grad():
              for test_data in test_loader:
                  test_images, test_labels = (test_data[0].to(device), test_data[1].to(device))
                  pred = model(test_images).argmax(dim=1)
          
                  for i in range(len(pred)):
                      y_true.append(torch.argmax(test_labels[i]).item())
                      y_pred.append(pred[i].item())
          
          test_images_to_show = 4
          plot_images(root_dir, test_images.cpu().numpy()[:test_images_to_show], y_true[:test_images_to_show], y_pred[:test_images_to_show],test_images_to_show,depth_slice) 
          
          #Calculate the performance metrics        
          print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
          
          cm = confusion_matrix(y_true, y_pred)
          disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
          disp.plot()
          plt.savefig(os.path.join(root_dir, "confusion_matrix.png"))
          plt.close()
          
          accuracy = balanced_accuracy_score(y_true, y_pred)  
          class_accuracies = {name: accuracy_score(np.array(y_true) == i, np.array(y_pred) == i) for i, name in enumerate(class_names)}
          
          class_accuracies['Balanced'] = accuracy
          np.save(os.path.join(root_dir, "accuracy.npy"), class_accuracies)
          np.save(os.path.join(root_dir, "learning_results.npy"), np.array(results_list))
          np.save(os.path.join(root_dir, "testing_results.npy"), np.array(accuracy))
          
          print(class_accuracies)


#Define a function to plot the learning curve for all epochs of a single run
def plot_learning_curves(results_folds, testing_folds, subplot_title, architecture, run, min_y, max_y):
    best_val = []
    best_train = []
    best_test = []
    max_epoch = []
    title = f'{architecture} Architecture, Run {run}'
    
    for results_fold in results_folds:
        best_val.append(np.max([result[2] for result in results_fold]))
        best_train.append([result[1] for result in results_fold][[result[2] for result in results_fold].index(best_val[-1])])
        max_epoch.append(len([result[2] for result in results_fold]))
        
    for testing_fold in testing_folds:
        best_test.append(testing_fold)
        
    # Sample data for each subplot
    data_lists = {}
    for i, (results_fold, best_val_, best_train_, subplot_title_) in enumerate(zip(results_folds, best_val, best_train,subplot_title), start=1):
        label = f"{subplot_title_}, Train: {np.round(best_train_, 2)},\n Val: {np.round(best_val_, 2)},\n Test: {np.round(best_test[i-1], 2)},\n Max Epoch: {max_epoch[i-1]}"
        data_lists[label] = {"Training": [result[1] for result in results_fold], "Validation": [result[2] for result in results_fold]}
        

    fig, axs = plt.subplots(1, len(data_lists), figsize=(20, 6))
    
    for (subplot_label, subplot_data), ax in zip(data_lists.items(), axs.flatten()):
        for label, data in subplot_data.items():
            ax.plot(np.arange(1,len(data)+1), data, label=label)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Classification Accuracy')
        ax.legend()
        ax.set_title(subplot_label)
        ax.set_xlim(1, len(data))
        ax.set_ylim(min_y, max_y)
    
    plt.suptitle(title)
    plt.tight_layout()    
    plt.savefig(results_dir+f'/{architecture}/Run{run}/{title}.png')
    plt.close()
    
    return best_train,best_val,best_test,np.min(max_epoch)

#Define a function to extract the learning results from model development and model testing
def results_full(architectures, total_runs, min_y, max_y):
    all_results = {}
    min_epoch_per_run = []

    for architecture in architectures:
        training_all = []
        validation_all = []
        testing_all = []
        
        all_training_results = []
        all_validation_results = []
        
        for run in np.arange(1,total_runs+1):
            results_fold = [np.load(results_dir+f'/{architecture}/Run{run}/Fold{i}/learning_results.npy') for i in range(1, 6)]
            testing_results_fold = [np.load(results_dir+f'/{architecture}/Run{run}/Fold{i}/testing_results.npy') for i in range(1, 6)]
            best_train,best_val,best_test,min_epoch = plot_learning_curves(results_fold, testing_results_fold, ['Fold1','Fold2','Fold3','Fold4','Fold5'], architecture, run, min_y, max_y)
            
            min_epoch_per_run.append(min_epoch)
            all_training_results.append([i[:,1] for i in results_fold])
            all_validation_results.append([i[:,2] for i in results_fold])
            
            training_all = np.concatenate([training_all,best_train])
            validation_all = np.concatenate([validation_all,best_val])
            testing_all = np.concatenate([testing_all,best_test])
        
        flattened_list_training = [item.tolist() for sublist in all_training_results for item in sublist]
        flattened_list_validation = [item.tolist() for sublist in all_validation_results for item in sublist]
        max_length = max(len(sublist) for sublist in flattened_list_training)
        padded_list_training = [sublist + [np.nan] * (max_length - len(sublist)) for sublist in flattened_list_training]
        padded_list_validation = [sublist + [np.nan] * (max_length - len(sublist)) for sublist in flattened_list_validation]
        average_learning_curve = [np.nanmean(padded_list_training,axis=0),np.nanmean(padded_list_validation,axis=0)]
        standarddev_learning_curve = [np.nanstd(padded_list_training,axis=0),np.nanstd(padded_list_validation,axis=0)]

        
        train_mean,train_std = np.round(np.mean(training_all),2),np.round(np.std(training_all),2)
        val_mean,val_std = np.round(np.mean(validation_all),2),np.round(np.std(validation_all),2)
        test_mean,test_std = np.round(np.mean(testing_all),2),np.round(np.std(testing_all),2)
        
        key = f'{architecture} Architecture'
        all_results[key] = {
            'Training': (training_all, train_mean, train_std),
            'Validation': (validation_all, val_mean, val_std),
            'Testing': (testing_all, test_mean, test_std),
            'Average Learning': (average_learning_curve,standarddev_learning_curve)
        }
        
        print(f'{architecture} Architecture:')   
        print(train_mean,train_std)
        print(val_mean,val_std)
        print(test_mean,test_std)
    
    return all_results,np.min(min_epoch_per_run)

#Define a function to calculate the statistical difference between validation results from different architectures
def conduct_t_test(results_dict, metric):
    combinations = list(results_dict.keys())
    num_combinations = len(combinations)
    
    for i in range(num_combinations):
        for j in range(i + 1, num_combinations):
            combination1 = combinations[i]
            combination2 = combinations[j]
            
            testing_results1 = results_dict[combination1][metric][0]
            testing_results2 = results_dict[combination2][metric][0]
            
            # Paired Student's t-test
            t_statistic, p_value_ttest = ttest_rel(testing_results1, testing_results2)
            
            # Wilcoxon signed-rank test
            _, p_value_wilcoxon = wilcoxon(testing_results1, testing_results2)
            
            print(f"Paired T-test between {metric} scores of {combination1} and {combination2}:")
            print(f"P-value (T-test): {p_value_ttest}")
            if p_value_ttest < 0.05:
                print("Statistically different (T-test)\n")
            else:
                print("Not statistically different (T-test)\n")
            
            print(f"Wilcoxon signed-rank test between {metric} scores of {combination1} and {combination2}:")
            print(f"P-value (Wilcoxon): {p_value_wilcoxon}")
            if p_value_wilcoxon < 0.05:
                print("Statistically different (Wilcoxon)\n")
            else:
                print("Not statistically different (Wilcoxon)\n")

#Define a function to plot the average learning curves for training and validation for each architecture tested in model development
def plot_average_learning_curve_single(architectures, results_dict,min_y,max_y,min_epoch_to_plot):
    
    colours = ['b', 'r', 'g']
    colour_counter = 0
    
    plt.figure(figsize=(10,10))
    
    for architecture in architectures:
        results_array = results_dict[f'{architecture} Architecture']['Average Learning']
        
        train_score = results_array[0][0]
        valid_score = results_array[0][1]
        
        train_std = results_array[1][0]
        valid_std = results_array[1][1]
        
        print(f'{architecture}, '+'Train-Validation: '+str(np.round(np.average(np.abs(train_score-valid_score)),3))+', Avg train std: '+str(np.round(np.average(train_std),3))+', Avg validation std: '+str(np.round(np.average(valid_std),3)))
        
        epochs = range(1, len(train_score) + 1)
        
        plt.plot(epochs, train_score, colours[colour_counter], linestyle='solid', label=f'{architecture} Architecture, Training')
        plt.plot(epochs, valid_score, colours[colour_counter], linestyle='dashed', label=f'{architecture} Architecture, Validation')
        plt.fill_between(epochs, 
                         [ts - tsd for ts, tsd in zip(train_score, train_std)],
                         [ts + tsd for ts, tsd in zip(train_score, train_std)],
                         color=colours[colour_counter], alpha=0.2)
        plt.fill_between(epochs, 
                         [vs - vsd for vs, vsd in zip(valid_score, valid_std)],
                         [vs + vsd for vs, vsd in zip(valid_score, valid_std)],
                         color=colours[colour_counter], alpha=0.2)
        
        plt.title('Average Training & Validation Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Classification Accuracy')
        plt.ylim([min_y,max_y])
        plt.xlim([1,min_epoch_to_plot])
        plt.legend(loc = 'lower right')
        
        colour_counter += 1
            
    plt.tight_layout()
    plt.savefig(results_dir+'/Average_Learning_Curve.png')
    plt.close()    

min_y = 0.4
max_y = 1.0

results_dict,min_epoch_to_plot = results_full(architectures, total_runs, min_y, max_y)
conduct_t_test(results_dict,'Validation')

plot_average_learning_curve_single(architectures,results_dict,min_y,max_y,min_epoch_to_plot)
  