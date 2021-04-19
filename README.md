# Fellowshipai-Challenge-1: Fine-grained Classification using the Cars Dataset

This notebook is a submission to a challenge for Fellowship.ai's program. 

# Packages Used

- torch==1.7.1
- pandas==1.0.5
- numpy==1.18.5
- torchvision==0.8.2
- tqdm==4.47.0
- matplotlib==3.2.2
- ray==1.2.0
- scipy==1.5.0
- nevergrad==0.4.3.post3
- Pillow==8.2.0
- protobuf==3.15.8
- scikit_learn==0.24.1

# The Challenge

The challenge was to train a pretrained ResNet-34 on the Stanford Cars Dataset and use GradCAM to visualize a mislabelled image of the worst class, then
improve the model without increasing the number of epochs. The notebook contains my analysis and training for all of the models.

## The Approach

1) Get a baseline of ResNet's performance
2) Identified that it could be improved through regularization and changing how images were cropped
3) Implemented GradCAM to understand how model was "seeing" the images and modified the crop of the images
4) Regularized the model with two dropout layers and a batch norm layer
5) Utilized Ray Tune to tune the hyperparameters

# Results
I was able to improve the model from 59% accuracy with the base model to 87% accuracy on the test set using my tuned model. 

![alt text](https://github.com/parkerashlan/Fellowshipai-Challenge-1/blob/master/metrics_graphs/val_acc.png)

