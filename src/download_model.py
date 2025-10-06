
!pip install facenet-pytorch

import torch
from facenet_pytorch import InceptionResnetV1

# Step 1: Load the model with pretrained weights (VGGFace2)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Step 2: Save the state_dict to a file
torch.save(model.state_dict(), "inception_resnet_v1_vggface2.pth")
print("Model weights saved as inception_resnet_v1_vggface2.pth")

# Step 3: Download the file to your local machine
from google.colab import files
files.download("inception_resnet_v1_vggface2.pth")
