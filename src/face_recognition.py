# -*- coding: utf-8 -*-
"""
Created on Mon Jan 8 12:11:01 2025

@author: Qandeel Fazal"""

#importing dependencies
import torch
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np
from PIL import Image
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity 
from torchvision import transforms

"""

Face Recognition class with all the functions

"""

class FaceRecognitionSystem:
    def __init__(self, model_path='inception_resnet_v1_vggface2.pth', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize face recognition system with custom pretrained model
        """
        print(f"Using device: {device}")
        self.device = device
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160,
            margin=40,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            post_process=True,
            select_largest = False,
            keep_all= True,
            device=self.device           
        )
        
        # Load custom model              
        model_path= 'inception_resnet_v1_vggface2.pth'
        self.model = self.load_custom_model(model_path)
        self.known_faces = {}
    
    def load_custom_model(self, model_path):
        """
        Load custom pretrained model from .pth file        
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        # Load state dict
        try:
            
            # Load the pretrained InceptionResnetV1 model 
            model = InceptionResnetV1(pretrained = None, classify = False) #pretrained = None loads model from .pth file 
            state_dict = torch.load(model_path, map_location = torch.device('cpu'))
            model.load_state_dict(state_dict)
            print(f"Loaded{model_path} from local file")
            print("Model loaded successfully through strict approach")            
                       
        except Exception as e:
            print(f" Error loading modelthrough strict approach: {e}")
            print(" Trying alternative loading approach...")
            
            # Filter out unexpected keys like the final classification layer of InceptionNet
            #Create new state_dict without keys causing mismatch
            
            model_dict= model.state()
            petrained_dict = {k:v for k,v in state_dict.items() if k in model_dict and state_dict[k].shape == model_dict[k].shape}
            #find unexpected keys
            unexpected_keys=[k for k in state_dict.keys()if k not in model_dict]
            if unexpected_keys:
                print(f"unexpected keys:{unexpected_keys}")
            #Randomly initialize keys that are not in pretrained model
            missing_keys = [k for k in model_dict.keys() if k not in state_dict]
            if missing_keys:
                print(f"missing keys:{missing_keys}")
            #Clean keys and remove DataParallel module i.e. 'module.' prefix
            clean_keys={k.replace('module.', ''):v for k,v in petrained_dict.items()}
            model_keys = set(model_dict)
            filtered_dict = {k:v for k,v in clean_keys.items() if k in model_keys}
            dropped_keys = set(clean_keys.keys())-set(filtered_dict.keys())
            print(f"dropped keys: {dropped_keys}")
            
            #Now try alternative loading method
            try:
                #load weights with strict=False
                load_state = model.load_state_dict(filtered_dict, strict=False)
                print("Model loaded with strict=False")
            except:
                print("Could not load model. Using random weights.")
                
        model.eval()
        return model              
    
    def take_face_image(self, vid, path):        
        """
        Capture frames from video stream to generate images for training
        """
        #clear the old items of directory
        try:
            for item in os.listdir(path):
                if not os.path.isdir(path):
                    break;                
                item_path = os.path.join(path, item)
                if item_path:
                    os.remove(item_path)
        except Exception as e:
            print(f" Error clearing old images database: {e}")
            
        cam = cv2.VideoCapture(vid)#'rtsp://admin:abcd1234@192.168.1.64/1')
  
        cam_width=int(cam.get(3))
        cam_height=int(cam.get(4))
        size= (cam_width,cam_height)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            if ret==True:
                #Convert to PIL for MTCNN
                pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                #Detect faces
                boxes, probs, landmarks = self.detect_faces(pil_image)
                if boxes is None:
                    print("No faces detected")
                    return img

                print(f" Detected {len(boxes)} face(s)")

                face_img= self.face_crop(pil_image, boxes)
                face_img = np.array(face_img)
                face_img = (cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(path+ '/person' + '-' + str(sampleNum) + '.jpg', face_img)
            
            sampleNum = sampleNum + 1
            # display the frame
            #cv2.imshow('Taking Images',img)                           
            # wait for 100 miliseconds
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif sampleNum > 300:
                break    # break if the sample number is more than 300      
        cam.release()
        cv2.destroyAllWindows()
        
    def train_from_directory(self,dir_path):
        """
        Train and get face embeddings of known faces
        """
        print("Training on database images")
        for image_name in os.listdir(dir_path):
            if os.path.isdir(dir_path):
                embeddings=[]             
           
                if image_name.lower().endswith(('.png','.jpg', '.jpeg')):
                    img_path= os.path.join(dir_path, image_name)
                    image =cv2.imread(img_path)
                    
                    if image is not None:
                        pil_image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        embedding = self.get_face_embedding(pil_image, image_name)
                        
                        if embedding is not None:
                            embeddings.append(embedding)
            if embeddings:
                avg_embedding=np.mean(embeddings, axis =0)
                name='person 1'
                self.known_faces[name] = avg_embedding
                print(f"added {name} with {len(embeddings)} samples")                    
    
    def detect_faces(self, image):
        """
        Detect faces using MTCNN
        """
        print("Detecting faces...")
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detect faces
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        return boxes, probs, landmarks
    
    def face_crop(self, pil_image, boxes):
        """
        Getting cropped face image for embedding generation
        """
        if boxes is None:
            print("No faces detected")
            return
        
        print(f"✅ Detected {len(boxes)} face(s)")
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            print(f"i= {i+1}")
            # Extract face region
            face_img_cropped = pil_image.crop((x1, y1, x2, y2))
            #face_img_cropped.show("detected face")
        return face_img_cropped
    
    def get_face_embedding(self, face_image, image_name):
        """
        Get face embedding from pretrained model
        """
        print(f"Getting face embeddings for {image_name}")
        try:
            # Define the preprocessing transforms 
            preprocess = transforms.Compose([ #used if detecting and cropping done
            transforms.Resize(160), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
            ]) 

            face_image=face_image.convert("RGB")
            face_tensor = preprocess(face_image).unsqueeze(0)  # Add batch dimension 
            
            # Preprocess face using MTCNN
            #face_tensor = self.mtcnn(face_image)#used if using mtcnn for face crop
            
            if face_tensor is None:
                print("no embeddings")
                return None
            
            # Add batch dimension and move to device
            #face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            # Get embedding
            with torch.no_grad():#evaluation mode so no gradients
                embedding = self.model(face_tensor) #facenet(face_tensor)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def save_database(self, file_path='face_database.pkl'):
        """
        Save known faces database
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.known_faces, f)
        print(f"Database saved to: {file_path}")
    
    def load_database(self, file_path='face_database.pkl'):
        """
        Load known faces database
        """
        try:
            with open(file_path, 'rb') as f:
                self.known_faces = pickle.load(f)
                                
            print(f" Database loaded from: {file_path}")
            print(f"Loaded {len(self.known_faces)} Known faces")
                  # {list(self.known_faces.keys())}")
        except FileNotFoundError:
            print(" Database file not found")
        except Exception as e:
            print(f" Error loading database: {e}")
    
    def process_video(self, name, vid_path, out_path, threshold):
        cam=cv2.VideoCapture(vid_path)
        #name = 'test person'
        if (not cam.isOpened()):
            print("no video found error")
            return
        
        cam_width=int(cam.get(3))
        cam_height=int(cam.get(4))
        size= (cam_width,cam_height)
        if(out_path):
            out=cv2.VideoWriter(out_path+ name + '.mov',cv2.VideoWriter_fourcc(*'MJPG'),10,size)
        
        while (True):
            ret, image = cam.read()
            if ret==False:
                break;
            
            # Convert to PIL for MTCNN
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Detect faces
            boxes, probs, landmarks = self.detect_faces(pil_image)
            
            if boxes is None:
                print(" No faces detected")
                return image
            
            result_image = image.copy()
            print(f" Detected {len(boxes)} face(s)")
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                print(f"i= {i+1}")
                # Extract face region
                face_img = pil_image.crop((x1, y1, x2, y2))
                #face_img.show("detected face")
                # Get embedding and recognize           
                embedding = self.get_face_embedding(face_img, name)
                if embedding is not None:
                    name, confidence = self.recognize_face(embedding, threshold)
                    print(f"name = {name}")
                    # Draw results
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{name}: {confidence:.3f}"
                    cv2.putText(result_image, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    print(f" Face {i+1}: {name} (confidence: {confidence:.3f})")
                    
            cv2.imshow('Face Recognition',result_image )
            
            if out_path:
                out.write(result_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cam.release()
        if out_path:
            out.release()
        cv2.destroyAllWindows()
    
    def recognize_face(self, face_embedding, threshold):
        """
        Recognize a face by comparing with known faces
        """
        print("Recognizing known face...")
        
        if not self.known_faces:
            return "Unknown", 0.0
        
        best_match = "Unknown"
        best_similarity = 0.0
        
        for name, known_embedding in self.known_faces.items():
            similarity = cosine_similarity(
                known_embedding.reshape(1, -1),
                face_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match = name
                print(f"best_match = {name}")
        
        return best_match, best_similarity           
            
def demonstrate_system():
    """
    Demonstrate the face recognition system
    """
    print(" Starting Face Recognition with Custom Model")
    print("=" * 50)
    print('demonstration')
    
    # Initialize face recognition system
    face_system = FaceRecognitionSystem(model_path='inception_resnet_v1_vggface2.pth')
    # Giving video for forming frames and training images
    training_vid = 'sample_faces/Vid/v1.mov'
    # Give path where training images will be stored
    training_dir = 'sample_faces/Vid'
    # Take face video for frame creation and training
    face_system.take_face_image(training_vid, training_dir)
    # Perform traing on known face frames
    name = 'person 1'
    face_system.train_from_directory(training_dir, name)    
    # Save embeddings of known person to databse file
    face_system.save_database()
    # Load embeddings of known person from databse file
    face_system.load_database()        
    #Process test frames from video
    test_vid = 'sample_faces/v2.mov'
    out_path_recog_vid= 'sample_faces/Vid/'
    threshold = 0.6
    test_name = 'test for person'
    face_system.process_video(test_name,test_vid, out_path_recog_vid, threshold)
           
if __name__ =="__main__":
    demonstrate_system() #demonstration
    
