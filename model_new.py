import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from x_transformers import Encoder, Decoder
import copy
import os
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
import sys
from tqdm import tqdm

# Custom progress bar to display metrics nicely
class LitProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.enable = True
        
    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        
        # Keep only essential metrics for the progress bar
        essential_metrics = ["train_loss", "val_loss"]
        for key in list(items.keys()):
            if key not in essential_metrics and not key.startswith("epoch"):
                items.pop(key, None)
                
        return items
        
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description("Training")
        return bar
        
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("Validating")
        return bar

'''
PatchEmbed class, adapted from https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
- This class is used to convert the image into patches using a convolutional layer
'''
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=64):
        super().__init__()
        if isinstance(img_size, int):
            img_size = img_size, img_size
        if isinstance(patch_size, int):
            patch_size = patch_size, patch_size
        #calculate the number of patches
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        #convolutional layer to convert the image into patches
        self.conv = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        

    def forward(self, x):
        x = self.conv(x)
        #flatten the patches
        x = rearrange(x, 'b e h w -> b (h w) e')
        return x

'''Lightweight Predictor Module using VIT to predict target patches from context patches'''
class Predictor(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        
        self.predictor = Decoder(dim = embed_dim, depth = depth, heads = num_heads)
    def forward(self, context_encoding, target_masks):
        x = torch.cat((context_encoding, target_masks), dim = 1)
        x = self.predictor(x)
        #return last len(target_masks) tokens
        l = x.shape[1]
        return x[:, l - target_masks.shape[1]:, :]
    
'''Main Model Class'''
class IJEPA_base(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, enc_depth, pred_depth, num_heads, post_emb_norm=False, M = 4, mode="train", layer_dropout=0.):
        super().__init__()
        self.M = M
        self.mode = mode
        self.layer_dropout = layer_dropout

        #define the patch embedding and positional embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_dim  = (self.patch_embed.patch_shape[0], self.patch_embed.patch_shape[1])
        self.num_tokens = self.patch_embed.patch_shape[0] * self.patch_embed.patch_shape[1]
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        #define the cls and mask tokens
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, 0.02)

        #define the encoder and decoder, as well as the layer normalization and dropout
        self.post_emb_norm = nn.LayerNorm(embed_dim) if post_emb_norm else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.teacher_encoder = Encoder(
            dim=embed_dim,
            heads=num_heads,
            depth=enc_depth, 
            layer_dropout=self.layer_dropout,
        )  
        self.student_encoder = copy.deepcopy(self.teacher_encoder).cuda()
        self.predictor = Predictor(embed_dim, num_heads, pred_depth)

    @torch.no_grad() 
    def get_target_block(self, target_encoder, x, patch_dim, aspect_ratio, scale, M):  
        #get the target block
        target_encoder = target_encoder.eval()
        x = target_encoder(x)
        x = self.norm(x)
        #get the patch dimensions
        patch_h, patch_w = patch_dim
        #get the number of patches
        num_patches = patch_h * patch_w
        #get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale)
        #get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        #get the patches in the target block
        target_block = torch.zeros((M, x.shape[0], block_h*block_w, x.shape[2]))
        target_patches = []
        all_patches = []
        for z in range(M):
            #get the starting patch
            start_patch_h = torch.randint(0, patch_h - block_h+1, (1,)).item()
            start_patch_w = torch.randint(0, patch_w - block_w+1, (1,)).item()
            start_patch = start_patch_h * patch_w + start_patch_w

            patches = []
            #get the patches in the target block
            for i in range(block_h):
                for j in range(block_w):
                    patches.append(start_patch + i * patch_w + j)
                    if start_patch + i * patch_w + j not in all_patches:
                        all_patches.append(start_patch + i * patch_w + j)
                    
            #get the target block
            target_patches.append(patches)
            target_block[z] = x[:, patches, :]
        return target_block.cuda(), target_patches, all_patches

    def get_context_block(self, x, patch_dim, aspect_ratio, scale, target_patches):
        patch_h, patch_w = patch_dim
        #get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale)
        #get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        #get the starting patch
        start_patch_h = torch.randint(0, patch_h - block_h+1, (1,)).item()
        start_patch_w = torch.randint(0, patch_w - block_w+1, (1,)).item()
        start_patch = start_patch_h * patch_w + start_patch_w
        #get the patches in the context_block
        patches = []
        for i in range(block_h):
            for j in range(block_w):
                if start_patch + i * patch_w + j not in target_patches: #remove the target patches
                    patches.append(start_patch + i * patch_w + j)
        return x[:, patches, :]


    def forward(self, x, target_aspect_ratio=1, target_scale=1, context_aspect_ratio=1, context_scale=1):
        #get the patch embeddings
        x = self.patch_embed(x)
        b, n, e = x.shape
        #add the positional embeddings
        x = x + self.pos_embedding
        #normalize the embeddings
        x = self.post_emb_norm(x)
        #if mode is test, we get return full embedding:
        if self.mode == 'test':
            return self.student_encoder(x)
        # #get target embeddings
        target_blocks, target_patches, all_patches = self.get_target_block(self.teacher_encoder, x, self.patch_dim, target_aspect_ratio, target_scale, self.M)
        m, b, n, e = target_blocks.shape
        #get context embedding

        context_block = self.get_context_block(x, self.patch_dim, context_aspect_ratio, context_scale, all_patches)
        context_encoding = self.student_encoder(context_block)
        context_encoding = self.norm(context_encoding)


        prediction_blocks = torch.zeros((m, b, n, e)).cuda()
        #get the prediction blocks, predict each target block separately
        for i in range(m):
            target_masks = self.mask_token.repeat(b, n, 1)
            target_pos_embedding = self.pos_embedding[:, target_patches[i], :]
            target_masks = target_masks + target_pos_embedding
            prediction_blocks[i] = self.predictor(context_encoding, target_masks)

        return prediction_blocks, target_blocks


'''TinyImageNet Dataset'''
class TinyImageNetDataset(Dataset):
    def __init__(self, dataset_path, stage='train', img_size=64, transform=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.stage = stage
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load images based on the stage
        self.images = []
        self.labels = []
        
        if stage == 'train':
            # Training set is organized in class folders
            train_dir = os.path.join(dataset_path, 'train')
            class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            
            for class_idx, class_dir in enumerate(sorted(class_dirs)):
                class_path = os.path.join(train_dir, class_dir)
                images_path = os.path.join(class_path, 'images')
                
                # Check if images are in an 'images' subfolder
                if os.path.exists(images_path):
                    image_files = [f for f in os.listdir(images_path) if f.endswith(('.JPEG', '.jpeg', '.jpg', '.png'))]
                    for img_file in image_files:
                        self.images.append(os.path.join(images_path, img_file))
                        self.labels.append(class_idx)
                else:
                    # If no 'images' subfolder, look for images directly in the class folder
                    image_files = [f for f in os.listdir(class_path) if f.endswith(('.JPEG', '.jpeg', '.jpg', '.png'))]
                    for img_file in image_files:
                        self.images.append(os.path.join(class_path, img_file))
                        self.labels.append(class_idx)
        
        elif stage == 'val':
            # Validation set structure
            val_dir = os.path.join(dataset_path, 'val')
            
            # Check if val_dir has subfolders or direct images
            if os.path.exists(os.path.join(val_dir, 'images')):
                # If validation images are in an 'images' subfolder
                images_dir = os.path.join(val_dir, 'images')
                image_files = [f for f in os.listdir(images_dir) if f.endswith(('.JPEG', '.jpeg', '.jpg', '.png'))]
                
                # Try to load annotations if available
                try:
                    annotations_file = os.path.join(val_dir, 'val_annotations.txt')
                    with open(annotations_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                img_file, class_id = parts[0], parts[1]
                                if img_file in image_files:
                                    self.images.append(os.path.join(images_dir, img_file))
                                    self.labels.append(class_id)  # Using class_id as label
                except:
                    # If no annotations, just load the images without labels
                    for img_file in image_files:
                        self.images.append(os.path.join(images_dir, img_file))
                        self.labels.append(-1)  # Unknown label
            else:
                # If validation set has class subfolders like the training set
                class_dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
                
                for class_idx, class_dir in enumerate(sorted(class_dirs)):
                    class_path = os.path.join(val_dir, class_dir)
                    image_files = [f for f in os.listdir(class_path) if f.endswith(('.JPEG', '.jpeg', '.jpg', '.png'))]
                    for img_file in image_files:
                        self.images.append(os.path.join(class_path, img_file))
                        self.labels.append(class_idx)
        
        elif stage == 'test':
            # Test set structure
            test_dir = os.path.join(dataset_path, 'test')
            
            if os.path.exists(test_dir):
                # Check if test_dir has subfolders or direct images
                if os.path.exists(os.path.join(test_dir, 'images')):
                    images_dir = os.path.join(test_dir, 'images')
                    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.JPEG', '.jpeg', '.jpg', '.png'))]
                    for img_file in image_files:
                        self.images.append(os.path.join(images_dir, img_file))
                        self.labels.append(-1)  # Unknown label
                else:
                    # If test set has class subfolders or direct images
                    items = os.listdir(test_dir)
                    if any(os.path.isdir(os.path.join(test_dir, item)) for item in items):
                        # Has subfolders
                        class_dirs = [d for d in items if os.path.isdir(os.path.join(test_dir, d))]
                        for class_idx, class_dir in enumerate(sorted(class_dirs)):
                            class_path = os.path.join(test_dir, class_dir)
                            image_files = [f for f in os.listdir(class_path) if f.endswith(('.JPEG', '.jpeg', '.jpg', '.png'))]
                            for img_file in image_files:
                                self.images.append(os.path.join(class_path, img_file))
                                self.labels.append(class_idx)
                    else:
                        # Direct images
                        image_files = [f for f in items if f.endswith(('.JPEG', '.jpeg', '.jpg', '.png'))]
                        for img_file in image_files:
                            self.images.append(os.path.join(test_dir, img_file))
                            self.labels.append(-1)  # Unknown label
        
        print(f"Loaded {len(self.images)} images for {stage} stage")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image in case of error
            return torch.zeros((3, self.img_size, self.img_size))


'''PyTorch Lightning DataModule for TinyImageNet'''
class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 batch_size=16,
                 num_workers=4,
                 pin_memory=True,
                 shuffle=True,
                 img_size=64,
                 val_split=0.1,  # Add validation split parameter
                 data_fraction=1.0  # Add parameter to control fraction of data to use
                 ):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.img_size = img_size
        self.val_split = val_split  # Store validation split ratio
        self.data_fraction = data_fraction  # Store data fraction
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def setup(self, stage=None):
        # Load training dataset
        full_train_dataset = TinyImageNetDataset(
            dataset_path=self.dataset_path, 
            stage='train',
            img_size=self.img_size,
            transform=self.train_transform
        )
        
        # Try to load validation dataset
        val_dataset = TinyImageNetDataset(
            dataset_path=self.dataset_path, 
            stage='val',
            img_size=self.img_size,
            transform=self.val_transform
        )
        
        # If validation dataset is empty, create one from training data
        if len(val_dataset) == 0:
            print("No validation data found. Creating validation set from training data...")
            
            # Calculate split sizes
            val_size = int(len(full_train_dataset) * self.val_split)
            train_size = len(full_train_dataset) - val_size
            
            # Get indices for train and validation
            indices = list(range(len(full_train_dataset)))
            import random
            random.seed(42)  # For reproducibility
            random.shuffle(indices)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            # Create separate training and validation datasets
            self.train_dataset = TinyImageNetDataset(
                dataset_path=self.dataset_path,
                stage='train',
                img_size=self.img_size,
                transform=self.train_transform
            )
            
            self.val_dataset = TinyImageNetDataset(
                dataset_path=self.dataset_path,
                stage='train',  # Using train data for validation
                img_size=self.img_size,
                transform=self.val_transform
            )
            
            # Update the datasets with only their respective images
            self.train_dataset.images = [full_train_dataset.images[i] for i in train_indices]
            self.train_dataset.labels = [full_train_dataset.labels[i] for i in train_indices]
            
            self.val_dataset.images = [full_train_dataset.images[i] for i in val_indices]
            self.val_dataset.labels = [full_train_dataset.labels[i] for i in val_indices]
            
            print(f"Created validation set with {len(self.val_dataset)} images from training data")
            print(f"Remaining training set has {len(self.train_dataset)} images")
        else:
            # Use the existing validation set
            self.train_dataset = full_train_dataset
            self.val_dataset = val_dataset
            
        # Apply data fraction to reduce dataset size if needed
        if self.data_fraction < 1.0:
            # Reduce training dataset size
            train_size = int(len(self.train_dataset.images) * self.data_fraction)
            import random
            random.seed(42)  # For reproducibility
            indices = random.sample(range(len(self.train_dataset.images)), train_size)
            self.train_dataset.images = [self.train_dataset.images[i] for i in indices]
            self.train_dataset.labels = [self.train_dataset.labels[i] for i in indices]
            
            # Reduce validation dataset size
            val_size = int(len(self.val_dataset.images) * self.data_fraction)
            indices = random.sample(range(len(self.val_dataset.images)), val_size)
            self.val_dataset.images = [self.val_dataset.images[i] for i in indices]
            self.val_dataset.labels = [self.val_dataset.labels[i] for i in indices]
            
            print(f"Using {self.data_fraction:.1%} of the data:")
            print(f"  - Training set: {len(self.train_dataset.images)} images")
            print(f"  - Validation set: {len(self.val_dataset.images)} images")
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True,  # Add this to fix the warning
        )


'''PyTorch Lightning Model'''
class IJEPA(pl.LightningModule):
    def __init__(
            self,
            img_size=64,
            patch_size=8,
            in_chans=3, 
            embed_dim=64,
            enc_heads=8,
            enc_depth=8,
            decoder_depth=6,
            lr=1e-4,
            weight_decay=0.05,
            target_aspect_ratio = (0.75,1.5),
            target_scale = (0.15, .2),
            context_aspect_ratio = 1,
            context_scale = (0.85,1.0),
            M = 4, #number of different target blocks
            m=0.996, #momentum
            m_start_end = (.996, 1.)
    ):
        super().__init__()
        self.save_hyperparameters()
        
        #define models
        self.model = IJEPA_base(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, 
                                enc_depth=enc_depth, num_heads=enc_heads, pred_depth=decoder_depth, M=M)

        #define hyperparameters
        self.M = M
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = m
        self.target_aspect_ratio = target_aspect_ratio
        self.target_scale = target_scale
        self.context_aspect_ratio = context_aspect_ratio
        self.context_scale = context_scale
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_tokens = (img_size // patch_size) ** 2
        self.m_start_end = m_start_end

        #define loss
        self.criterion = nn.MSELoss()
        
        # For tracking and printing losses
        self.last_train_loss = 0.0
        self.last_val_loss = 0.0
        self.print_newline = True
        self.total_train_steps = 0
        self.total_val_steps = 0
        self.current_train_step = 0
        self.current_val_step = 0
    
    def on_train_start(self):
        # Print header for the loss display
        print("\nEpoch  Step/Total    Train Loss    Val Loss")
        print("-----  ----------  ----------  ----------")
        
    def on_train_epoch_start(self):
        # Calculate total steps for this epoch
        try:
            # Try to get the dataloader length from the trainer's datamodule
            if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                train_dataloader = self.trainer.datamodule.train_dataloader()
                self.total_train_steps = len(train_dataloader)
            elif hasattr(self.trainer, 'train_dataloader') and self.trainer.train_dataloader is not None:
                self.total_train_steps = len(self.trainer.train_dataloader)
            else:
                self.total_train_steps = 0
                
            self.current_train_step = 0
        except (TypeError, AttributeError) as e:
            # Fallback if we can't access the dataloader
            self.total_train_steps = 0
            print(f"Warning: Could not determine total training steps: {e}")
            
    def on_validation_epoch_start(self):
        # Calculate total steps for validation
        try:
            # Try to get the dataloader length from the trainer's datamodule
            if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                val_dataloader = self.trainer.datamodule.val_dataloader()
                self.total_val_steps = len(val_dataloader)
            elif hasattr(self.trainer, 'val_dataloaders') and self.trainer.val_dataloaders is not None:
                self.total_val_steps = len(self.trainer.val_dataloaders)
            else:
                self.total_val_steps = 0
                
            self.current_val_step = 0
        except (TypeError, AttributeError) as e:
            # Fallback if we can't access the dataloader
            self.total_val_steps = 0
            print(f"Warning: Could not determine total validation steps: {e}")
        
    def print_loss_info(self):
        # Print the loss information in a fixed location
        epoch = self.current_epoch
        
        # Format step as current/total
        if self.trainer.validating:
            if self.total_val_steps > 0:
                step_info = f"{self.current_val_step}/{self.total_val_steps}"
            else:
                step_info = f"{self.current_val_step}/?"
        else:
            if self.total_train_steps > 0:
                step_info = f"{self.current_train_step}/{self.total_train_steps}"
            else:
                step_info = f"{self.current_train_step}/?"
        
        # Create the formatted string with current losses - show only 4 digits
        info_str = f"\r{epoch:5d}  {step_info:>10}  {self.last_train_loss:10.4f}  {self.last_val_loss:10.4f}"
        
        # Print the string and flush to ensure it updates in place
        sys.stdout.write(info_str)
        sys.stdout.flush()
    
    def forward(self, x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale):
        return self.model(x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)
    
    '''Update momentum for teacher encoder'''
    def update_momentum(self, m):
        student_model = self.model.student_encoder.eval()
        teacher_model = self.model.teacher_encoder.eval()
        with torch.no_grad():
            for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
                teacher_param.data.mul_(other=m).add_(other=student_param.data, alpha=1 - m)

    def training_step(self, batch, batch_idx):
        x = batch
        # Increment the current step counter
        self.current_train_step = batch_idx + 1
        
        #generate random target and context aspect ratio and scale
        target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
        target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
        context_aspect_ratio = self.context_aspect_ratio
        context_scale = np.random.uniform(self.context_scale[0], self.context_scale[1])

        y_student, y_teacher = self(x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)
        loss = self.criterion(y_student, y_teacher)
        
        # Log step-level loss without progress bar
        self.log('train_loss_step', loss, on_step=True, on_epoch=False)
        
        # Log epoch-level loss with progress bar
        self.log('train_loss', loss, prog_bar=False, on_step=False, on_epoch=True)
        
        # Update the last training loss and print the info
        self.last_train_loss = loss.item()
        self.print_loss_info()
                    
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        # Increment the current validation step counter
        self.current_val_step = batch_idx + 1
        
        target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
        target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
        context_aspect_ratio = self.context_aspect_ratio
        context_scale = np.random.uniform(self.context_scale[0], self.context_scale[1])

        y_student, y_teacher = self(x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)
        loss = self.criterion(y_student, y_teacher)
        
        # Log step-level loss without progress bar
        self.log('val_loss_step', loss, on_step=True, on_epoch=False)
        
        # Log epoch-level loss with progress bar
        self.log('val_loss', loss, prog_bar=False, on_step=False, on_epoch=True)
        
        # Update the last validation loss and print the info
        self.last_val_loss = loss.item()
        self.print_loss_info()
        
        return loss
        
    def on_validation_epoch_end(self):
        # Add a newline after validation to separate epochs
        print()
        
    def on_train_epoch_end(self):
        # The train_loss_epoch will be automatically logged to the progress bar
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx):
        target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
        target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
        context_aspect_ratio = self.context_aspect_ratio
        context_scale = 1
        self.model.mode = "test"

        return self(batch, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale) #just get teacher embedding

    def on_after_backward(self):
        self.update_momentum(self.m)
        self.m += (self.m_start_end[1] - self.m_start_end[0]) / self.trainer.estimated_stepping_batches

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


if __name__ == '__main__':
    # Path to the Tiny ImageNet dataset
    dataset_path = 'tiny-imagenet-200'
    
    # Create the data module
    data_module = TinyImageNetDataModule(
        dataset_path=dataset_path,
        batch_size=32,
        num_workers=4,
        img_size=64
    )
    
    # Create the model
    model = IJEPA(
        img_size=64,  # TinyImageNet images are 64x64
        patch_size=8,  # Smaller patch size for 64x64 images
        in_chans=3,
        embed_dim=192,
        enc_heads=8,
        enc_depth=12,
        decoder_depth=4,
        lr=1e-3,
        M=4
    )
    
    # Define callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="ijepa-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    
    # Create the trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=16,
        max_epochs=50,
        callbacks=[lr_monitor, model_summary, checkpoint_callback],
        gradient_clip_val=0.1,
    )
    
    # Train the model
    trainer.fit(model, data_module) 