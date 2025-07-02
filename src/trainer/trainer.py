"""Training module for the MvHo-IB model.

Manages the training, validation, and testing process of the multi-view
higher-order information bottleneck model.
"""

import logging
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any

from ..models.gin_model import GINModel
from ..models.brain3dcnn import Brain3DCNN
from ..models.fusion_model import FusionModel
from ..utils.info_bottleneck import renyi_entropy


class Trainer:
    """Multi-view model trainer."""

    def __init__(self, config: Dict, device: torch.device) -> None:
        """Initialize trainer."""
        self.config = config
        self.device = device
        
        # Model configurations
        self.dataset_name = config['dataset_name']
        self.dataset_config = config['datasets'][self.dataset_name]
        self.num_classes = self.dataset_config['num_classes']
        
        # Training configurations
        self.train_config = config['training']
        self.num_epochs = self.train_config['num_epochs']
        self.learning_rate = self.train_config['learning_rate']
        self.weight_decay = self.train_config['weight_decay']
        self.batch_size = self.train_config['batch_size']
        
        # IB configurations
        self.ib_config = config['information_bottleneck']
        self.beta_gin = self.ib_config['beta_gin']
        self.beta_cnn = self.ib_config['beta_cnn']
        self.use_ib = self.ib_config['use_ib']
        
        # Ablation study configurations
        self.ablation_config = config.get('ablation', {})
        self.use_gin = self.ablation_config.get('use_gin', True)
        self.use_cnn = self.ablation_config.get('use_brain3dcnn', True)
        
        # Class names for display
        self.class_names = self._get_class_names()
        
        # Initialize models
        self.gin_model = None
        self.cnn_model = None
        self.fusion_model = None
        self.optimizer = None
        self.criterion = None
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Results storage
        self.results = {}

    def _get_class_names(self) -> List[str]:
        """Get class names based on dataset."""
        if self.dataset_name == "UCLA":
            return ["ASD", "TD"]
        elif self.dataset_name == "EOEC":
            return ["EO", "EC"]
        elif self.dataset_name == "ADNI":
            classification_mode = self.dataset_config.get('classification_mode', 'three_class')
            if classification_mode == "three_class":
                return ["AD", "MCI", "NC"]
            elif classification_mode == "binary_ad_nc":
                return ["AD", "NC"]
            elif classification_mode == "binary_ad_mci":
                return ["AD", "MCI"]
            elif classification_mode == "binary_mci_nc":
                return ["MCI", "NC"]
        return [f"Class_{i}" for i in range(self.num_classes)]

    def setup_models(self, example_graph_batch, example_tensor_batch: torch.Tensor) -> None:
        """Initialize and configure models."""
        # Determine input feature dimensions
        num_graph_features = example_graph_batch.x.shape[1]
        
        # Calculate fusion model input size
        fusion_input_size = 0
        
        # Initialize GIN model
        if self.use_gin:
            self.gin_model = GINModel(
                num_features=num_graph_features,
                embedding_dim=self.train_config['gin_embedding_dim'],
                hidden_dims=self.train_config['gin_hidden_dims'],
                dropout_rate=self.train_config['dropout_rate']
            ).to(self.device)
            fusion_input_size += self.train_config['gin_embedding_dim']
            logging.info(f"GIN model initialized: {self.gin_model.get_num_parameters()} parameters")

        # Initialize CNN model  
        if self.use_cnn:
            self.cnn_model = Brain3DCNN(
                example_tensor=example_tensor_batch,
                embedding_dim=self.train_config['cnn_embedding_dim'],
                channels=tuple(self.train_config['cnn_channels']),
                dropout_rate=self.train_config['dropout_rate']
            ).to(self.device)
            fusion_input_size += self.train_config['cnn_embedding_dim']
            logging.info(f"Brain3DCNN model initialized: {self.cnn_model.get_num_parameters()} parameters")

        # Initialize fusion model
        if fusion_input_size > 0:
            self.fusion_model = FusionModel(
                input_size=fusion_input_size,
                num_classes=self.num_classes,
                dropout_rate=self.train_config['dropout_rate']
            ).to(self.device)
            logging.info(f"Fusion model initialized: {self.fusion_model.get_num_parameters()} parameters")
        else:
            raise ValueError("At least one model (GIN or CNN) must be enabled")

        # Setup optimizer and loss function
        all_params = []
        if self.gin_model:
            all_params.extend(self.gin_model.parameters())
        if self.cnn_model:
            all_params.extend(self.cnn_model.parameters())
        all_params.extend(self.fusion_model.parameters())

        self.optimizer = optim.Adam(
            all_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        logging.info(f"Models setup completed. Total parameters: {self._count_total_parameters()}")

    def _count_total_parameters(self) -> int:
        """Count total number of parameters in all models."""
        total = 0
        if self.gin_model:
            total += self.gin_model.get_num_parameters()
        if self.cnn_model:
            total += self.cnn_model.get_num_parameters()
        if self.fusion_model:
            total += self.fusion_model.get_num_parameters()
        return total

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch."""
        # Set models to training mode
        if self.gin_model:
            self.gin_model.train()
        if self.cnn_model:
            self.cnn_model.train()
        self.fusion_model.train()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        # Progress bar for training batches
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (x1_batch, x2_batch, labels) in enumerate(pbar):
            # Move data to device
            x1_batch = x1_batch.to(self.device)
            x2_batch = x2_batch.unsqueeze(1).to(self.device)  # Add channel dimension: [B, D, H, W] -> [B, 1, D, H, W]
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            gin_embedding = None
            cnn_embedding = None

            if self.gin_model:
                gin_embedding = self.gin_model(x1_batch)

            if self.cnn_model:
                cnn_embedding = self.cnn_model(x2_batch)

            # Fusion
            fused_features, logits = self.fusion_model(gin_embedding, cnn_embedding)

            # Classification loss
            classification_loss = self.criterion(logits, labels)

            # Information bottleneck loss
            ib_loss = 0.0
            if self.use_ib:
                if gin_embedding is not None:
                    gin_ib_loss = renyi_entropy(gin_embedding)
                    ib_loss += self.beta_gin * gin_ib_loss

                if cnn_embedding is not None:
                    cnn_ib_loss = renyi_entropy(cnn_embedding)
                    ib_loss += self.beta_cnn * cnn_ib_loss

            # Total loss
            total_batch_loss = classification_loss + ib_loss

            # Backward pass
            total_batch_loss.backward()
            self.optimizer.step()

            # Collect predictions and labels
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += total_batch_loss.item()

            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            current_acc = accuracy_score(all_labels, all_predictions)
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}'
            })

        # Calculate metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)

        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate one epoch."""
        # Set models to evaluation mode
        if self.gin_model:
            self.gin_model.eval()
        if self.cnn_model:
            self.cnn_model.eval()
        self.fusion_model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        # Progress bar for validation batches
        pbar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for batch_idx, (x1_batch, x2_batch, labels) in enumerate(pbar):
                # Move data to device
                x1_batch = x1_batch.to(self.device)
                x2_batch = x2_batch.unsqueeze(1).to(self.device)  # Add channel dimension: [B, D, H, W] -> [B, 1, D, H, W]
                labels = labels.to(self.device)

                # Forward pass
                gin_embedding = None
                cnn_embedding = None

                if self.gin_model:
                    gin_embedding = self.gin_model(x1_batch)

                if self.cnn_model:
                    cnn_embedding = self.cnn_model(x2_batch)

                # Fusion
                fused_features, logits = self.fusion_model(gin_embedding, cnn_embedding)

                # Loss calculation
                loss = self.criterion(logits, labels)

                # Collect predictions and labels
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                total_loss += loss.item()

                # Update progress bar
                current_loss = total_loss / (batch_idx + 1)
                current_acc = accuracy_score(all_labels, all_predictions)
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })

        # Calculate metrics
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)

        return epoch_loss, epoch_acc

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Complete training process."""
        logging.info("=" * 50)
        logging.info("TRAINING STARTED")
        logging.info("=" * 50)
        
        best_val_acc = 0.0
        best_epoch = 0

        # Main training loop with progress bar
        epoch_pbar = tqdm(range(self.num_epochs), desc="Training Progress")
        
        for epoch in epoch_pbar:
            start_time = time.time()

            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Record training history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                self.save_model()

            epoch_time = time.time() - start_time

            # Update progress bar with epoch information
            epoch_pbar.set_postfix({
                'Train_Acc': f'{train_acc:.4f}',
                'Val_Acc': f'{val_acc:.4f}',
                'Best_Val': f'{best_val_acc:.4f}',
                'Time': f'{epoch_time:.1f}s'
            })

            # Log epoch information
            logging.info(
                f"Epoch [{epoch+1}/{self.num_epochs}] - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )

        logging.info("=" * 50)
        logging.info("TRAINING COMPLETED")
        logging.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")
        logging.info("=" * 50)

    def test_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Test the model and return detailed metrics."""
        logging.info("Testing model...")
        
        # Load best model
        self.load_model()
        
        # Set models to evaluation mode
        if self.gin_model:
            self.gin_model.eval()
        if self.cnn_model:
            self.cnn_model.eval()
        self.fusion_model.eval()

        all_predictions = []
        all_labels = []

        # Progress bar for test batches
        pbar = tqdm(test_loader, desc="Testing")

        with torch.no_grad():
            for x1_batch, x2_batch, labels in pbar:
                # Move data to device
                x1_batch = x1_batch.to(self.device)
                x2_batch = x2_batch.unsqueeze(1).to(self.device)  # Add channel dimension: [B, D, H, W] -> [B, 1, D, H, W]

                # Forward pass
                gin_embedding = None
                cnn_embedding = None

                if self.gin_model:
                    gin_embedding = self.gin_model(x1_batch)

                if self.cnn_model:
                    cnn_embedding = self.cnn_model(x2_batch)

                # Fusion
                fused_features, logits = self.fusion_model(gin_embedding, cnn_embedding)

                # Collect predictions
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update progress bar
                current_acc = accuracy_score(all_labels, all_predictions)
                pbar.set_postfix({'Acc': f'{current_acc:.4f}'})

        # Calculate detailed metrics
        test_accuracy = accuracy_score(all_labels, all_predictions)
        
        # Handle different averaging strategies based on problem type
        average_strategy = 'binary' if self.num_classes == 2 else 'macro'
        
        test_precision = precision_score(all_labels, all_predictions, average=average_strategy, zero_division=0)
        test_recall = recall_score(all_labels, all_predictions, average=average_strategy, zero_division=0)
        test_f1 = f1_score(all_labels, all_predictions, average=average_strategy, zero_division=0)

        test_results = {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1
        }

        # Log results
        logging.info("=" * 50)
        logging.info("TEST RESULTS")
        logging.info("=" * 50)
        for metric, value in test_results.items():
            logging.info(f"{metric.capitalize()}: {value:.4f}")
        logging.info("=" * 50)

        return test_results

    def save_model(self) -> None:
        """Save model state."""
        model_state = {}
        
        if self.gin_model:
            model_state['gin_model'] = self.gin_model.state_dict()
        if self.cnn_model:
            model_state['cnn_model'] = self.cnn_model.state_dict()
        
        model_state['fusion_model'] = self.fusion_model.state_dict()
        model_state['optimizer'] = self.optimizer.state_dict()
        
        save_path = 'best_model.pth'
        torch.save(model_state, save_path)

    def load_model(self) -> None:
        """Load model state."""
        load_path = 'best_model.pth'
        if not os.path.exists(load_path):
            logging.warning(f"Model file {load_path} not found")
            return
            
        model_state = torch.load(load_path, map_location=self.device)
        
        if self.gin_model and 'gin_model' in model_state:
            self.gin_model.load_state_dict(model_state['gin_model'])
        if self.cnn_model and 'cnn_model' in model_state:
            self.cnn_model.load_state_dict(model_state['cnn_model'])
            
        self.fusion_model.load_state_dict(model_state['fusion_model'])
        
        logging.info("Model loaded successfully")

    def save_results(self, test_results: Dict[str, float]) -> None:
        """Save training and test results."""
        # Prepare results data
        self.results = {
            'test_results': test_results,
            'training_history': {
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies
            },
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        # Save to file
        import json
        results_path = 'results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logging.info(f"Results saved to {results_path}")

    def get_results(self) -> Dict[str, Any]:
        """Get training and test results."""
        return self.results 