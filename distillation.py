import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm


class KnowledgeDistillation:
    def __init__(
            self,
            teacher_model,
            student_model,
            relevant_class_idxs,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            temperature=3.0,
            alpha=0.7,  # Weight for distillation loss
            lr=1e-3
    ):
        """
        Args:
            teacher_model: Pretrained model with more classes
            student_model: Smaller model to train (fewer output classes)
            relevant_class_idxs: List/tensor of indices for relevant classes in teacher
            device: 'cuda' or 'cpu'
            temperature: Temperature for softmax (higher = softer probabilities)
            alpha: Weight for distillation loss (1-alpha for hard label loss)
            lr: Learning rate
        """
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.relevant_idxs = torch.tensor(relevant_class_idxs).to(device)
        self.device = device
        self.temperature = temperature
        self.alpha = alpha

        self.optimizer = Adam(self.student.parameters(), lr=lr)
        self.ce_loss = nn.CrossEntropyLoss()

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(self, student_logits, teacher_logits):
        """Compute KL divergence between student and teacher soft predictions"""
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        # KL divergence
        kd_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return kd_loss

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.student.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Teacher predictions (no gradients)
            with torch.no_grad():
                teacher_logits = self.teacher(images)
                # Extract only relevant class logits
                teacher_logits = teacher_logits[:, self.relevant_idxs]

            # Student predictions
            student_logits = self.student(images)

            # Combined loss
            hard_loss = self.ce_loss(student_logits, labels)
            soft_loss = self.distillation_loss(student_logits, teacher_logits)
            loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        return total_loss / len(dataloader), 100. * correct / total

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate student model"""
        self.student.eval()
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.student(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        return accuracy

    def train(self, train_loader, val_loader=None, epochs=10):
        """Full training loop"""
        best_acc = 0
        best_epoch = 0

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            train_loss, train_acc = self.train_epoch(train_loader)
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

            if val_loader:
                val_acc = self.evaluate(val_loader)
                print(f'Val Acc: {val_acc:.2f}%')

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    torch.save(self.student.state_dict(), 'best_student.pth')
                    print(f'Saved best model (acc: {best_acc:.2f}%)')

        return self.student, best_acc, best_epoch
