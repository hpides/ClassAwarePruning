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
            selected_classes=None,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            temperature=3.0,
            alpha=0.7,
            lr=1e-3
    ):
        """
        Basic class for knowledge distillation used for retraining in a class-aware setting.

        Args:
            teacher_model (nn.Module): Teacher - intended to be an unstructured pruned model.
            student_model (nn.Module): Student - intended to be a structured pruned model.
            selected_classes ([int]): List of target classes that both models have been pruned for
            device (str): Device to run on.
            temperature (float): Temperature for softmax (higher = softer probabilities)
            alpha (float): Weight for distillation loss (1-alpha for hard label loss)
            lr (float): Learning rate for training
        """
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.selected_classes = torch.tensor(selected_classes).to(device) if selected_classes else None
        self.optimizer = Adam(self.student.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(self, student_logits, teacher_logits):
        """
        Computes the divergence between student and teacher soft predictions.

        Args:
            teacher_logits (torch.Tensor): Raw outputs from the teacher of shape (batch_size, num_classes).
            student_logits (torch.Tensor): Raw outputs from the student of shape (batch_size, num_classes).
        
        Returns: 
            kd_loss (torch.Tensor): The scalar loss tensor.
        """
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
        """
        Trains for one epoch.

        Args:
            dataloader (DataLoader): Iterable for yielding the batches of images and labels for training.

        Returns:
            Tuple: Average loss per batch and accuracy.
        """
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

            # Student predictions
            student_logits = self.student(images)

            # Combined loss
            hard_loss = self.loss(student_logits, labels)
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
    def evaluate(self, dataloader, print_results=False, all_classes=False):
        """
        Evaluation loop for student model.

        Args:
            dataloader (DataLoader): Iterable for yielding the batches of images and labels for evaluation.
            print_results (boolean): Print results to consol.
            all_classes (boolean): TODO: what does all_classes do again?

        Returns:
            Tuple: Overall accuracy and accuracies per class
        """
        self.student.eval()
        correct = 0
        total = 0

        num_classes = len(self.selected_classes) if self.selected_classes is not None else 10
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.student(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Track per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        accuracy = 100. * correct / total

        if print_results:
            print(f"Accuracy of the model on the test set: {accuracy:.2f}%")

        class_accuracies = {}
        if all_classes and print_results:
            for i in range(num_classes):
                if class_total[i] > 0:
                    accuracy_i = 100 * class_correct[i] / class_total[i]
                    class_accuracies[i] = accuracy_i
                    print(f"%%%%%% Accuracy of class {i}: {accuracy_i:.2f}%")

        return accuracy, class_accuracies

    def train(self, train_loader, val_loader=None, epochs=10, patience=5, min_delta=0.1, log_results=False):
        """
        Full training loop with early stopping.

        Args:
            train_loader (DataLoader): Iterable for yielding the batches of images and labels for training.
            val_loader (DataLoader): Iterable for yielding the batches of images and labels for validation.
            epochs (int): Number of epochs to train.
            patience (int): Number of consecutive epochs without improvement in validation until early stopping.
            min_delta (float): How large improvements need to be to count as such.
            log_results (boolean): Whether to log results to WandB.

        Returns:
            Tuple: The trained student model, the best accuracy during validation and the best epoch it occurred.
        """
        best_accuracy = 0.0
        best_epoch = 1
        epochs_without_improvement = 0

        print("=" * 60)
        print(f"Starting knowledge distillation for {epochs} epochs")
        print(f"Train batches per epoch: {len(train_loader)}")
        print(f"Train samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Val samples: {len(val_loader.dataset)}")
        print("=" * 60)

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            train_loss, train_acc = self.train_epoch(train_loader)
            print(f'***** Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}', end='')

            if val_loader:
                val_acc, _ = self.evaluate(val_loader, print_results=False, all_classes=False)
                print(f', Val Accuracy: {val_acc:.2f}')

                if log_results:
                    try:
                        import wandb
                        wandb.log({
                            "epoch": epoch,
                            "loss": train_loss,
                            "train_accuracy_epoch": train_acc,
                            "test_accuracy_epoch": val_acc,
                        })
                    except ImportError:
                        pass

                # Save after 10 epochs and if accuracy improves
                if val_acc > best_accuracy + min_delta:
                    best_accuracy = val_acc
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    if (epoch + 1) >= 10:
                        print(f"%%%%%% Saving model with improved accuracy: {val_acc:.2f}%")
                        torch.save(self.student.state_dict(), 'best_student.pth')
                else:
                    # No significant improvement
                    epochs_without_improvement += 1
                    print(f"%%%%%% No improvement for {epochs_without_improvement}/{patience} epochs")

                    if epochs_without_improvement >= patience:
                        print(f"%%%%%% Early stopping triggered after {epoch + 1} epochs")
                        print(f"%%%%%% Best accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")
                        break
            else:
                print()  # Just newline if no validation

        print("%%%%%% Training complete. Best accuracy achieved:", best_accuracy)
        return self.student, best_accuracy, best_epoch