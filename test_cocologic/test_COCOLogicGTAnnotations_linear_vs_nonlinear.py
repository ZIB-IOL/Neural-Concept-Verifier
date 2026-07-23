import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, balanced_accuracy_score
from generate_COCOLogicGTAnnotations import LogicalConceptAnnotationDataset, load_category_mapping

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_and_evaluate_logical_classifier(dataset, model_type='linear', batch_size=64, epochs=10, lr=1e-2, verbose=True, use_class_weights=True):
    """
    model_type: 'linear' or 'mlp'
    """
    input_dim = dataset[0][0].shape[0]
    num_classes = len(dataset.class_names)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model selection
    if model_type == 'linear':
        model = nn.Linear(input_dim, num_classes)
    elif model_type == 'mlp':
        model = MLPClassifier(input_dim, num_classes)
    else:
        raise ValueError("model_type must be 'linear' or 'mlp'")

    # Optionally compute class weights
    if use_class_weights:
        y_train_all = [label.item() for _, label in train_dataset]
        class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_train_all)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    report = classification_report(y_true, y_pred, target_names=dataset.class_names, zero_division=0)
    print(f"\n📊 Classification Report ({model_type} model):\n")
    print(report)

    # After collecting y_true and y_pred
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"\n🎯 Test Balanced Accuracy: {balanced_acc:.4f}\n")

    return model


if __name__ == "__main__":
    # Replace these paths with your local COCO dataset paths
    annotation_path = "datasets/coco/annotations/instances_train2017.json"

    # Load category mapping
    category_map = load_category_mapping(annotation_path)

    # Load dataset
    dataset = LogicalConceptAnnotationDataset(
        annotation_file=annotation_path,
        category_id_to_name=category_map,
        filter_no_labels=True,
        exclusive_label=True,
        exclusive_match_only=True,
        log_statistics=True
    )

    # Run linear classifier
    train_and_evaluate_logical_classifier(dataset, model_type='linear', epochs=150)

    # # Run MLP classifier
    train_and_evaluate_logical_classifier(dataset, model_type='mlp', epochs=20)
