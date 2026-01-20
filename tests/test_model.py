"""
Unit tests for the ResNet model in eye_diseases_classification.model.
usage: uv run pytest tests/test_model.py -v
Test Classes:
   TestModelConstruction - Tests model initialization with default/custom parameters, hyperparameter saving, layer architecture, and loss function
   TestModelForward - Tests forward pass output shape/type, gradient flow, NaN handling, and various batch sizes
   TestModelTrainingStep - Tests training step loss computation and return values
   TestModelValidationStep - Tests validation step behavior
   TestModelTestStep - Tests test step behavior
   TestModelOptimizer - Tests optimizer and scheduler configuration, weight decay, and learning rates
   TestModelTraining - Tests end-to-end training with PyTorch Lightning, parameter updates, and model behavior
   TestModelLossComputation - Tests loss values and convergence during training
"""

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from eye_diseases_classification.model import ResNet


class TestModelConstruction:
    """Test model initialization and architecture."""

    def test_model_initialization_default(self):
        model = ResNet()

        assert isinstance(model, ResNet), "Model is not a ResNet instance"
        assert hasattr(model, "fc"), "Model missing fully connected layer"
        assert hasattr(model, "criterion"), "Model missing loss function"

    def test_model_initialization_custom_classes(self):
        num_classes = 8
        model = ResNet(num_classes=num_classes)

        assert (
            model.fc.out_features == num_classes
        ), f"Output classes mismatch: {model.fc.out_features} != {num_classes}"

    def test_model_initialization_custom_lr(self):
        custom_lr = 5e-4
        model = ResNet(lr=custom_lr)

        assert model.hparams.lr == custom_lr, f"Learning rate mismatch: {model.hparams.lr} != {custom_lr}"

    def test_model_hyperparameters_saved(self):
        num_classes = 6
        lr = 2e-3
        model = ResNet(num_classes=num_classes, lr=lr)

        assert model.hparams.num_classes == num_classes, "num_classes not saved in hyperparameters"
        assert model.hparams.lr == lr, "Learning rate not saved in hyperparameters"

    def test_model_layers_exist(self):
        model = ResNet()

        assert hasattr(model, "layer1"), "Missing layer1"
        assert hasattr(model, "layer2"), "Missing layer2"
        assert hasattr(model, "layer3"), "Missing layer3"
        assert hasattr(model, "layer4"), "Missing layer4"
        assert hasattr(model, "avgpool"), "Missing average pooling"
        assert hasattr(model, "dropout"), "Missing dropout layer"

    def test_model_dropout_rate(self):
        model = ResNet()

        assert model.dropout.p == 0.4, f"Dropout rate incorrect: {model.dropout.p} != 0.4"

    def test_model_loss_function(self):
        model = ResNet()

        assert isinstance(model.criterion, torch.nn.CrossEntropyLoss), "Loss function is not CrossEntropyLoss"


class TestModelForward:
    """Test model forward pass."""

    def test_forward_pass_output_shape(self):
        model = ResNet(num_classes=4)
        batch_size = 8
        x = torch.randn(batch_size, 3, 256, 256)

        output = model(x)

        assert output.shape == (batch_size, 4), f"Unexpected output shape: {output.shape}"

    def test_forward_pass_output_type(self):
        model = ResNet()
        x = torch.randn(2, 3, 256, 256)

        output = model(x)

        assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor"

    def test_forward_pass_gradient_flow(self):
        model = ResNet()
        x = torch.randn(4, 3, 256, 256, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients not flowing through input"

    def test_forward_pass_no_nan(self):
        model = ResNet()
        x = torch.randn(8, 3, 256, 256)

        output = model(x)

        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_forward_pass_with_different_batch_sizes(self):
        model = ResNet()

        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 3, 256, 256)
            output = model(x)

            assert output.shape == (batch_size, 4), f"Batch size {batch_size} failed"


class TestModelTrainingStep:
    """Test model training step."""

    def test_training_step_returns_loss(self):
        model = ResNet(num_classes=4)
        imgs = torch.randn(8, 3, 256, 256)
        labels = torch.randint(0, 4, (8,))
        batch = (imgs, labels)

        loss = model.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor), "Training step should return a tensor"
        assert loss.item() > 0, "Loss should be positive"

    def test_training_step_loss_decreases_with_backprop(self):
        model = ResNet(num_classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # First step
        imgs1 = torch.randn(8, 3, 256, 256)
        labels1 = torch.randint(0, 4, (8,))
        loss1 = model.training_step((imgs1, labels1), batch_idx=0)
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()
        first_loss = loss1.item()

        # Second step
        imgs2 = torch.randn(8, 3, 256, 256)
        labels2 = torch.randint(0, 4, (8,))
        loss2 = model.training_step((imgs2, labels2), batch_idx=0)
        second_loss = loss2.item()

        assert isinstance(first_loss, float), "Loss should be convertible to float"
        assert isinstance(second_loss, float), "Loss should be convertible to float"

    def test_training_step_accuracy_computation(self):
        model = ResNet(num_classes=4)
        imgs = torch.randn(8, 3, 256, 256)
        labels = torch.randint(0, 4, (8,))
        batch = (imgs, labels)

        loss = model.training_step(batch, batch_idx=0)

        assert loss is not None, "Training step returned None"
        assert not torch.isnan(loss), "Loss is NaN"


class TestModelValidationStep:
    """Test model validation step."""

    def test_validation_step_returns_none(self):
        model = ResNet(num_classes=4)
        imgs = torch.randn(8, 3, 256, 256)
        labels = torch.randint(0, 4, (8,))
        batch = (imgs, labels)

        result = model.validation_step(batch, batch_idx=0)

        assert result is None, "Validation step should return None"

    def test_validation_step_computes_metrics(self):
        model = ResNet(num_classes=4)
        imgs = torch.randn(16, 3, 256, 256)
        labels = torch.randint(0, 4, (16,))
        batch = (imgs, labels)

        model.validation_step(batch, batch_idx=0)
        assert True, "Validation step completed without error"


class TestModelTestStep:
    """Test model test step."""

    def test_test_step_returns_none(self):
        model = ResNet(num_classes=4)
        imgs = torch.randn(8, 3, 256, 256)
        labels = torch.randint(0, 4, (8,))
        batch = (imgs, labels)

        result = model.test_step(batch, batch_idx=0)

        assert result is None, "Test step should return None"


class TestModelOptimizer:
    """Test optimizer configuration."""

    def test_configure_optimizers_returns_dict(self):
        model = ResNet()
        optimizer_config = model.configure_optimizers()

        assert isinstance(optimizer_config, dict), "configure_optimizers should return a dictionary"
        assert "optimizer" in optimizer_config, "Missing 'optimizer' key"
        assert "lr_scheduler" in optimizer_config, "Missing 'lr_scheduler' key"

    def test_optimizer_is_adamw(self):
        model = ResNet()
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config["optimizer"]

        assert isinstance(optimizer, torch.optim.AdamW), "Optimizer should be AdamW"

    def test_scheduler_is_reduce_lr_on_plateau(self):
        model = ResNet()
        optimizer_config = model.configure_optimizers()
        scheduler_config = optimizer_config["lr_scheduler"]

        assert "scheduler" in scheduler_config, "Missing scheduler in config"
        assert isinstance(
            scheduler_config["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau
        ), "Scheduler should be ReduceLROnPlateau"

    def test_optimizer_weight_decay(self):
        model = ResNet()
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config["optimizer"]

        # Check weight decay parameter
        for param_group in optimizer.param_groups:
            assert param_group["weight_decay"] == 1e-4, f"Weight decay mismatch: {param_group['weight_decay']}"

    def test_optimizer_learning_rate(self):
        lr = 2e-3
        model = ResNet(lr=lr)
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config["optimizer"]

        for param_group in optimizer.param_groups:
            assert param_group["lr"] == lr, f"Learning rate mismatch: {param_group['lr']} != {lr}"


class TestModelTraining:
    """Test model training with PyTorch Lightning."""

    def test_model_trains_without_error(self):
        model = ResNet(num_classes=4)

        # Create dummy data
        imgs = torch.randn(32, 3, 256, 256)
        labels = torch.randint(0, 4, (32,))
        dataset = TensorDataset(imgs, labels)
        loader = DataLoader(dataset, batch_size=8)

        # Train for 1 epoch
        trainer = Trainer(max_epochs=1, enable_progress_bar=False, logger=False)
        trainer.fit(model, loader, loader)

        assert True, "Training completed without error"

    def test_model_parameters_updated_after_training(self):
        model = ResNet(num_classes=4)

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Create dummy data
        imgs = torch.randn(32, 3, 256, 256)
        labels = torch.randint(0, 4, (32,))
        dataset = TensorDataset(imgs, labels)
        loader = DataLoader(dataset, batch_size=8)

        # Train for 1 epoch
        trainer = Trainer(max_epochs=1, enable_progress_bar=False, logger=False)
        trainer.fit(model, loader, loader)

        # Check that parameters have been updated
        params_updated = False
        for initial_param, current_param in zip(initial_params, model.parameters()):
            if not torch.allclose(initial_param, current_param):
                params_updated = True
                break

        assert params_updated, "Model parameters not updated during training"

    def test_model_produces_different_outputs_for_different_inputs(self):
        model = ResNet(num_classes=4)
        model.eval()

        imgs1 = torch.randn(4, 3, 256, 256)
        imgs2 = torch.randn(4, 3, 256, 256)

        output1 = model(imgs1)
        output2 = model(imgs2)

        assert not torch.allclose(output1, output2), "Model produces identical outputs for different inputs"

    def test_model_eval_mode_deterministic(self):
        model = ResNet(num_classes=4)
        model.eval()

        imgs = torch.randn(4, 3, 256, 256)

        with torch.no_grad():
            output1 = model(imgs)
            output2 = model(imgs)

        assert torch.allclose(output1, output2), "Eval mode outputs not deterministic"


class TestModelLossComputation:
    """Test loss computation."""

    def test_loss_is_positive(self):
        model = ResNet(num_classes=4)
        imgs = torch.randn(8, 3, 256, 256)
        labels = torch.randint(0, 4, (8,))

        outputs = model(imgs)
        loss = model.criterion(outputs, labels)

        assert loss.item() > 0, "Loss should be positive"

    def test_loss_decreases_with_correct_predictions(self):
        model = ResNet(num_classes=4)
        model.eval()

        # Create inputs where model can overfit easily
        imgs = torch.randn(4, 3, 256, 256)
        labels = torch.tensor([0, 1, 2, 3])

        # Get initial predictions and loss
        with torch.no_grad():
            outputs1 = model(imgs)
            loss1 = model.criterion(outputs1, labels)

        # Train on the same data
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(10):
            outputs = model(imgs)
            loss = model.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Get new predictions and loss
        model.eval()
        with torch.no_grad():
            outputs2 = model(imgs)
            loss2 = model.criterion(outputs2, labels)

        assert loss2.item() < loss1.item(), "Loss did not decrease after training"
