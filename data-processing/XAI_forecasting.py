import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#from aix360.algorithms.lime import LimeTabularExplainer
from typing import Union, Any, List, Tuple
import random
from datetime import datetime, timedelta
import xgboost as xgb
import time
from lemon import LemonExplainer, gaussian_kernel
from scipy.special import gammainccinv
from sklearn.linear_model import Ridge

class ForecastExplainer:
    def __init__(
        self,
        model: Any,
        training_data: Union[np.ndarray, torch.Tensor],
        training_outputs: Union[np.ndarray, torch.Tensor] = None,  # Made optional with default None
        use_residuals: bool = False,
        device: torch.device = None
    ):
        """
        Initialize the ForecastExplainer.

        Args:
            model (Any): A trained forecasting model (PyTorch nn.Module or sklearn/xgboost model).
            training_data (Union[np.ndarray, torch.Tensor]): Training data of shape (num_samples, seq_length).
                Used for LEMON explanations, residuals mode, and bootstrap noise estimation.
            training_outputs (Union[np.ndarray, torch.Tensor], optional): Training outputs of shape (num_samples,).
                Required only when use_residuals is True. Defaults to None.
            use_residuals (bool): Whether to calculate bounds using residuals. Default is False.
            device (torch.device, optional): Device to run the model on (CPU or GPU). If None, it is auto-selected.

        Raises:
            ValueError: If use_residuals is True but training_outputs is None.
        """
        # Check if training_outputs is provided when use_residuals is True
        if use_residuals and training_outputs is None:
            raise ValueError("training_outputs must be provided when use_residuals is True")

        self.model = model
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.use_residuals = use_residuals

        # Convert training_data and training_outputs to numpy arrays if they're tensors
        if isinstance(training_data, torch.Tensor):
            training_data = training_data.detach().cpu().numpy()
        if isinstance(training_outputs, torch.Tensor) and training_outputs is not None:
            training_outputs = training_outputs.detach().cpu().numpy()

        self.training_data = training_data
        self.training_outputs = training_outputs
        self.num_samples, self.seq_length = self.training_data.shape

        # Calculate and store training data statistics based on the selected mode
        if self.use_residuals:
            self.residuals = self.calculate_residuals()
        else:
            # Pre-calculate training data std for bootstrap mode
            self.training_std = np.std(self.training_data)

    def calculate_residuals(self) -> np.ndarray:
        """
        Calculate residuals between the model's predictions on the training data and the actual training outputs.
        
        Returns:
            np.ndarray: Residuals of shape (num_samples,).
        """
        if isinstance(self.model, nn.Module):
            # For PyTorch models, predict in batches
            inputs_tensor = torch.from_numpy(self.training_data.reshape(self.num_samples, self.seq_length, 1)).float().to(self.device)
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(inputs_tensor).cpu().numpy().flatten()
        else:
            # For sklearn/xgboost models, predict directly on the entire dataset
            predictions = self.model.predict(self.training_data).flatten()
        
        residuals = self.training_outputs - predictions
        return residuals

    def predict(self, input_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict a single step ahead given an input sequence.

        Handles both PyTorch and sklearn/xgboost models.

        For PyTorch models:
        - input_data is reshaped to (1, seq_length, 1)
        - A torch.Tensor is created and passed through the model in eval mode without gradients.

        For sklearn/xgboost models:
        - input_data is reshaped to (1, seq_length)
        - The model's 'predict' method is called directly.

        Args:
            input_data (Union[np.ndarray, torch.Tensor]): The input sequence of shape (seq_length,).

        Globals:
            None

        Raises:
            None

        Returns:
            np.ndarray: A 1D numpy array (shape (1,)) representing the model's prediction.
        """
        # Ensure input_data is a numpy array
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.detach().cpu().numpy()

        if isinstance(self.model, nn.Module):
            # PyTorch model prediction
            input_data_reshaped = input_data.reshape(1, self.seq_length, 1)
            input_tensor = torch.from_numpy(input_data_reshaped).float().to(self.device)
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(input_tensor).cpu().numpy().flatten()
        else:
            # sklearn/xgboost model prediction
            input_data_reshaped = input_data.reshape(1, -1)
            prediction = self.model.predict(input_data_reshaped).flatten()

        return prediction

    def predict_with_uncertainty(
        self,
        input_data: np.ndarray,
        n_samples: int = 100,
        confidence: float = 0.95,
        step: int = 0
    ) -> Tuple[float, float, float, float]:
        """
        Make a prediction with uncertainty estimation.

        Two modes of operation:
        1. Residuals mode (use_residuals=True):
           - Uses historical residuals to compute standard deviation
           - Applies z-scores based on confidence level
           - Scales uncertainty with square root of horizon

        2. Bootstrap mode (use_residuals=False):
           - Uses pre-calculated training data standard deviation
           - Scales noise with prediction horizon
           - Generates perturbed versions of input data
           - Runs model predictions on perturbed inputs
           - Calculates bounds from distribution of predictions

        Args:
            input_data (np.ndarray): The input sequence of shape (seq_length,).
            n_samples (int, optional): Number of bootstrap samples/perturbed inputs. Default is 100.
            confidence (float, optional): Confidence level for the interval (e.g. 0.95 for 95%). Default is 0.95.
            step (int, optional): The step number in the autoregressive sequence, used for uncertainty scaling. Default is 0.

        Returns:
            Tuple[float, float, float, float]: A tuple containing:
                mean_pred (float): Mean prediction (raw prediction in residuals mode, bootstrap mean in bootstrap mode).
                lower_bound (float): Lower bound of the confidence interval.
                upper_bound (float): Upper bound of the confidence interval.
                confidence (float): Confidence level used for the interval.
        """
        mean_pred = self.predict(input_data)[0]

        # Scale uncertainty with prediction horizon ("Square Root of Time" rule in volatility scaling)
        uncertainty_scale = np.sqrt(1 + step)  # Square root growth of uncertainty

        if self.use_residuals:
            # Calculate statistics of residuals
            residual_std = np.std(self.residuals)
            
            # Calculate z-score for the desired confidence interval
            # For example, for 95% confidence, z_score ≈ 1.96
            z_score = np.abs(np.percentile(np.random.standard_normal(10000), 
                                         [((1-confidence)/2)*100, (1-(1-confidence)/2)*100]))
            
            # Calculate bounds
            lower_bound = mean_pred - z_score[0] * residual_std * uncertainty_scale
            upper_bound = mean_pred + z_score[1] * residual_std * uncertainty_scale

        else:
            # Use pre-calculated training data std and scale it with step
            bootstrap_noise_std = self.training_std * uncertainty_scale

            perturbed_inputs = np.repeat(input_data.reshape(1, -1), n_samples, axis=0) 
            perturbed_inputs += np.random.normal(0, bootstrap_noise_std, size=perturbed_inputs.shape)

            if isinstance(self.model, nn.Module):
                # PyTorch model: run batch through model
                inputs_tensor = torch.from_numpy(perturbed_inputs.reshape(n_samples, self.seq_length, 1)).float().to(self.device)
                self.model.eval()
                with torch.no_grad():
                    predictions = self.model(inputs_tensor).cpu().numpy().flatten()
            else:
                # sklearn/xgboost model: predict directly on batch
                predictions = self.model.predict(perturbed_inputs)

            predictions = np.array(predictions)
            lower_bound = np.percentile(predictions, ((1 - confidence) / 2) * 100)
            upper_bound = np.percentile(predictions, (confidence + (1 - confidence) / 2) * 100)
            mean_pred = np.mean(predictions)

        return mean_pred, lower_bound, upper_bound, confidence


    def explain_prediction(
        self,
        input_data: np.ndarray,
        input_labels: List[str],
        num_features: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Generate a LEMON explanation for the model's prediction on input_data.

        Args:
            input_data (np.ndarray): Input data of shape (seq_length,).
            input_labels (List[str]): Labels corresponding to each step in input_data.
            num_features (int, optional): Number of features to include in the explanation. Default is 10.

        Globals:
            None

        Raises:
            None

        Returns:
            List[Tuple[str, float]]: A list of (feature_label, importance) pairs.
        """
        input_data_flat = input_data.flatten()

        def predict_fn(data):
            batch_size = data.shape[0]
            if isinstance(self.model, nn.Module):
                # For PyTorch models, reshape to (batch, seq_length, 1)
                inputs = data.reshape(batch_size, self.seq_length, 1)
                inputs_tensor = torch.from_numpy(inputs).float().to(self.device)
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(inputs_tensor).cpu().numpy()
            else:
                outputs = self.model.predict(data)
            return outputs.flatten()
        
        # Let's make a lemonade
        # Same kernel as LIME!!!
        DIMENSIONS = len(input_data_flat)
        kernel_width = np.sqrt(DIMENSIONS) * .75
        p = 0.99
        kernel = lambda x: gaussian_kernel(x, kernel_width)
        radius = kernel_width * np.sqrt(2 * gammainccinv(DIMENSIONS / 2, (1 - p)))

        lemon_explainer = LemonExplainer(
            training_data=self.training_data,
            radius_max=radius,
            distance_kernel=kernel,
            random_state=42
            )
        
        exp = lemon_explainer.explain_instance(input_data_flat, predict_fn, surrogate=Ridge(fit_intercept=True, random_state=123))[0]
        explanation = [(label, value) for label, value in zip(input_labels[:num_features], exp.feature_contribution)]
        return explanation

    def predict_and_explain(
        self,
        input_data: Union[np.ndarray, torch.Tensor],
        n_predictions: int,
        input_labels: List[str],
        num_features: int = 5,
        confidence: float = 0.95,
        n_samples: int = 100,
        use_mean_pred: bool = False
    ) -> dict:
        """
        Perform autoregressive prediction and explanation for n_predictions steps.

        Uncertainty bounds are calculated differently based on use_residuals:
        - If True: Uses residuals and z-scores with square root time scaling
        - If False: Uses bootstrap sampling of perturbed inputs

        The prediction used for the next step can be either:
        - Bootstrap mean prediction if use_mean_pred=True (only affects bootstrap mode)
        - Raw model prediction if use_mean_pred=False

        Args:
            input_data (Union[np.ndarray, torch.Tensor]): Initial input sequence of shape (seq_length,).
            n_predictions (int): Number of autoregressive predictions to make.
            input_labels (List[str]): Labels corresponding to the input_data.
            num_features (int, optional): Number of features for LEMON explanation. Default is 10.
            confidence (float, optional): Confidence level for interval estimation. Default is 0.95.
            n_samples (int, optional): Number of bootstrap samples for uncertainty estimation. Must be >=100 for meaningful confidence. Default is 100.
            use_mean_pred (bool, optional): If True and in bootstrap mode, use mean of bootstrap 
                                          predictions as final prediction; else use raw prediction. 
                                          Has no effect in residuals mode. Default is False.

        Globals:
            None

        Raises:
            None

        Returns:
            dict: A dictionary containing:
                'Predicted_value' (List[float]): List of predicted values.
                'Lower_bound' (List[float]): List of lower bounds.
                'Upper_bound' (List[float]): List of upper bounds.
                'Confidence_score' (List[float]): List of confidence levels used.
                'lemon_explaination' (List[List[Tuple[str,float]]]): LEMON explanations per step.
                'Date_prediction' (List[str]): Predicted date labels for each step.
        """
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.detach().cpu().numpy()

        predicted_values = []
        lower_bounds = []
        upper_bounds = []
        confidence_scores = []
        lemon_explanations = []
        date_predictions = []

        current_input = input_data.copy()
        current_labels = input_labels.copy()

        for i in range(n_predictions):
            # Compute uncertainties
            mean_pred, lower_bound, upper_bound, confidence_level = self.predict_with_uncertainty(
                current_input, n_samples=n_samples, confidence=confidence, step=i
            )
            # Compute raw prediction
            raw_pred = self.predict(current_input)[0]

            # Decide which prediction to use
            final_pred = mean_pred if use_mean_pred else raw_pred

            explanation = self.explain_prediction(current_input, current_labels, num_features=num_features)

            predicted_values.append(final_pred)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            confidence_scores.append(confidence_level)
            lemon_explanations.append(explanation)

            # Update the input_data and labels for the next step
            current_input = np.append(current_input[1:], final_pred)
            last_label_date = datetime.strptime(current_labels[-1], "%Y-%m-%d")
            new_label_date = last_label_date + timedelta(days=1)
            new_label = new_label_date.strftime("%Y-%m-%d")
            current_labels = current_labels[1:] + [new_label]
            date_predictions.append(new_label)

        out_dict = {
            'Predicted_value': predicted_values,
            'Lower_bound': lower_bounds,
            'Upper_bound': upper_bounds,
            'Confidence_score': confidence_scores,
            'lemon_explanation': lemon_explanations,
            'Date_prediction': date_predictions
        }

        return out_dict


def main():
    print()

if __name__ == '__main__':
    main()