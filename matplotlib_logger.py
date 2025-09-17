"""
Beautiful Matplotlib logger for tracking training metrics

This module provides a clean, professional way to visualize training metrics
using Matplotlib and Seaborn, generating publication-ready plots that update
in real-time during training.
"""

import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class MatplotlibLogger:
    """
    Beautiful Matplotlib logger for tracking training metrics

    Creates professional, publication-ready plots that update in real-time
    during training. Much cleaner and more customizable than TensorBoard.
    """

    def __init__(self, save_dir=None, experiment_name=None, style="whitegrid"):
        """
        Initialize the Matplotlib logger

        Args:
            save_dir (str, optional): Directory to save plots.
                                    Defaults to 'outputs/plots'
            experiment_name (str, optional): Name for this experiment.
                                           Defaults to timestamp-based name
            style (str): Seaborn style theme. Options: whitegrid, darkgrid, white, dark, ticks
        """
        # Set default save directory
        if save_dir is None:
            save_dir = "outputs/plots"

        # Set default experiment name with timestamp
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"captcha_training_{timestamp}"

        # Create full path for this experiment
        self.save_path = os.path.join(save_dir, experiment_name)
        self.experiment_name = experiment_name

        # Ensure directory exists
        os.makedirs(self.save_path, exist_ok=True)

        # Set up beautiful styling
        plt.style.use("seaborn-v0_8")
        sns.set_theme(style=style)
        sns.set_palette("husl")

        # Initialize data storage
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.accuracies = []

        # Configure matplotlib for better plots
        plt.rcParams.update(
            {
                "figure.figsize": (12, 8),
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 11,
                "font.size": 11,
                "lines.linewidth": 2.5,
                "lines.markersize": 8,
            }
        )

        print(f"Matplotlib logging initialized: {self.save_path}")

    def log_epoch_metrics(
        self, epoch, train_loss, val_loss, learning_rate, accuracy=None
    ):
        """
        Log metrics for a single epoch and update plots

        Args:
            epoch (int): Current epoch number
            train_loss (float): Training loss for this epoch
            val_loss (float): Validation loss for this epoch
            learning_rate (float): Current learning rate
            accuracy (float, optional): Validation accuracy for this epoch
        """
        # Store the data
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(learning_rate)
        if accuracy is not None:
            self.accuracies.append(accuracy)

        # Generate updated plots
        self._create_plots()

        # Only print every 5 epochs to reduce clutter
        if epoch % 5 == 0 or epoch == 1:
            print(f"Plots updated for epoch {epoch}")

    def _create_plots(self):
        """Create and save all training plots"""
        if len(self.epochs) < 1:
            return

        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 10))

        # Plot 1: Loss curves
        plt.subplot(2, 2, 1)
        plt.plot(
            self.epochs,
            self.train_losses,
            "o-",
            label="Training Loss",
            color="#e74c3c",
            linewidth=2.5,
            markersize=6,
        )
        plt.plot(
            self.epochs,
            self.val_losses,
            "s-",
            label="Validation Loss",
            color="#3498db",
            linewidth=2.5,
            markersize=6,
        )
        plt.title("Training & Validation Loss", fontweight="bold", pad=20)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)

        # Add trend lines if we have enough data
        if len(self.epochs) > 2:
            z_train = np.polyfit(self.epochs, self.train_losses, 1)
            p_train = np.poly1d(z_train)
            plt.plot(
                self.epochs,
                p_train(self.epochs),
                "--",
                color="#e74c3c",
                alpha=0.5,
                linewidth=1.5,
            )

            z_val = np.polyfit(self.epochs, self.val_losses, 1)
            p_val = np.poly1d(z_val)
            plt.plot(
                self.epochs,
                p_val(self.epochs),
                "--",
                color="#3498db",
                alpha=0.5,
                linewidth=1.5,
            )

        # Plot 2: Learning Rate
        plt.subplot(2, 2, 2)
        plt.plot(
            self.epochs,
            self.learning_rates,
            "D-",
            label="Learning Rate",
            color="#f39c12",
            linewidth=2.5,
            markersize=6,
        )
        plt.title("Learning Rate Schedule", fontweight="bold", pad=20)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.yscale("log")  # Log scale for learning rate
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)

        # Plot 3: Accuracy (if available)
        if self.accuracies:
            plt.subplot(2, 2, 3)
            plt.plot(
                self.epochs,
                [acc * 100 for acc in self.accuracies],
                "^-",
                label="Validation Accuracy",
                color="#27ae60",
                linewidth=2.5,
                markersize=6,
            )
            plt.title("Model Accuracy", fontweight="bold", pad=20)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.ylim(0, 100)
            plt.legend(frameon=True, fancybox=True, shadow=True)
            plt.grid(True, alpha=0.3)

            # Add accuracy trend line
            if len(self.epochs) > 2:
                z_acc = np.polyfit(
                    self.epochs, [acc * 100 for acc in self.accuracies], 1
                )
                p_acc = np.poly1d(z_acc)
                plt.plot(
                    self.epochs,
                    p_acc(self.epochs),
                    "--",
                    color="#27ae60",
                    alpha=0.5,
                    linewidth=1.5,
                )

        # Plot 4: Loss difference and convergence
        plt.subplot(2, 2, 4)
        loss_diff = [abs(t - v) for t, v in zip(self.train_losses, self.val_losses)]
        plt.plot(
            self.epochs,
            loss_diff,
            "v-",
            label="|Train - Val| Loss",
            color="#9b59b6",
            linewidth=2.5,
            markersize=6,
        )
        plt.title("Training Convergence", fontweight="bold", pad=20)
        plt.xlabel("Epoch")
        plt.ylabel("Loss Difference")
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)

        # Add overall title
        fig.suptitle(
            f"Training Progress - {self.experiment_name}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the plot
        plot_path = os.path.join(self.save_path, "training_progress.png")
        plt.savefig(
            plot_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )

        # Also save individual plots for easy viewing
        self._save_individual_plots()

        # Close the figure to free memory
        plt.close(fig)

    def _save_individual_plots(self):
        """Save individual plots for each metric"""
        # Individual loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.epochs,
            self.train_losses,
            "o-",
            label="Training Loss",
            color="#e74c3c",
            linewidth=3,
            markersize=8,
        )
        plt.plot(
            self.epochs,
            self.val_losses,
            "s-",
            label="Validation Loss",
            color="#3498db",
            linewidth=3,
            markersize=8,
        )
        plt.title("Loss Curves", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        loss_path = os.path.join(self.save_path, "loss_curves.png")
        plt.savefig(
            loss_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        plt.close()

        # Individual accuracy plot (if available)
        if self.accuracies:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.epochs,
                [acc * 100 for acc in self.accuracies],
                "^-",
                label="Validation Accuracy",
                color="#27ae60",
                linewidth=3,
                markersize=8,
            )
            plt.title("Accuracy Progress", fontsize=16, fontweight="bold", pad=20)
            plt.xlabel("Epoch", fontsize=14)
            plt.ylabel("Accuracy (%)", fontsize=14)
            plt.ylim(0, 100)
            plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            acc_path = os.path.join(self.save_path, "accuracy_progress.png")
            plt.savefig(
                acc_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            plt.close()

    def create_summary_plot(self):
        """Create a final summary plot at the end of training"""
        if len(self.epochs) < 1:
            return

        # Create a comprehensive summary
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Training Summary - {self.experiment_name}",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # Plot 1: Loss curves with statistics
        ax1 = axes[0, 0]
        ax1.plot(
            self.epochs,
            self.train_losses,
            "o-",
            label="Training Loss",
            color="#e74c3c",
            linewidth=2.5,
        )
        ax1.plot(
            self.epochs,
            self.val_losses,
            "s-",
            label="Validation Loss",
            color="#3498db",
            linewidth=2.5,
        )
        ax1.set_title("Final Loss Curves", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add final values as text
        final_train = self.train_losses[-1]
        final_val = self.val_losses[-1]
        ax1.text(
            0.05,
            0.95,
            f"Final Train Loss: {final_train:.4f}\nFinal Val Loss: {final_val:.4f}",
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Plot 2: Learning rate
        ax2 = axes[0, 1]
        ax2.plot(self.epochs, self.learning_rates, "D-", color="#f39c12", linewidth=2.5)
        ax2.set_title("Learning Rate Schedule", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Accuracy (if available)
        ax3 = axes[0, 2]
        if self.accuracies:
            ax3.plot(
                self.epochs,
                [acc * 100 for acc in self.accuracies],
                "^-",
                color="#27ae60",
                linewidth=2.5,
            )
            ax3.set_title("Accuracy Progress", fontweight="bold")
            ax3.set_ylim(0, 100)
            final_acc = self.accuracies[-1] * 100
            ax3.text(
                0.05,
                0.95,
                f"Final Accuracy: {final_acc:.2f}%",
                transform=ax3.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
            )
        else:
            ax3.text(
                0.5,
                0.5,
                "No Accuracy Data",
                transform=ax3.transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )
            ax3.set_title("Accuracy Progress", fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Accuracy (%)")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Loss difference
        ax4 = axes[1, 0]
        loss_diff = [abs(t - v) for t, v in zip(self.train_losses, self.val_losses)]
        ax4.plot(self.epochs, loss_diff, "v-", color="#9b59b6", linewidth=2.5)
        ax4.set_title("Overfitting Monitor", fontweight="bold")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("|Train - Val| Loss")
        ax4.grid(True, alpha=0.3)

        # Plot 5: Training statistics
        ax5 = axes[1, 1]
        ax5.axis("off")
        stats_text = f"""
Training Statistics:
        
Total Epochs: {len(self.epochs)}
Best Train Loss: {min(self.train_losses):.4f}
Best Val Loss: {min(self.val_losses):.4f}
Final Learning Rate: {self.learning_rates[-1]:.2e}
"""
        if self.accuracies:
            stats_text += f"Best Accuracy: {max(self.accuracies)*100:.2f}%\n"
            stats_text += f"Final Accuracy: {self.accuracies[-1]*100:.2f}%"

        ax5.text(
            0.1,
            0.9,
            stats_text,
            transform=ax5.transAxes,
            verticalalignment="top",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        # Plot 6: Loss improvement
        ax6 = axes[1, 2]
        if len(self.train_losses) > 1:
            train_improvement = [
                (self.train_losses[0] - loss) / self.train_losses[0] * 100
                for loss in self.train_losses
            ]
            val_improvement = [
                (self.val_losses[0] - loss) / self.val_losses[0] * 100
                for loss in self.val_losses
            ]

            ax6.plot(
                self.epochs,
                train_improvement,
                "o-",
                label="Train Improvement",
                color="#e74c3c",
                linewidth=2.5,
            )
            ax6.plot(
                self.epochs,
                val_improvement,
                "s-",
                label="Val Improvement",
                color="#3498db",
                linewidth=2.5,
            )
            ax6.set_title("Loss Improvement (%)", fontweight="bold")
            ax6.set_xlabel("Epoch")
            ax6.set_ylabel("Improvement (%)")
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(
                0.5,
                0.5,
                "Need > 1 Epoch",
                transform=ax6.transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )
            ax6.set_title("Loss Improvement (%)", fontweight="bold")

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        # Save summary plot
        summary_path = os.path.join(self.save_path, "training_summary.png")
        plt.savefig(
            summary_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        print(f"Training summary saved: {summary_path}")

    def close(self):
        """Close the logger and create final summary"""
        self.create_summary_plot()
        print(f"Matplotlib logging completed!")
        print(f"All plots saved in: {self.save_path}")
        print(f"View plots by opening the PNG files in your browser or image viewer")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
