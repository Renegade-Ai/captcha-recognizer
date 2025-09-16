#!/usr/bin/env python3
"""
Enhanced test script with output logging for GitHub repository
"""

import json
import os
from datetime import datetime

import torch

from load_model import load_model_with_metadata, predict_captcha


def save_prediction_results(results, timestamp):
    """Save prediction results to outputs directory"""
    os.makedirs("outputs/predictions", exist_ok=True)

    # Save as JSON
    json_path = f"outputs/predictions/predictions_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save summary as text
    txt_path = f"outputs/predictions/accuracy_report_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write(f"CAPTCHA Prediction Results - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Samples: {results['summary']['total_samples']}\n")
        f.write(f"Correct Predictions: {results['summary']['total_correct']}\n")
        f.write(f"Accuracy: {results['summary']['accuracy']:.2%}\n\n")

        f.write("Individual Results:\n")
        f.write("-" * 30 + "\n")
        for pred in results["predictions"]:
            status = "✓" if pred["correct"] else "✗"
            f.write(
                f"{status} {pred['image']:15s} | {pred['actual']:8s} → {pred['predicted']:8s}\n"
            )

    return json_path, txt_path


def test_single_prediction(image_path="input/JjkAq.png"):
    """
    Test prediction on a single CAPTCHA image with enhanced logging
    """
    print("🧪 Testing CAPTCHA Prediction")
    print("=" * 50)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check if model file exists
    model_file = "outputs/models/captcha_model_with_metadata_latest.pth"
    if not os.path.exists(model_file):
        print(f"❌ Model file '{model_file}' not found!")
        print("💡 Train the model first by running: python train.py")
        return

    # Check if test image exists
    if not os.path.exists(image_path):
        # Try to find any image in the input directory
        if os.path.exists("input") and os.listdir("input"):
            image_files = [f for f in os.listdir("input") if f.endswith(".png")]
            if image_files:
                image_path = os.path.join("input", image_files[0])
                print(f"📸 Using sample image: {image_path}")
            else:
                print("❌ No .png images found in input/ directory")
                return
        else:
            print(f"❌ Image file '{image_path}' not found!")
            return

    try:
        # Load the trained model
        print("🔄 Loading trained model...")
        model, metadata = load_model_with_metadata(model_file)
        print("✅ Model loaded successfully!")

        # Display model information
        print(f"\n📊 Model Information:")
        print(f"   • Number of characters: {metadata['num_chars']}")
        print(
            f"   • Image dimensions: {metadata['image_height']}x{metadata['image_width']}"
        )
        print(f"   • Epochs trained: {metadata['epoch_trained']}")
        print(f"   • Character set: {''.join(metadata['char_classes'])}")

        # Extract the actual CAPTCHA text from filename (for comparison)
        filename = os.path.basename(image_path)
        actual_text = os.path.splitext(filename)[0]  # Remove .png extension

        print(f"\n🎯 Prediction Test:")
        print(f"   • Image: {image_path}")
        print(f"   • Actual text: {actual_text}")

        # Make prediction
        print("🔮 Running prediction...")
        predicted_text = predict_captcha(
            model=model, image_path=image_path, char_classes=metadata["char_classes"]
        )

        print(f"   • Predicted text: {predicted_text}")

        # Compare results
        is_correct = predicted_text == actual_text
        if is_correct:
            print("✅ CORRECT! Prediction matches actual text")
        else:
            print("❌ INCORRECT! Prediction doesn't match")

        # Calculate character-level accuracy
        correct_chars = sum(1 for a, p in zip(actual_text, predicted_text) if a == p)
        total_chars = max(len(actual_text), len(predicted_text))
        char_accuracy = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
        print(
            f"📈 Character accuracy: {char_accuracy:.1f}% ({correct_chars}/{total_chars})"
        )

        # Save results
        results = {
            "session_info": {
                "timestamp": timestamp,
                "model_used": os.path.basename(model_file),
                "test_type": "single_prediction",
            },
            "prediction": {
                "image": filename,
                "actual": actual_text,
                "predicted": predicted_text,
                "correct": is_correct,
                "character_accuracy": char_accuracy,
            },
        }

        os.makedirs("outputs/predictions", exist_ok=True)
        result_path = f"outputs/predictions/single_prediction_{timestamp}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"💾 Results saved to: {result_path}")

        return results

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        print("💡 Make sure the model was trained successfully")


def test_batch_predictions(num_samples=10):
    """
    Test predictions on multiple CAPTCHA images with comprehensive logging
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n🧪 Testing Batch Predictions ({num_samples} samples)")
    print("=" * 50)

    if not os.path.exists("input"):
        print("❌ Input directory not found!")
        return

    # Get sample images
    image_files = [f for f in os.listdir("test_input") if f.endswith(".png")][
        :num_samples
    ]

    if not image_files:
        print("❌ No .png images found in input/ directory")
        return

    try:
        # Load model once
        model, metadata = load_model_with_metadata()

        predictions = []
        correct_predictions = 0
        total_predictions = len(image_files)

        print(f"📝 Testing {total_predictions} images...")
        print(
            "   "
            + "ID".ljust(3)
            + "File".ljust(20)
            + "Actual".ljust(10)
            + "Predicted".ljust(10)
            + "Status"
        )
        print("   " + "-" * 55)

        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join("input", filename)
            actual_text = os.path.splitext(filename)[0]

            # Make prediction
            predicted_text = predict_captcha(
                model=model,
                image_path=image_path,
                char_classes=metadata["char_classes"],
            )

            is_correct = predicted_text == actual_text
            if is_correct:
                correct_predictions += 1

            # Calculate character accuracy
            correct_chars = sum(
                1 for a, p in zip(actual_text, predicted_text) if a == p
            )
            total_chars = max(len(actual_text), len(predicted_text))
            char_accuracy = correct_chars / total_chars if total_chars > 0 else 0

            status = "✅" if is_correct else "❌"
            print(
                f"   {i:2d}. {filename[:15]:15s} {actual_text:8s} {predicted_text:8s} {status}"
            )

            # Store result
            predictions.append(
                {
                    "image": filename,
                    "actual": actual_text,
                    "predicted": predicted_text,
                    "correct": is_correct,
                    "character_accuracy": char_accuracy,
                }
            )

        # Calculate overall accuracy
        accuracy = correct_predictions / total_predictions
        char_accuracies = [p["character_accuracy"] for p in predictions]
        avg_char_accuracy = (
            sum(char_accuracies) / len(char_accuracies) if char_accuracies else 0
        )

        print(f"\n📊 Overall Results:")
        print(f"   • Correct predictions: {correct_predictions}/{total_predictions}")
        print(f"   • Sequence accuracy: {accuracy:.1%}")
        print(f"   • Average character accuracy: {avg_char_accuracy:.1%}")

        # Prepare results for saving
        results = {
            "session_info": {
                "timestamp": timestamp,
                "model_used": metadata.get("training_timestamp", "unknown"),
                "total_samples": total_predictions,
                "test_type": "batch_prediction",
            },
            "predictions": predictions,
            "summary": {
                "total_correct": correct_predictions,
                "total_samples": total_predictions,
                "accuracy": accuracy,
                "average_character_accuracy": avg_char_accuracy,
            },
        }

        # Save results
        json_path, txt_path = save_prediction_results(results, timestamp)
        print(f"💾 Results saved to:")
        print(f"   • JSON: {json_path}")
        print(f"   • Report: {txt_path}")

        return results

    except Exception as e:
        print(f"❌ Error during batch testing: {str(e)}")
        return None


def analyze_errors(results):
    """Analyze prediction errors for insights"""
    if not results or "predictions" not in results:
        return

    print(f"\n🔍 Error Analysis:")
    print("=" * 30)

    errors = [p for p in results["predictions"] if not p["correct"]]
    if not errors:
        print("🎉 No errors found! Perfect predictions!")
        return

    print(f"Total errors: {len(errors)}")

    # Character-level error analysis
    char_errors = {}
    for error in errors:
        actual = error["actual"]
        predicted = error["predicted"]
        for i, (a, p) in enumerate(zip(actual, predicted)):
            if a != p:
                key = f"{a}→{p}"
                char_errors[key] = char_errors.get(key, 0) + 1

    if char_errors:
        print("\nMost common character confusions:")
        sorted_errors = sorted(char_errors.items(), key=lambda x: x[1], reverse=True)
        for confusion, count in sorted_errors[:5]:
            print(f"   {confusion}: {count} times")


if __name__ == "__main__":
    print("🚀 Enhanced CAPTCHA Prediction Test Suite")
    print("=" * 60)

    # Test single prediction
    # results = test_single_prediction()

    # Test batch predictions
    results = test_batch_predictions(num_samples=20)

    print(f"\n✨ Testing completed!")
    print(f"\n📁 Outputs saved to:")
    print(f"   • outputs/predictions/ - Prediction results")
    print(f"   • outputs/models/ - Trained models")
    print(f"\n💡 Usage examples:")
    print("   • View results: cat outputs/predictions/accuracy_report_*.txt")
    print(
        '   • Load model: python -c "from load_model import *; model, meta = load_model_with_metadata()"'
    )
    print("   • Custom test: modify num_samples in test_batch_predictions()")
