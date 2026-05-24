import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms

STAGE1_IMG_SIZE = (224, 224)   
STAGE2_IMG_SIZE = (224, 224)   

WAGNER_LABELS = {
    0: "Grade 1",
    1: "Grade 2",
    2: "Grade 3",
    3: "Grade 4",}

WAGNER_DESCRIPTIONS = {
    0: "Superficial ulcer — no infection, involves skin only.",
    1: "Deep ulcer — reaches tendon, capsule, or bone.",
    2: "Deep ulcer with abscess, osteomyelitis, or joint sepsis.",
    3: "Partial foot gangrene.",}

GRADE1_PENALTY = 0.35   
TTA_PASSES     = 7      
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Stage1Detector:
    

    def __init__(self, model_path: str):
        print(f"[Stage 1] Loading model from: {model_path}")
        self.model = load_model(model_path)
        print("[Stage 1] Model loaded ✓")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize(STAGE1_IMG_SIZE)
        arr = img_to_array(img)  
        return np.expand_dims(arr, axis=0)   

    def predict(self, image: Image.Image) -> dict:
        x    = self.preprocess(image)
        pred = self.model.predict(x, verbose=0)[0]  

        if len(pred) == 2:
            ulcer_prob = float(pred[0])
        else:
            ulcer_prob = float(pred[0])

        is_ulcer   = ulcer_prob > 0.5
        confidence = ulcer_prob if is_ulcer else 1.0 - ulcer_prob

        return {
            "is_ulcer"  : is_ulcer,
            "confidence": round(confidence * 100, 2),
            "raw_prob"  : round(ulcer_prob, 4),
        }

class DFUWagnerModel(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.45, hidden: int = 512):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=False,
            num_classes=0, drop_rate=dropout, drop_path_rate=0.25,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.backbone.num_features   
        self.head = nn.Sequential(
            nn.Flatten(),                        
            nn.BatchNorm1d(in_features),         
            nn.Dropout(p=dropout),               
            nn.Linear(in_features, hidden),      
            nn.SiLU(),                           
            nn.BatchNorm1d(hidden),              
            nn.Dropout(p=dropout * 0.5),         
            nn.Linear(hidden, num_classes),      
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)   
        x = self.pool(x)                         
        return self.head(x)                      


class Stage2Grader:
   

    def __init__(self, model_path: str):
        print(f"[Stage 2] Loading model from: {model_path}")

        checkpoint = torch.load(model_path, map_location=DEVICE)
        num_classes  = checkpoint.get("num_classes",  4)
        dropout      = checkpoint.get("dropout",      0.3)
        img_size     = checkpoint.get("img_size",     224)
        mean         = checkpoint.get("mean",         [0.485, 0.456, 0.406])
        std          = checkpoint.get("std",          [0.229, 0.224, 0.225])
        best_penalty = checkpoint.get("best_penalty", GRADE1_PENALTY)
        class_names  = checkpoint.get("class_names",  None)

        self.img_size     = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean         = mean
        self.std          = std
        self.best_penalty = best_penalty
        self.num_classes  = num_classes

        print(f"  img_size={self.img_size}, dropout={dropout}, "
              f"penalty={self.best_penalty}, classes={class_names}")

        state_dict = checkpoint["model_state_dict"]
        hidden_dim = int(state_dict["head.3.weight"].shape[0]) if "head.3.weight" in state_dict else 512
        print(f"  hidden_dim={hidden_dim} (auto-detected)")

        self.model = DFUWagnerModel(num_classes=num_classes, dropout=dropout, hidden=hidden_dim)
        self.model.load_state_dict(state_dict)
        self.model.to(DEVICE)
        self.model.eval()
        print("[Stage 2] Model loaded ✓")

        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            sz       = self.img_size[0]   # e.g. 224
            mn, sd   = self.mean, self.std
            self._tta_transforms = [
                A.Compose([A.Resize(sz, sz),      A.Normalize(mn, sd), ToTensorV2()]),
                A.Compose([A.Resize(sz, sz),      A.HorizontalFlip(p=1), A.Normalize(mn, sd), ToTensorV2()]),
                A.Compose([A.Resize(sz, sz),      A.VerticalFlip(p=1),   A.Normalize(mn, sd), ToTensorV2()]),
                A.Compose([A.Resize(sz+24, sz+24),A.CenterCrop(sz, sz),  A.Normalize(mn, sd), ToTensorV2()]),
                A.Compose([A.Resize(sz, sz),      A.RandomRotate90(p=1), A.Normalize(mn, sd), ToTensorV2()]),
                A.Compose([A.Resize(sz, sz),      A.Transpose(p=1),      A.Normalize(mn, sd), ToTensorV2()]),
                A.Compose([A.Resize(sz, sz),      A.CLAHE(clip_limit=3, p=1), A.Normalize(mn, sd), ToTensorV2()]),
            ]
            self._use_albumentations = True
            print("  TTA: using albumentations (matches training) ✓")
        except ImportError:
          
            import warnings
            warnings.warn("albumentations not found — falling back to torchvision TTA. "
                          "Run: pip install albumentations")
            resize_to = (int(self.img_size[0] * 1.14), int(self.img_size[1] * 1.14))
            self._base_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
            self._aug_transform = transforms.Compose([
                transforms.Resize(resize_to),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
            self._use_albumentations = False

    def _apply_grade1_penalty(self, probs: np.ndarray) -> np.ndarray:
        """Multiply Grade-1 probability by penalty factor, then re-normalise."""
        probs = probs.copy()
        probs[0] *= self.best_penalty   
        probs /= probs.sum()
        return probs

    def predict(self, image: Image.Image) -> dict:
        img_rgb      = image.convert("RGB")
        img_np       = np.array(img_rgb)
        accumulated  = np.zeros(self.num_classes, dtype=np.float32)
        n_passes     = 0

        with torch.no_grad():
            if self._use_albumentations:
               
                for aug in self._tta_transforms:
                    tensor = aug(image=img_np)["image"].unsqueeze(0).to(DEVICE)
                    probs  = F.softmax(self.model(tensor), dim=1).cpu().numpy()[0]
                    accumulated += probs
                    n_passes    += 1
            else:
              
                for i in range(TTA_PASSES):
                    tf     = self._base_transform if i == 0 else self._aug_transform
                    tensor = tf(img_rgb).unsqueeze(0).to(DEVICE)
                    probs  = F.softmax(self.model(tensor), dim=1).cpu().numpy()[0]
                    accumulated += probs
                    n_passes    += 1

        avg_probs = accumulated / n_passes
        avg_probs = self._apply_grade1_penalty(avg_probs)

        predicted_idx = int(np.argmax(avg_probs))

        return {
            "grade"       : predicted_idx + 1,          
            "label"       : WAGNER_LABELS[predicted_idx],
            "description" : WAGNER_DESCRIPTIONS[predicted_idx],
            "confidence"  : round(float(avg_probs[predicted_idx]) * 100, 2),
            "all_probs"   : {
                WAGNER_LABELS[i]: round(float(avg_probs[i]) * 100, 2)
                for i in range(4)
            },
        }


class DFUPipeline:

    def __init__(self, stage1_path: str, stage2_path: str):
        self.stage1 = Stage1Detector(stage1_path)
        self.stage2 = Stage2Grader(stage2_path)
        print("\n[Pipeline] Both stages ready ✓\n")

    def predict(self, image_input) -> dict:
        image = self._load_image(image_input)

        # Stage 1
        s1 = self.stage1.predict(image)

        if not s1["is_ulcer"]:
            return {
                "has_ulcer"          : False,
                "stage1_confidence"  : s1["confidence"],
                "wagner_grade"       : None,
                "wagner_label"       : None,
                "wagner_description" : None,
                "stage2_confidence"  : None,
                "all_grade_probs"    : None,
                "summary"            : (
                    f"✅ No diabetic foot ulcer detected "
                    f"(confidence: {s1['confidence']}%).\n"
                    "The foot appears healthy."
                ),
            }
        # Stage 2
        s2 = self.stage2.predict(image)

        return {
            "has_ulcer"          : True,
            "stage1_confidence"  : s1["confidence"],
            "wagner_grade"       : s2["grade"],
            "wagner_label"       : s2["label"],
            "wagner_description" : s2["description"],
            "stage2_confidence"  : s2["confidence"],
            "all_grade_probs"    : s2["all_probs"],
            "summary"            : (
                f"⚠️  Diabetic foot ulcer detected (Stage 1 confidence: {s1['confidence']}%).\n"
                f"Wagner Classification → {s2['label']} "
                f"(confidence: {s2['confidence']}%)\n"
                f"Description: {s2['description']}"
            ),
        }

    @staticmethod
    def _load_image(image_input) -> Image.Image:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            return Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            return image_input
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

    def predict_path(self, path: str) -> dict:
        return self.predict(path)


    def batch_predict(self, folder: str, extensions=(".jpg", ".jpeg", ".png")) -> list:
        """Run pipeline on all images in a folder."""
        results = []
        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(extensions)
        ]
        print(f"[Batch] Found {len(files)} images in '{folder}'")

        for fname in files:
            path = os.path.join(folder, fname)
            try:
                result = self.predict(path)
                result["filename"] = fname
                results.append(result)
                print(f"  {fname:40s} → {result['summary'].splitlines()[0]}")
            except Exception as e:
                print(f"  {fname:40s} → ERROR: {e}")
                results.append({"filename": fname, "error": str(e)})

        return results


    def upload_and_predict(self):
        """Interactive upload widget for Google Colab."""
        try:
            from google.colab import files
        except ImportError:
            raise EnvironmentError("upload_and_predict() is only available in Google Colab.")

        uploaded = files.upload()
        for fname, data in uploaded.items():
            import io
            image = Image.open(io.BytesIO(data))
            result = self.predict(image)
            self._pretty_print(fname, result)

    def evaluate_pipeline(self, test_folder: str) -> dict:
      
        from collections import defaultdict

        correct = defaultdict(int)
        total   = defaultdict(int)

        grade_map = {"grade1": 1, "grade2": 2, "grade3": 3, "grade4": 4}

        for grade_folder, true_grade in grade_map.items():
            folder_path = os.path.join(test_folder, grade_folder)
            if not os.path.isdir(folder_path):
                print(f"[Eval] Skipping missing folder: {folder_path}")
                continue

            images = [f for f in os.listdir(folder_path)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            for img_file in images:
                path = os.path.join(folder_path, img_file)
                try:
                    result = self.predict(path)
                    pred_grade = result.get("wagner_grade")
                    total[true_grade] += 1
                    if pred_grade == true_grade:
                        correct[true_grade] += 1
                except Exception as e:
                    print(f"  Error on {img_file}: {e}")

        metrics = {}
        for g in range(1, 5):
            t = total[g]
            c = correct[g]
            acc = round(c / t * 100, 2) if t > 0 else None
            metrics[f"Grade {g}"] = {"correct": c, "total": t, "accuracy": acc}

        all_correct = sum(correct.values())
        all_total   = sum(total.values())
        metrics["Overall"] = {
            "correct" : all_correct,
            "total"   : all_total,
            "accuracy": round(all_correct / all_total * 100, 2) if all_total > 0 else None,
        }

        print("\n[Eval] Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v['correct']}/{v['total']} = {v['accuracy']}%")

        return metrics


    @staticmethod
    def _pretty_print(filename: str, result: dict):
        print(f"\n{'═'*55}")
        print(f"  File : {filename}")
        print(f"{'─'*55}")
        print(f"  {result['summary']}")
        if result.get("all_grade_probs"):
            print(f"\n  Grade probabilities:")
            for label, prob in result["all_grade_probs"].items():
                bar = "█" * int(prob / 5)
                print(f"    {label} : {prob:5.1f}%  {bar}")
        print(f"{'═'*55}\n")
