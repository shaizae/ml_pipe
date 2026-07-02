import datetime
import os.path
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score, recall_score,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


class Results:
    def __init__(self, model):
        self.model = model
        self.features_test = None
        self.target_test = None
        self._pred_test = None
        self._predict_score = None
        self._accuracy: float = 0
        self._precision: float = 0
        self._recall: float = 0
        self._f1: float = 0

    @property
    def name(self):
        return self.model.__class__.__name__

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def precision(self):
        return self._precision

    @property
    def recall(self):
        return self._recall

    @property
    def f1(self):
        return self._f1

    def set_test(self, features_test, target_test):
        self.features_test = features_test
        self.target_test = target_test
        self.predict()

    def predict(self):
        self._pred_test = self.model.predict(self.features_test)
        if not hasattr(self.model, "predict_proba"):
            print(f"model {self.name} must support predict_proba for ROC curve")
            return
        self._predict_score = self.model.predict_proba(self.features_test)
        self._accuracy = accuracy_score(self.target_test, self._pred_test)
        self._precision = precision_score(self.target_test, self._pred_test, average="weighted")
        self._recall = recall_score(self.target_test, self._pred_test, average="weighted")
        self._f1 = f1_score(self.target_test, self._pred_test, average="weighted")

    def plot_confusion_matrix(self, labels=None, normalize=None, show: bool = True):
        """
        normalize: None, 'true', 'pred', 'all'
        """

        cm = confusion_matrix(self.target_test, self._pred_test, normalize=normalize)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format=".2f" if normalize else "d")
        plt.title(f"Confusion Matrix - {self.name}")
        if show:
            plt.show()

    def plot_roc_curve(self, show: bool = True):
        """
        Works for binary classification.
        For multiclass, you’d need binarization (handled below).
        """
        if self._predict_score is None:
            return
        classes = np.unique(self.target_test)

        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(self.target_test, self._predict_score[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {self.name}")
            plt.legend()
            if show:
                plt.show()
            return

        y_test_bin = label_binarize(self.target_test, classes=classes)

        plt.figure()

        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], self._predict_score[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Multiclass ROC - {self.name}")
        plt.legend()
        if show:
            plt.show()

    def append_results(self, result: Results):
        """
        Merge another Results object into this one.
        """
        # merge test features
        if result.features_test is not None:
            if self.features_test is None:
                self.features_test = result.features_test.copy()
            else:
                self.features_test = np.concatenate(
                    [self.features_test, result.features_test],
                    axis=0
                )

        if result.target_test is not None:
            if self.target_test is None:
                self.target_test = result.target_test.copy()
            else:
                self.target_test = np.concatenate(
                    [self.target_test, result.target_test],
                    axis=0
                )

        if result._pred_test is not None:
            if self._pred_test is None:
                self._pred_test = result._pred_test.copy()
            else:
                self._pred_test = np.concatenate(
                    [self._pred_test, result._pred_test],
                    axis=0
                )

        if result._predict_score is not None:
            if self._predict_score is None:
                self._predict_score = result._predict_score.copy()
            else:
                self._predict_score = np.concatenate(
                    [self._predict_score, result._predict_score],
                    axis=0
                )

        return self

    def save_pdf_report(self, filename: str):
        filename = os.path.join(filename,
                                f"report{self.name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pdf")
        if self._pred_test is None:
            raise ValueError("No predictions available")

        styles = getSampleStyleSheet()

        doc = SimpleDocTemplate(filename)
        elements = []

        elements.append(
            Paragraph(f"Classification Report - {self.name}",
                      styles["Title"])
        )

        elements.append(Spacer(1, 12))

        elements.append(
            Paragraph(f"Accuracy: {self.accuracy:.4f}", styles["BodyText"])
        )
        elements.append(
            Paragraph(f"Precision: {self.precision:.4f}", styles["BodyText"])
        )
        elements.append(
            Paragraph(f"Recall: {self.recall:.4f}", styles["BodyText"])
        )
        elements.append(
            Paragraph(f"F1 Score: {self.f1:.4f}", styles["BodyText"])
        )

        elements.append(Spacer(1, 20))

        with TemporaryDirectory() as tmpdir:

            cm_path = Path(tmpdir) / "cm.png"

            self.plot_confusion_matrix(show=False)
            plt.savefig(cm_path, bbox_inches="tight")
            plt.close()

            elements.append(
                Paragraph("Confusion Matrix", styles["Heading2"])
            )
            elements.append(Image(str(cm_path), width=400, height=300))

            if self._predict_score is not None:
                roc_path = Path(tmpdir) / "roc.png"

                self.plot_roc_curve(show=False)
                plt.savefig(roc_path, bbox_inches="tight")
                plt.close()

                elements.append(Spacer(1, 20))
                elements.append(
                    Paragraph("ROC Curve", styles["Heading2"])
                )
                elements.append(
                    Image(str(roc_path), width=400, height=300)
                )

            doc.build(elements)
