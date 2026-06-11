import matplotlib.pyplot as plt
import numpy as np
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

    @property
    def name(self):
        return self.model.__class__.__name__

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

    def plot_confusion_matrix(self, labels=None, normalize=None):
        """
        normalize: None, 'true', 'pred', 'all'
        """

        cm = confusion_matrix(self.target_test, self._pred_test, normalize=normalize)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format=".2f" if normalize else "d")
        plt.title(f"Confusion Matrix - {self.name}")
        plt.show()

    def plot_roc_curve(self):
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
