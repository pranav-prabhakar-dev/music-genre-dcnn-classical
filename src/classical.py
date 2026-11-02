"""The four classical classifiers from the paper, wrapped as sklearn pipelines."""
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C

ALGORITHMS = ["LR", "GDA", "RF", "SVM"]


def make_classifier(name: str, probability: bool = False, svm_gamma="scale"):
    if name == "LR":     # softmax (multinomial) logistic regression
        return LogisticRegression(max_iter=5000, C=1.0)
    if name == "GDA":    # Gaussian class-conditionals with a shared covariance = LDA
        return LinearDiscriminantAnalysis(solver="svd")
    if name == "RF":     # Gini splits, max depth 7 as in the paper
        return RandomForestClassifier(n_estimators=300, max_depth=7, criterion="gini",
                                      random_state=C.SEED, n_jobs=-1)
    if name == "SVM":    # RBF kernel, gamma = 1 / (n_features * Var(X)), one-vs-one multiclass
        # `probability` is deprecated in sklearn>=1.9, so only pass it when the demo needs it
        extra = {"probability": True} if probability else {}
        return SVC(kernel="rbf", gamma=svm_gamma, C=1.0, decision_function_shape="ovo",
                   random_state=C.SEED, **extra)
    raise ValueError(name)


def make_pipeline_for(name: str, pca_components: int | None = None,
                      probability: bool = False, scale: bool = True, svm_gamma="scale"):
    """StandardScaler -> [PCA] -> classifier. `probability` enables SVM predict_proba (for the demo)."""
    steps = [StandardScaler()] if scale else []
    if pca_components:
        steps.append(PCA(n_components=pca_components, random_state=C.SEED))
    steps.append(make_classifier(name, probability, svm_gamma))
    return make_pipeline(*steps)
