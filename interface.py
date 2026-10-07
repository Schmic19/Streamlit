import streamlit as st
st.title("My Machine Learning Classifier App")

st.write("""
### Explore different classifier
""")

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Iris", "Breast Cancer", "Wine")
)

classifier_name = st.sidebar.selectbox(
    "Select Classifier",
    ("KNN", "SVM", "Random Forest")
)

#get the dataset
def get_dataset(name):
    from sklearn import datasets
    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y
X, y = get_dataset(dataset_name)

st.write("Shape of dataset:", X.shape)
st.write("Number of classes:", len(set(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        leaf_size = st.sidebar.slider("Leaf Size", 1, 50)
        params["leaf_size"] = leaf_size
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        degree = st.sidebar.slider("Degree", 1, 5)
        params["C"] = C
        params["degree"] = degree
    else:
        max_depth = st.sidebar.slider("Max Depth", 2, 15)
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 100)
        min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["min_samples_split"] = min_samples_split
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"], leaf_size=params["leaf_size"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"], degree=params["degree"])
    else:
        clf = RandomForestClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"], min_samples_split=params["min_samples_split"])
    return clf

clf = get_classifier(classifier_name, params)


#splitting our dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training the classifier
clf.fit(X_train, y_train)

#evaluating the classifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write("Classifier accuracy:", accuracy)


st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

#PLOT
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

st.write("PCA Plot:")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
scatter = ax.scatter(x1, x2, c=y, cmap="viridis")
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
st.pyplot(fig)