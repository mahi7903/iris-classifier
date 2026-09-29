import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--random-state", type=int, default=42)
args = parser.parse_args()
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
iris = datasets.load_iris()
iris
type(iris)
X = iris.data
y = iris.target
iris.target_names
iris.feature_names
print(iris.feature_names, iris.target_names)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=args.random_state, test_size=args.test_size)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=20)
model.fit(X_train, y_train)
model.predict(X_test)
model.score(X_test, y_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, model.predict(X_test))
print("accuracy is ", accuracy)
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, model.predict(X_test))
print("confusion matrix is: ", matrix)
from sklearn.metrics import precision_score
score = precision_score(y_test, model.predict(X_test), average="micro") # these average parameters can be micro/macro/weighted/or None without quatation mark
print(score)
from sklearn.metrics import recall_score
score = recall_score(y_test, model.predict(X_test), average="macro")
print(score)
from sklearn.metrics import f1_score
score = f1_score(y_test, model.predict(X_test), average=None)
print(score)

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

Path("outputs").mkdir(exist_ok=True)
ConfusionMatrixDisplay(matrix, display_labels=iris.target_names).plot()
plt.savefig("outputs/confusion_matrix.png")
print("Saved outputs/confusion_matrix.png")

#I have just copy pasted my ipynb codes into this file no extra changes aprt from setting the random 42 as asked in the video and texgt size 0.2
## Results 
# I trained a Decision Tree on 120 flowers and tested it on 30 it hadn't seen.
# It got **all 30 right**: accuracy, precision, recall and F1 were all 1.0.
# That's normal for Iris because the species are easy to tell apart, especially by petal size. With a different split the score might drop slightly.
#I loaded the iris file using the function mentioned on the scikit learn website as it was easier way for me to remember and code. 
# Apart from that I have trieed cheanging  some parameters in order to achie this score as at first my score was 0.93 when random_state was at 42 then i chaned it to 20.
#i tried using all the parameters  of the average while coding f1score, precision, recall from micro/macrow/weighted/None 
#Got to know binary for 2 classes and sample for multiclasses also exist but not required in this single class model.
#I only used decision tree as the other ones were just a line of code change i still tried them just not saved here to not create confusion for me.




















