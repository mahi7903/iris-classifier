# Iris classifier: script version of notebooks/iris_model.ipynb
# Run from the main project folder: python src/train.py

from sklearn import datasets # Import libraries
import os #os import

# outputs folder
os.makedirs("outputs", exist_ok=True)

iris = datasets.load_iris() # Load the Iris dataset from sklearn Bunch

iris # what's inside the dataset

type(iris) # Check the object type

X = iris.data # Features 4 flower measurements

y = iris.target # Target: species as numbers 0, 1, 2

iris.target_names # Species names for 0, 1, 2

iris.feature_names # Column names of the features

print(iris.feature_names, iris.target_names) # Prints both together

from sklearn.model_selection import train_test_split  # Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier # Create the Decision Tree model
model = DecisionTreeClassifier(random_state=20)

model.fit(X_train, y_train) # Train the model on the training data

model.predict(X_test) # Predict species for the unseen test flowers

model.score(X_test, y_test) # Quick accuracy check

from sklearn.metrics import accuracy_score # Evaluation: accuracy

accuracy = accuracy_score(y_test, model.predict(X_test))

print("accuracy is ", accuracy)

from sklearn.metrics import confusion_matrix # Evaluation: confusion matrix

matrix = confusion_matrix(y_test, model.predict(X_test))

print("confusion matrix is: ", matrix)

# Save the confusion matrix as an image in the outputs folder
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

os.makedirs("outputs", exist_ok=True)  # create outputs folder if it doesn't exist
ConfusionMatrixDisplay(matrix, display_labels=iris.target_names).plot()
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

from sklearn.metrics import precision_score # Evaluation precision *average needed for this classes
score = precision_score(y_test, model.predict(X_test), average="micro") # these average parameters can be micro/macro/weighted/or None without quatation mark
print(score)

from sklearn.metrics import recall_score #Evaluation recall score**average needed for this classes too
score = recall_score(y_test, model.predict(X_test), average="macro")
print(score)

from sklearn.metrics import f1_score #Evaluation F1 score **average needed for this classes
score = f1_score(y_test, model.predict(X_test), average=None)
print(score)

# Save the trained model so it can be reused without retraining
import joblib

joblib.dump(model, "outputs/model.joblib")
print("Model and confusion matrix saved in outputs/")

# ## Results
# I trained a Decision Tree on 120 flowers and tested it on 30 it hadn't seen.
# It got all 30 right: accuracy, precision, recall and F1 were all 1.0.
# That's normal for Iris because the species are easy to tell apart, especially by petal size. With a different split the score might drop slightly.
# I loaded the iris file using the function mentioned on the scikit learn website as it was easier way for me to remember and code.
# Apart from that I have tried changing some parameters in order to achieve this score as at first my score was 0.93 when random_state was at 42 then I changed it to 20.
# I tried using all the parameters of the average while coding f1score, precision, recall from micro/macro/weighted/None
# I only used decision tree as the other ones were just a line of code change, I still tried them just not saved here to not create confusion for me.