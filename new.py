import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assuming you have already trained your SVM model and obtained the predicted labels and true labels
true_labels = np.array([1, 0, 1, 1, 0, 0, 1])
predicted_labels = np.array([1, 0, 0, 1, 0, 1, 1])

# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Display the confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Negative', 'Positive'])
plt.yticks(tick_marks, ['Negative', 'Positive'])
plt.xlabel("Predicted class/performance")
plt.ylabel("True class/performance")

# Add labels to each cell in the confusion matrix
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", 
                 color="white" if cm[i, j] > thresh else "black")

plt.show()
