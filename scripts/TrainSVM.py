from libsvm.svmutil import *

# ---load training data---
y, x = svm_read_problem("model5/training.txt")

# ---train SVM---
model = svm_train(y, x)

# ---save model---
svm_save_model("model5/face_svm.model", model)

print("Training finished")
print("Model saved as face_svm.model")