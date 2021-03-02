import csv
import matplotlib.pyplot as plt 

loss = []
val_loss = []
acc = []
val_acc = []

with open('result_show/result2_6/trash-classification4-result.csv','r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        loss.append(float(row['loss']))
        val_loss.append(float(row['val_loss']))
        acc.append(float(row['acc']))
        val_acc.append(float(row['val_acc']))



epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'mediumslateblue', label='Traning_Loss', marker='.', linestyle='-')
plt.plot(epochs, val_loss, 'deeppink', label='Validation_loss', marker='.', linestyle='-')
plt.title('Traning_Loss and Validation_loss')
plt.legend()
plt.grid()

plt.figure()
plt.plot(epochs, acc, 'mediumslateblue', label='Traning_accuracy', marker='.', linestyle='-')
plt.plot(epochs, val_acc, 'deeppink', label='Validation_accuracy', marker='.', linestyle='-')
plt.title('Traning_Accuracy and Validation_Accuracy')
plt.legend(loc='lower right')
plt.grid()
plt.show()