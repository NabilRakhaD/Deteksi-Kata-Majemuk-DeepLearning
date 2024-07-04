import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Values
TP = 22
TN = 10
FP = 1
FN = 0

# Creating a confusion matrix array
cm = np.array([[TP, FP],
               [FN, TN]])

# Plot confusion matrix
plt.figure(figsize=(8,5))
plt.matshow(cm, cmap='Blues', fignum=1)
plt.title('Confusion Matrix With Articles ')
plt.colorbar()
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')

# Add text annotations with white font color
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > cm.max()/2 else "black"
        plt.text(j, i, cm[i, j], ha="center", va="center", color=color)

plt.show()



"""# import MWETokenizer() method from nltk 
from nltk.tokenize import MWETokenizer 
   
# Create a reference variable for Class MWETokenizer 
tk = MWETokenizer([('g', 'f', 'g'), ('geeks', 'for', 'geeks')]) 
tk.add_mwe(('at', 'laa')) 
   
# Create a string input 
gfg = "who are youu at la geeks for geeks"
   
# Use tokenize method 
geek = tk.tokenize(gfg.split()) 
   
print(geek)"""
