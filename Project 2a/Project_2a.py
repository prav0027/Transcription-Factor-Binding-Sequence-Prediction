from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def read_reads(read_fn):
    all_reads = []
    with open(read_fn, 'r') as f:
        lineString = ""
        for line in f:
            if '>' in line:
                if lineString!="":
                    all_reads.append(lineString)
                lineString = ""
                continue
            lineString+=line.strip()
            #line = line.strip()
            #all_reads.append(line)
    return all_reads

# Load the DNA sequences
sequencesArr = read_reads('sequences_og.txt')
sequences = np.array(sequencesArr)
#sequences = np.loadtxt('sequences_2b.txt', dtype='str')

notboundArr = read_reads('notbound_og.txt')
notboundseqs = np.array(notboundArr)
#notboundseqs = np.loadtxt('notboundseqs_2b.txt', dtype='str')
# Extract k-mer frequencies
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
X = vectorizer.fit_transform(sequences).toarray()
x_label = np.ones(len(X))

#vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
X_unbound = vectorizer.fit_transform(notboundseqs).toarray()
x_label_unbound = np.zeros(len(X_unbound))

combined_X = np.concatenate((X, X_unbound), axis = 0)
combined_y_labels = np.concatenate((x_label, x_label_unbound), axis = 0)

# Save the k-mer frequencies as a CSV file
np.savetxt('features.csv', X, delimiter=',')

# Load the data
X = combined_X
y = combined_y_labels

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)

# Train the logistic regression model
lr = LogisticRegression(max_iter=2000, class_weight = 'balanced')
model = lr.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = lr.predict(X_val)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy:', accuracy)


##test
sequences3Arr = read_reads('testseqs_og.txt')
sequences3 = np.array(sequences3Arr)
#sequences3 = np.loadtxt('testseqs_2b.txt', dtype='str')
#vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
X_p = vectorizer.fit_transform(sequences3).toarray()


probabilities = lr.predict_proba(X_p)
li = []
for i in range(len(probabilities)):
    if probabilities[i][1] > 0.3: #lol it was making like no predictions at one point idk when i changed it to this
        li.append([probabilities[i][0], i]) #append prob y = 0 and sort from smallest to largest
        
li.sort()

printout = '\n'.join(f'seq{li[i][1]+1}' for i in range(2000))

def save_results(fileName = 'results.csv'):
	f = open(fileName, 'w')
	f.write(printout)
	f.close()
	return

save_results()

answer = lr.predict(X_p)

printout = '\n'.join([f'seq{i}' for i in range(len(answer)) if answer[i] != 0])
print(printout)
print(len(answer))

