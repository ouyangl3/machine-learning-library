from sklearn.svm import SVC

X = [[-1, 0], [0, -1], [1, 0], [0, 1]]
y = [-1, -1, 1, 1]

clf = SVC(kernel='linear', C=1e10)
clf.fit(X, y)

w = clf.coef_[0]
margin = 1 / (sum(w**2) ** 0.5)

print("margin:", margin)