# To be implemented
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def prep(X_train, X_test, y_train, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    pca = PCA()
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    explained_variance = pca.explained_variance_ratio 

def decision_tree(X_train, X_test, y_train, y_test):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)