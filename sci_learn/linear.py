def linear():
    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    reg.coef_


def decision_tree():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # 加载鸢尾花数据集
    data = load_iris()
    X, y = data.data, data.target

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建决策树分类器
    clf = DecisionTreeClassifier(random_state=42)

    # 训练模型
    clf.fit(X_train, y_train)

    # 进行预测
    y_pred = clf.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)


if __name__ == '__main__':


    decision_tree()
