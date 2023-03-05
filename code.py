import pandas as pd
df = pd.read_csv("titanc.csv") 
print (df.head())
col_names=["pregnant","gulucose","BP","skin","insulin","BMI"]
features = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X=df[features]
y=df.label
from sklearn.tree import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train,x_test,y_train,y_test = train_test_split(X.y,x_size=0.3,random_states=1)
clf=DecisionTree()
clf=clf.fit(x_train,y_train)
y_predicted=clf.predict(x_test)
accuracy=metrics.accuracy_score(y_test,y_predicted)
print(accuracy)
dot_data = StringIO()
export_graphviz(clf, out_file = dot_data, filled = True, rounded = True,
                special_characters = True, feature_names = features, class_names = ['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())
