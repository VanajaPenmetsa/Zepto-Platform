import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def train_model():
    df = pd.read_csv('analytics/titanic.csv').dropna(subset=['embarked'])
    df['age'] = df['age'].fillna(df['age'].median())

    X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
    y = df['survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = ['age', 'fare', 'sibsp', 'parch']
    categorical_features = ['pclass', 'sex', 'embarked']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 5, 10]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'analytics/zepto_titanic_pipeline.joblib')
    
    accuracy = best_model.score(X_test, y_test)
    print(f"Model trained successfully! Test Accuracy: {accuracy * 100:.2f}%")
    print("Saved model artifact to analytics/zepto_titanic_pipeline.joblib")

if __name__ == "__main__":
    train_model()