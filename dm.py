
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

classes = {
    "German Styles": ["Oktoberfest/Märzen", "Altbier", "Dark Mild", "Doppelbock", "Dusseldorf Altbier", "Eisbock", "Festbier", "Kölsch", "Märzen", "Mild", "Roggenbier", "Weizen/Weissbier"],
    "Belgian and French Styles": ["Straight (Unblended) Lambic", "Saison", "Belgian Blond Ale", "Belgian Dubbel", "Belgian Specialty Ale", "Bière de Garde", "Flanders Brown Ale/Oud Bruin", "Flanders Red Ale", "Fruit Lambic", "Gueuze", "Lambic", "Oud Bruin", "Saison Straight (Unblended) Lambic", "Trappist Single"],
    "Ciders and Meads": ["Cyser (Apple Melomel)", "Apple Wine", "Common Cider", "Cyster (Apple Melomel)", "Dry Mead", "English Cider", "French Cider", "Metheglin", "New England Cider", "Open Category Mead", "Other Fruit Melomel", "Pyment (Grape Melomel)", "Semi-Sweet Mead", "Sweet Mead", "Traditional Perry"],
    "Dark Ales": ["Tropical Stout", "Brown Porter", "Dry Stout", "American Brown Ale", "American Porter", "American Stout", "Baltic Porter", "British Brown Ale", "Brow Porter", "English Porter", "Foreign Extra Stout", "Irish Extra Stout", "Irish Stout", "London Brown Ale", "Northern English Brown", "Oatmeal Stout", "Robust Porter", "Russian Imperial Stout", "Southern English Brown", "Sweet Stout", "Tropical Stou"],
    "Pale Ales": ["Ordinary Bitter", "Standard/Ordinary Bitter", "Irish Red Ale", "Zombie Dust Clone - EXTRACT", "American Amber Ale", "American IPA", "American Pale Ale", "Belgian Pale Ale", "Blonde Ale", "British Golden Ale", "Double IPA", "English IPA", "Imperial IPA", "Specialty IPA: Belgian IPA", "Specialty IPA: Black IPA", "Specialty IPA: Brown IPA", "Specialty IPA: Red IPA", "Specialty IPA: Rye IPA", "Specialty IPA: White IPA"],
    "Hybrid and Specialty": ["Vanilla Cream Ale", "Alternative Grain Beer", "Alternative Sugar Beer", "Braggot", "Brett Beer", "California Common", "Gose", "Mixed-Fermentation Sour Beer", "Sahti", "Wild Specialty Beer"],
    "Lagers and Bocks": ["International Dark Lager", "International Dark Lager", "International Amber Lager", "Czech Premium Pale Lager", "International Pale Lager", "American Light Lager", "Traditional Bock", "Bohemian Pilsener", "American Lager", "American Light Lager Bohemian Pilsner", "California Common Beer", "Classic American Pilsner", "Classic Rauchbier", "Classic Style Smoked Beer", "Cream Ale", "Czech Amber Lager", "Czech Dark Lager", "Czech Pale Lager", "Czech premium Pale Lager", "Dark American Lager", "Dortmunder Export", "Dunkles Bock", "German Helles Exportbier", "German Leichtbier", "German Pils", "German Pilsner (Pils)", "Helles Bock", "International Pales Lager", "Kellerbier: Amber Kellerbier", "Kellerbier: Pale Kellerbier", "Kentucky Common", "Light American Lager", "Maibock/Helles Bock", "Munich Dunkel", "Munich Helles", "North German Altbier", "Piwo Grodziskie", "Pre-Prohibition Lager", "Pre-Prohibition Porter", "Premium American Lager", "Rauchbier", "Schwarzbier", "Scottish Export", "Scottish Export 80/-", "Scottish Heavy", "Scottish Heavy 70/-", "Scottish Light", "Scottish Light 60/-", "Standard American Lager", "Vienna Lager"],
    "Seasonal and Specialty": ["Spice Herb or Vegetable Beer", "Southern Tier Pumking clone", "Australian Sparkling Ale", "Autumn Seasonal Beer", "Clone Beer", "Experimental Beer", "Fruit and Spice Beer", "Fruit Beer", "Fruit Cider", "Holiday/Winter Special Spiced Beer", "Lichtenhainer", "Mixed-Style Beer", "Other Smoked Beer", "Other Specialty Cider or Perry", "Specialty Beer", "Specialty Fruit Beer", "Specialty Smoked Beer", "Specialty Wood-Aged Beer", "Winter Seasonal Beer", "Wood-Aged Beer"],
    "Strong Ales": ["Special/Best/Premium Bitter", "American Barleywine", "American Strong Ale", "Belgian Dark Strong Ale", "Belgian Golden Strong Ale", "Belgian Tripel", "Best Bitter", "British Strong Ale", "English Barleywine", "Extra Special/Strong Bitter (ESB)", "Imperial Stout", "Old Ale", "SPecial/Best/Premium Bitter", "Strong Bitter", "Strong Scotch Ale", "Wee Heavy"],
    "Wheat and Rye Beers": ["Dunkles Weissbier", "Wheatwine", "Witbier", "American Wheat Beer", "American Wheat or Rye Beer", "Berliner Weisse", "Dunkelweizen", "Roggenbier (German Rye Beer)", "Weissbier", "Weizenbock", "WheatWine", "Wittbie"]
}

def findclass(subgenre):
    for key, values in classes.items():
        if subgenre in values:
            return key
    return 'N/A'

def standardizing(df, column):
    mean = df[column].mean()
    std = df[column].std()
    standardized_column = (df[column] - mean) / std
    return standardized_column

def convert_nominal_to_numerical(df, column, method='one-hot'):
    if method == 'label':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    elif method == 'one-hot':
        df = pd.get_dummies(df, columns=[column], prefix=[column])
        for col in df:
            if column in col:
                df[col] = df[col].astype(int)
    else:
        raise ValueError("Method should be either 'label' or 'one-hot'")
    return df

# Load the dataset from a specified directory path
df = pd.read_csv('/Users/olgagranecka/Documents/studia/4sem/data mining/modified_2.csv')

# Check for missing values
print(df.isna().sum())
class_counts = df["Genre"].value_counts()
print("Number of samples for each class:")
print(class_counts)

# Convert nominal to numerical
df = convert_nominal_to_numerical(df, "SugarScale")
df = convert_nominal_to_numerical(df, "BrewMethod", "label")
df = convert_nominal_to_numerical(df, "Genre", "label")

# Remove certain classes if necessary
df = df[(df['Genre'] != 4) & (df["Genre"] != 1)]
df.drop(columns=["Style"], inplace=True)


# Prepare features and target
x = df.drop('Genre', axis=1)
y = df['Genre']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


def most_confused(con):
    for genre in range(len(con)):
        classification = list(con[genre])
        sample_size = sum(classification)
        sorted = list(con[genre])
        sorted.sort(reverse=True)
        print("Genre no. ",genre,"\tSamples in test set: ",sample_size,"\nMost often confused with:")
        for i in range(3):
            print(i+1,". Genre: ", classification.index(sorted[i+1])," samples: ",sorted[i+1], "[",round(sorted[i+1]/sample_size*100,2),"%]")
    return None

 #RANDOM FOREST
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)
print("Random Forest:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred)}")
most_confused(confusion_matrix(y_test, y_pred))
