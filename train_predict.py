import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import extract_v3
import categorize

def print_table_row(trait, prediction):
    """
    Print a row in the table with the personality trait and its prediction.

    Parameters:
        trait (str): The personality trait.
        prediction (list): The prediction for the trait.
    """
    prediction_str = ", ".join(map(str, prediction))
    print(f"| {trait:<30} | {prediction_str:^30} |")

X_baseline_angle = []
X_top_margin = []
X_letter_size = []
X_line_spacing = []
X_word_spacing = []
X_pen_pressure = []
X_slant_angle = []
y_t1 = []
y_t2 = []
y_t3 = []
y_t4 = []
y_t5 = []
y_t6 = []
y_t7 = []
y_t8 = []
page_ids = []

if os.path.isfile("label_list"):
    print("Info: label_list found.")
    # =================================================================
    with open("label_list", "r") as labels:
        for line in labels:
            content = line.split()

            baseline_angle = float(content[0])
            X_baseline_angle.append(baseline_angle)

            top_margin = float(content[1])
            X_top_margin.append(top_margin)

            letter_size = float(content[2])
            X_letter_size.append(letter_size)

            line_spacing = float(content[3])
            X_line_spacing.append(line_spacing)

            word_spacing = float(content[4])
            X_word_spacing.append(word_spacing)

            pen_pressure = float(content[5])
            X_pen_pressure.append(pen_pressure)

            slant_angle = float(content[6])
            X_slant_angle.append(slant_angle)

            trait_1 = float(content[7])
            y_t1.append(trait_1)

            trait_2 = float(content[8])
            y_t2.append(trait_2)

            trait_3 = float(content[9])
            y_t3.append(trait_3)

            trait_4 = float(content[10])
            y_t4.append(trait_4)

            trait_5 = float(content[11])
            y_t5.append(trait_5)

            trait_6 = float(content[12])
            y_t6.append(trait_6)

            trait_7 = float(content[13])
            y_t7.append(trait_7)

            trait_8 = float(content[14])
            y_t8.append(trait_8)

            page_id = content[15]
            page_ids.append(page_id)
    # ===============================================================

    # emotional stability
    X_t1 = []
    for a, b in zip(X_baseline_angle, X_slant_angle):
        X_t1.append([a, b])

    # mental energy or will power
    X_t2 = []
    for a, b in zip(X_letter_size, X_pen_pressure):
        X_t2.append([a, b])

    # modesty
    X_t3 = []
    for a, b in zip(X_letter_size, X_top_margin):
        X_t3.append([a, b])

    # personal harmony and flexibility
    X_t4 = []
    for a, b in zip(X_line_spacing, X_word_spacing):
        X_t4.append([a, b])

    # lack of discipline
    X_t5 = []
    for a, b in zip(X_slant_angle, X_top_margin):
        X_t5.append([a, b])

    # poor concentration
    X_t6 = []
    for a, b in zip(X_letter_size, X_line_spacing):
        X_t6.append([a, b])

    # non communicativeness
    X_t7 = []
    for a, b in zip(X_letter_size, X_word_spacing):
        X_t7.append([a, b])

    # social isolation
    X_t8 = []
    for a, b in zip(X_line_spacing, X_word_spacing):
        X_t8.append([a, b])

    # print X_t1
    # print type(X_t1)
    # print len(X_t1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_t1, y_t1, test_size=.30, random_state=8)
    clf1 = SVC(kernel='rbf')
    clf1.fit(X_train, y_train)
    print("SVM Classifier 1 [emotional stability] accuracy: ", accuracy_score(clf1.predict(X_test), y_test))

    X_train, X_test, y_train, y_test = train_test_split(
        X_t2, y_t2, test_size=.30, random_state=16)
    clf2 = SVC(kernel='rbf')
    clf2.fit(X_train, y_train)
    print("SVM Classifier 2 [mental energy or will power] accuracy: ", accuracy_score(clf2.predict(X_test), y_test))

    X_train, X_test, y_train, y_test = train_test_split(
        X_t3, y_t3, test_size=.30, random_state=32)
    clf3 = SVC(kernel='rbf')
    clf3.fit(X_train, y_train)
    print("SVM Classifier 3 [modesty] accuracy: ", accuracy_score(clf3.predict(X_test), y_test))

    X_train, X_test, y_train, y_test = train_test_split(
        X_t4, y_t4, test_size=.30, random_state=64)
    clf4 = SVC(kernel='rbf')
    clf4.fit(X_train, y_train)
    print("SVM Classifier 4 [personal harmony and flexibility] accuracy: ", accuracy_score(clf4.predict(X_test), y_test))

    X_train, X_test, y_train, y_test = train_test_split(
        X_t5, y_t5, test_size=.30, random_state=42)
    clf5 = SVC(kernel='rbf')
    clf5.fit(X_train, y_train)
    print("SVM Classifier 5 [lack of discipline] accuracy: ", accuracy_score(clf5.predict(X_test), y_test))

    X_train, X_test, y_train, y_test = train_test_split(
        X_t6, y_t6, test_size=.30, random_state=52)
    clf6 = SVC(kernel='rbf')
    clf6.fit(X_train, y_train)
    print("SVM Classifier 6 [Poor Concentration] accuracy: ", accuracy_score(clf6.predict(X_test), y_test))

    X_train, X_test, y_train, y_test = train_test_split(
        X_t7, y_t7, test_size=.30, random_state=21)
    clf7 = SVC(kernel='rbf')
    clf7.fit(X_train, y_train)
    print("SVM Classifier 7 [non communicativeness] accuracy: ", accuracy_score(clf7.predict(X_test), y_test))

    X_train, X_test, y_train, y_test = train_test_split(
        X_t8, y_t8, test_size=.30, random_state=73)
    clf8 = SVC(kernel='rbf')
    clf8.fit(X_train, y_train)
    print("SVM Classifier 8 [Social Isolation] accuracy: ", accuracy_score(clf8.predict(X_test), y_test))

    # ================================================================================================

def print_table_row(trait, prediction):
    HEADER_COLOR = '\033[95m'  # Purple color for headers
    OKGREEN = '\033[92m'       # Green color for positive predictions
    FAIL = '\033[91m'          # Red color for negative predictions
    ENDC = '\033[0m'           # Reset color

    # Determine the symbol and color based on the prediction value
    if prediction[0] == 0:
        symbol = "🟥"   # Red ball for 0
        color = FAIL    # Red color for 0
    else:
        symbol = "🟩"   # Green ball for 1
        color = OKGREEN # Green color for 1
        HEADER_COLOR = OKGREEN

    # Print the table row with color and symbol
    print(f"| {HEADER_COLOR}{trait:^40}{ENDC} | {color}{symbol:^38}{ENDC} |")
    

# def print_table_row(trait, prediction):
#     HEADER_COLOR = '\033[95m'  # Purple color for headers
#     OKGREEN = '\033[92m'       # Green color for positive predictions
#     FAIL = '\033[91m'          # Red color for negative predictions
#     ENDC = '\033[0m'           # Reset color

#     # Determine the color based on the prediction value
#     if prediction[0] == 0:
#         color = FAIL  # Red color for 0
#     else:
#         color = OKGREEN  # Green color for 1

#     # Print the table row with color
#     print(f"| {HEADER_COLOR}{trait:<40}{ENDC} | {color}{prediction[0]:^40}{ENDC} |")




def print_table_row_1(trait, value):
    print(f"| {trait:<40} | {value:^40} |")

while True:
    file_name = input("Enter file name to predict or z to exit: ")
    if file_name == 'z':
        break

    raw_features = extract_v3.start(file_name)

    print("+------------------------------------------+------------------------------------------+")
    raw_baseline_angle = raw_features[0]
    baseline_angle, comment = categorize.determine_baseline_angle(raw_baseline_angle)
    print_table_row_1("Baseline Angle", comment)

    raw_pen_pressure = raw_features[5]
    pen_pressure, comment = categorize.determine_pen_pressure(raw_pen_pressure)
    print_table_row_1("Pen Pressure", comment)

    raw_word_spacing = raw_features[4]
    word_spacing, comment = categorize.determine_word_spacing(raw_word_spacing)
    print_table_row_1("Word Spacing", comment)

    raw_line_spacing = raw_features[3]
    line_spacing, comment = categorize.determine_line_spacing(raw_line_spacing)
    print_table_row_1("Line Spacing", comment)

    raw_top_margin = raw_features[1]
    top_margin, comment = categorize.determine_top_margin(raw_top_margin)
    print_table_row_1("Top Margin", comment)

    raw_letter_size = raw_features[2]
    letter_size, comment = categorize.determine_letter_size(raw_letter_size)
    print_table_row_1("Letter Size", comment)

    raw_slant_angle = raw_features[6]
    slant_angle, comment = categorize.determine_slant_angle(raw_slant_angle)
    print_table_row_1("Slant", comment)

    print("+------------------------------------------+------------------------------------------+")
    print()
    print("\n\033[93m+------------------------------------------+------------------------------------------+")
    print("|            Personality Trait             |               Prediction                 |")
    print("+------------------------------------------+------------------------------------------+\033[0m")

    print_table_row("Emotional Stability", clf1.predict([[baseline_angle, slant_angle]]))
    print_table_row("Mental Energy or Will Power", clf2.predict([[letter_size, pen_pressure]]))
    print_table_row("Modesty", clf3.predict([[letter_size, top_margin]]))
    print_table_row("Personal Harmony and Flexibility", clf4.predict([[line_spacing, word_spacing]]))
    print_table_row("Lack of Discipline", clf5.predict([[slant_angle, top_margin]]))
    print_table_row("Poor Concentration", clf6.predict([[letter_size, line_spacing]]))
    print_table_row("Non Communicativeness", clf7.predict([[letter_size, word_spacing]]))
    print_table_row("Social Isolation", clf8.predict([[line_spacing, word_spacing]]))
    print("---------------------------------------------------------------------------------------")
    print()

else:
    print("Error: label_list file not found.")


