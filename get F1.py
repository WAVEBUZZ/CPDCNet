from sklearn.metrics import f1_score
def calculate_f1_score(true_labels, predicted_labels):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == 1 and predicted_label == 1:
            true_positives += 1
        elif true_label == 0 and predicted_label == 1:
            false_positives += 1
        elif true_label == 1 and predicted_label == 0:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score