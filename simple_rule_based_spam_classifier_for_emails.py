import numpy as np
def email_features(email):
# Extracts features from an email address.
    return np.array([ len(email) > 10, sum(c.isalpha() for c in email) / len(email), # fraction of alphabetic characters
'@' in email, email.endswith('.com'), email.endswith('.org') ], dtype=float)
def linear_classifier(email, weights):
    features = email_features(email)
    score = np.dot(features, weights)
    return 1 if score > 0 else 0
# Weights for each feature
weights = np.array([-1.2, 0.6, 3, 2.2, 1.4])
# Example email addresses
emails = [
"john@example.com",
"spam@offers.org",
"noatsign.com",
"short@org"
,"valid.email@example.com"]
# Classify each email
for email in emails:
    print(f"{email}: {'Spam' if linear_classifier(email, weights) else 'Not Spam'}")