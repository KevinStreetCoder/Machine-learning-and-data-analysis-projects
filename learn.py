from sklearn import datasets

digits = datasets.load_digits()
features = digits.data
target = digits.target

print(features[1])
