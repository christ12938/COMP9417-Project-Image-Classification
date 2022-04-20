import numpy as np
from sklearn.model_selection import train_test_split

X = np.load('X_train.npy')
y = np.load('y_train.npy')

data_x_train, X_test, data_y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

data_x_train_flipped_vertical = data_x_train[:, ::-1, :, :]
data_x_train_flipped_horizontal = data_x_train[:, :, ::-1, :]
data_x_train_flipped = np.append(data_x_train_flipped_horizontal, data_x_train_flipped_vertical, axis=0)

data_x_train_rotated_90 = np.rot90(data_x_train, 1, (1,2))
data_x_train_rotated_180 = np.rot90(data_x_train, 2, (1,2))
data_x_train_rotated_270 = np.rot90(data_x_train, 3, (1,2))
data_x_train_rotated = np.append(data_x_train_rotated_90, data_x_train_rotated_180, axis=0)
data_x_train_rotated = np.append(data_x_train_rotated, data_x_train_rotated_270, axis=0)

data_x_train_new = np.append(data_x_train, data_x_train_flipped, axis=0)
data_x_train_new = np.append(data_x_train_new, data_x_train_rotated, axis=0)

data_y_train_new = np.append(data_y_train, data_y_train, axis=0)
data_y_train_new = np.append(data_y_train_new, data_y_train, axis=0)
data_y_train_new = np.append(data_y_train_new, data_y_train, axis=0)
data_y_train_new = np.append(data_y_train_new, data_y_train, axis=0)
data_y_train_new = np.append(data_y_train_new, data_y_train, axis=0)

with open('X_train_split.npy', 'wb') as f1:
    np.save(f1, data_x_train_new)
with open('X_test_split.npy', 'wb') as f2:
    np.save(f2, X_test)
with open('y_train_split.npy', 'wb') as f3:
    np.save(f3, data_y_train)
with open('y_test_split.npy', 'wb') as f4:
    np.save(f4, y_test)
