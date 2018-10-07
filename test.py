import numpy as np
import matplotlib.pyplot as plt
from tsne import tsne


dim = 150
lines = []
word_index = {}
word_occ = {}
count = 0

with open('~w5_.txt') as data_file:  # https://www.ngrams.info/download_coca.asp
    for line in data_file:
        if np.random.random() > 0.2:
            continue
        line = line.split('\t')
        line.pop(0)
        line = [l.strip().lower() for l in line]
        for word in line:
            if word not in word_index:
                word_index[word] = count
                count += 1
                word_occ[word] = 1
            else:
                word_occ[word] += 1
        lines.append(line)

print(len(lines))
print(len(word_index))
embedding_matrix = np.random.normal(size=(len(word_index), dim), scale=0.01)
pmf = np.ones((len(word_index),))
for word, num in word_index.items():
    pmf[num] = word_occ[word] ** 0.75
pmf /= np.sum(pmf)
print(np.sum(pmf))


def apply_derivative(a_index, b_index, choices):
    global embedding_matrix
    choices = np.setdiff1d(choices, np.array([a_index, b_index]))
    temp = np.exp(np.matmul(embedding_matrix[choices, dim // 2:],
                            embedding_matrix[a_index][:dim // 2]))
    temp_with_b = np.append(temp, np.exp(np.dot(
        embedding_matrix[a_index][:dim // 2], embedding_matrix[b_index][dim // 2:])))
    sum_temp = np.sum(temp_with_b)
    delta_embedding_matrix = np.zeros((len(choices) + 2, dim // 2))
    delta_embedding_matrix[-1] = embedding_matrix[b_index][dim // 2:] - \
        (np.matmul(embedding_matrix[choices, dim // 2:].T, temp)
         + embedding_matrix[b_index][dim // 2:] * temp_with_b[-1]) / sum_temp
    delta_embedding_matrix[-2] = embedding_matrix[a_index][:dim // 2] - \
        embedding_matrix[a_index][:dim // 2] * temp_with_b[-1] / sum_temp

    delta_embedding_matrix[:-2, :] = -np.tile(np.expand_dims(embedding_matrix[a_index][:dim // 2], 0), (len(
        choices), 1)) * np.tile(np.expand_dims(temp, 1), (1, dim // 2)) / sum_temp

    embedding_matrix[choices, dim // 2:] += 0.05 * \
        delta_embedding_matrix[:-2, :]
    embedding_matrix[b_index][dim // 2:] += 0.05 * delta_embedding_matrix[-2]
    embedding_matrix[a_index][:dim // 2] += 0.05 * delta_embedding_matrix[-1]


# choices = np.arange(0, len(word_index), 1)
# for i in range(200):
#     apply_derivative(0, 1, choices)
#     print(i, np.exp(np.dot(embedding_matrix[0][:dim // 2], embedding_matrix[1][dim // 2:])) / np.sum(
#         np.exp(np.matmul(embedding_matrix[:, dim // 2:], embedding_matrix[0][:dim // 2]))))


for i in range(10):
    np.random.shuffle(lines)
    for line in lines:
        choices = np.random.choice(len(embedding_matrix), 10, False, pmf)
        for j in range(0, 4):
            apply_derivative(word_index[line[j]],
                             word_index[line[j + 1]], choices)
    print(i)

inv_map = {v: k for k, v in word_index.items()}

for i in range(300):
    a = np.random.randint(0, len(word_index))
    print(inv_map[a], end=' - ')
    b = np.exp(np.matmul(embedding_matrix[:, dim // 2:], embedding_matrix[a][:dim // 2]))
    b = b / np.sum(b)
    b[a] = -1
    temp = sorted(np.arange(0, len(word_index), 1),
                  key=lambda x: b[x], reverse=True)[:10]
    for w in temp:
        print(inv_map[w], '(', b[w], ')', end=',   ')
    print(end='\n\n')
print()

reduced_embedding_matrix = tsne(embedding_matrix/100, 2, 50, 20)
fig, ax = plt.subplots()
for txt in word_index.keys():
    ax.annotate(
        txt, (reduced_embedding_matrix[word_index[txt]][0], reduced_embedding_matrix[word_index[txt]][1]))
ax.set_ylim(-100, 100)
ax.set_xlim(-100, 100)
plt.show()
