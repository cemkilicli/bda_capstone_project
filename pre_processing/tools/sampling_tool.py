import numpy as np


lines = []
counter = 0
with open('../exp_data/raw_data/train.csv') as fp:
    for line in fp:
        rand_num = np.random.randint(100, size=1)[0]
        if counter == 0 or rand_num <= 2:
            lines.append(line.strip())
        counter += 1


with open('../exp_data/sample_mini.csv', 'w') as f:
    f.write("\n".join(lines))



