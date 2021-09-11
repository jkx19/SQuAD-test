import os
import sys

input = 'SQuAD-test/data/dev-v2.0.json'
output = 'SQuAD-test/out/prediction.json'

# print(sys.argv)

if len(sys.argv) >= 3:
    input = sys.argv[1]
    output = sys.argv[2]


COMMANDLINE = f'python SQuAD-test/cls.py \
    --input {input} \
    --output {output}'

os.system(COMMANDLINE)