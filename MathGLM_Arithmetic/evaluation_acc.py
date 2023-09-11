import numpy as np
import re

with open('./samples_result/test_eval.txt', 'r') as fin:
    inputs_2 = fin.readlines()


nums_ans = 0
for ex in inputs_2:
    question = ex.strip().split('=')[0]
    generate_ans = ex.strip().split('=')[-1]
    if '[' in question:
        question = question.replace('[', '(')
        question = question.replace(']', ')')
    if '%' in question:
        tokens = re.split(r"(\+|\-|\*|/)", question)
        for i, token in enumerate(tokens):
            if '%' in token:
                token = token.replace('%', '*1.0/100')
                print("token", token)
                if token[-1] == ')' and token[-2].isdigit():
                    tokens[i] = str(eval(token[:-1])) + ')'
                elif token[-1] == ')' and token[-2] == ')' and token[-3].isdigit():
                    tokens[i] = str(eval(token[:-2])) + '))'
                elif token[0] == '(' and token[1].isdigit():
                    tokens[i] = '(' + str(eval(token[1:]))
                elif token[0] == '(' and token[1] == '(' and token[2].isdigit():
                    tokens[i] = '((' + str(eval(token[2:]))
                else:
                    tokens[i] = eval(token)
        question = ""
        for token in tokens:
            question += str(token)
    
    if '/' in generate_ans:
        generate_ans = eval(generate_ans)
    truth_ans = eval(question)
    if '+' in str(generate_ans):
        continue
    if round(float(generate_ans), 2) == round(float(truth_ans), 2):
        nums_ans += 1

print("Accuracy:{}%".format(nums_ans * 1.0 / len(inputs_2) * 100))

