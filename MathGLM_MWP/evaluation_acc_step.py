import numpy as np
import json
import re

# ground truth
with open('test.jsonl', 'r') as fin:
    inputs = [json.loads(line) for line in fin.readlines() if line]
    
truth_answer = []
for value in inputs:
    question = value['question']
    answer = value['answer']
    truth_answer.append(answer)

test_answer = []
with open('./samples_result/test_eval.txt', 'r') as fin:
    for line in fin:
        line = line.strip()
        q_a_list = line.split('答:')
        if len(q_a_list) == 2:
            question, answer = q_a_list[0], q_a_list[1]
        else:
            answer = q_a_list[-1]
        test_answer.append(answer)
        
print("test_answer", len(test_answer))

nums_ans = 0
nums_expression = 0
truth = []
error = []

num_calculate = 0

unexpect_ans = []

truth_ans = ""

for idx, ans1 in enumerate(truth_answer):
    ans2 = test_answer[idx]
    list1 = ans1.split('=')
    list2 = ans2.split('=')
    
    if 'x' in ans1 and 'x' in ans2:
        if len(ans2) >= 2:
            if list1[-1] == list2[-1]:
                nums_ans += 1
                nums_expression += 1
            elif list1[1] == list2[1]:
                nums_expression += 1
                fraction_pattern = r'^(\d+)\((\d+/\d+)\)$'
                m_frac = re.match(fraction_pattern, list1[-1])             
                if '%' in list1[-1]:
                    new_ans1 = list1[-1].replace('%', '/100')
                    new_ans1 = round(eval(new_ans1), 4)
                    try:
                        generate_ans = round(eval(list2[-1]), 4)
                    except SyntaxError:
                        print("idx=====", idx+1)
                        print("SyntaxError--------list2[-1]@@@@", list2[-1]) 
                    except TypeError:
                        print("idx=====", idx+1)
                        print("TypeError------list2[-1]@@@@", list2[-1]) 
                    except ZeroDivisionError:
                        print("idx=====", idx+1)
                        print("ZeroDivisionError-----list2[-1]@@@@", list2[-1])
                    except NameError:
                        print("idx=====", idx+1)
                        print("NameError-----list2[-1]@@@@", list2[-1])
                    if new_ans1 == generate_ans:
                        nums_ans += 1
                    else:
                        num_calculate += 1
                elif m_frac:
                    whole_number = int(m_frac.group(1))
                    fraction = m_frac.group(2)
                    expression = str(whole_number) + '+(' + fraction + ')'
                    new_ans1 = round(eval(expression), 4)
                    try:
                        generate_ans = round(eval(list2[-1]), 4)
                    except SyntaxError:
                        print("idx=====", idx+1)
                        print("SyntaxError--------list2[-1]@@@@", list2[-1]) 
                    except TypeError:
                        print("idx=====", idx+1)
                        print("TypeError------list2[-1]@@@@", list2[-1]) 
                    except ZeroDivisionError:
                        print("idx=====", idx+1)
                        print("ZeroDivisionError-----list2[-1]@@@@", list2[-1])
                    except NameError:
                        print("idx=====", idx+1)
                        print("NameError-----list2[-1]@@@@", list2[-1])
                    if new_ans1 == generate_ans:
                        nums_ans += 1
                    else:
                        num_calculate += 1               
                elif '/' in list1[-1]:
                    new_ans1 = round(eval(list1[-1]), 4)
                    try:
                        generate_ans = round(eval(list2[-1]), 4)
                    except SyntaxError:
                        print("idx=====", idx+1)
                        print("SyntaxError--------list2[-1]@@@@", list2[-1]) 
                    except TypeError:
                        print("idx=====", idx+1)
                        print("TypeError------list2[-1]@@@@", list2[-1]) 
                    except ZeroDivisionError:
                        print("idx=====", idx+1)
                        print("ZeroDivisionError-----list2[-1]@@@@", list2[-1])
                    except NameError:
                        print("idx=====", idx+1)
                        print("NameError-----list2[-1]@@@@", list2[-1])
                    if new_ans1 == generate_ans:
                        nums_ans += 1
                    else:
                        num_calculate += 1
                else:
                    try:
                        generate_ans = round(eval(list2[-1]), 4)
                    except SyntaxError:
                        print("idx=====", idx+1)
                        print("SyntaxError--------list2[-1]@@@@", list2[-1]) 
                    except TypeError:
                        print("idx=====", idx+1)
                        print("TypeError------list2[-1]@@@@", list2[-1]) 
                    except ZeroDivisionError:
                        print("idx=====", idx+1)
                        print("ZeroDivisionError-----list2[-1]@@@@", list2[-1])
                    except NameError:
                        print("idx=====", idx+1)
                        print("NameError-----list2[-1]@@@@", list2[-1]) 
                    if round(eval(list1[-1]), 4) == generate_ans:
                        nums_ans += 1
                    else:   
                        num_calculate += 1     
            else:
                # check Arithmetic
                generate_expression = ans2.split('=')[1]
                if '%' in generate_expression:
                    generate_expression = generate_expression.replace('%', '/100')
                try:
                    generate_ans = round(eval(generate_expression), 4)
                    truth_ans = ans1.split('=')[-1]
                    if '/' not in truth_ans:
                        if '%' in truth_ans:
                            truth_ans = round(float(truth_ans[:-1])*1.0/100, 4)
                        if '+' in truth_ans or '-' in truth_ans or '/' in truth_ans or '*' in truth_ans:
                            truth_ans = round(float(eval(truth_ans)), 4)  
                        else:
                            truth_ans = round(float(truth_ans), 4)  
                except SyntaxError:
                    print("idx=====", idx+1)
                    print("SyntaxError--------generate_expression@@@@", generate_expression) 
                except TypeError:
                    print("idx=====", idx+1)
                    print("TypeError------generate_expression@@@@", generate_expression) 
                except ZeroDivisionError:
                    print("idx=====", idx+1)
                    print("ZeroDivisionError-----generate_expression@@@@", generate_expression)
                except NameError:
                    print("idx=====", idx+1)
                    print("NameError-----generate_expression@@@@", generate_expression)
                if generate_ans == truth_ans:
                    nums_expression += 1 
                    num_calculate += 1
                else:
                    question = inputs[idx]['question']
                    error_json = {}
                    truth_json = {}
                    
                    truth_json["question"] = question
                    error_json["question"] = question
                    
                    truth_json["answer"] = ans1
                    error_json["answer"] = ans2
                
                    truth.append(json.dumps(truth_json, ensure_ascii=False))
                    error.append(json.dumps(error_json, ensure_ascii=False))
        else:  
            question = inputs[idx]['question']
            error_json = {}
            truth_json = {}
            truth_json["question"] = question
            error_json["question"] = question
            truth_json["answer"] = ans1
            error_json["answer"] = ans2
            truth.append(json.dumps(truth_json, ensure_ascii=False))
            error.append(json.dumps(error_json, ensure_ascii=False))
    else:  
        if list1[-1] == list2[-1]:
            nums_ans += 1
            nums_expression += 1
        else:
            question = inputs[idx]['question']
            error_json = {}
            truth_json = {}
            truth_json["question"] = question
            error_json["question"] = question
            truth_json["answer"] = ans1
            error_json["answer"] = ans2
            truth.append(json.dumps(truth_json, ensure_ascii=False))
            error.append(json.dumps(error_json, ensure_ascii=False))   
            unexpect_ans.append(json.dumps(error_json, ensure_ascii=False))

            
print("Accuracy of Answer: {}%".format(nums_ans*100/len(inputs)))
print("Accuracy of Arithmetic: {}%".format(nums_expression*100/len(inputs)))

print("caluation_error", num_calculate)

