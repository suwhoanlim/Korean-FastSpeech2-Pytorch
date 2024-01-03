# EXPLANATION: DEFINITION
# reads in AIHUB data's json file and transcribe them to fit KSS style.

import json
import os
import tqdm, re
from tqdm import tqdm
from jamo import h2j
from glob import glob

def process_json_file(json_file_path, fileno):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    org_label_text = data['전사정보']['OrgLabelText']
    return f"{fileno}|{org_label_text}"

def write_to_file(input_folder, output_folder, last_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    json_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')])

    tot = []
    for filename in json_files:
        if filename.endswith('.json'):
            json_file_path = os.path.join(input_folder, filename)
            output_text = process_json_file(json_file_path, filename[-11:-5])
            tot.append(output_text)
        if filename == last_name :
            break
        

    output_file_path = os.path.join(output_folder, f"{input_folder[-3:]}.txt") # save all the .lab files
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for a in tot:
            output_file.write(f'{a}\n')
        print('write successful')

## EXPLANATION: DEFINITION
# create .lab files
# after making .lab files, sum them up and return it so that it can be used to create jamo pair dictionary



def make_lab(text, base_dir):
    filters = '([.,!?])'

    with open(text, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            temp = line.split('|')
            file_dir, script = temp[0], temp[1]
            script = re.sub(re.compile(filters), '', script)
            # file_dir = file_dir.split('/')
            # fn = file_dir[0] + '/' + file_dir[1][:-3] + 'lab' # filename configuration? '1/1_0015.lab'
            file_list = sorted(glob(os.path.join(base_dir, '*.wav')))
            fn = ''
            for ff in file_list:
                # print(ff[-10:-4], file_dir)
                if ff[-10:-4] == file_dir:
                    fn = ff[:-4] + '.lab'
                    break
            # fn = file_dir + '.lab'
            file_dir = os.path.join(base_dir, fn) # base_dir/1/1_0015.lab
            with open(file_dir, 'w', encoding='utf-8') as f:
                f.write(script)
    ###
    file_list = sorted(glob(os.path.join(base_dir, '*.lab')))
    print(file_list)
    ###
    jamo_dict = {}
    for file_name in tqdm(file_list):
        sentence =  open(file_name, 'r', encoding='utf-8').readline()
        jamo = h2j(sentence).split(' ')
        for i, s in enumerate(jamo):
            if s not in jamo_dict:
                jamo_dict[s.rstrip()] = ' '.join(jamo[i].rstrip()) # FREAKING NEWLINE BABY!!
    
    return jamo_dict

##### TODO: ONLY NEED TO CHANGE THIS DIR
base_dir = '/home/soma1/문서/AIHUB_selected_label_and_sound'
#####

BYK_ori = '0220_G1A3E7_BYK'
CKY_ori = '0347_G1A5E7_CKY'
JMH_ori = '1538_G2A1E7_JMH'
KSB_ori = '9042_G2A6E7_KSB'


last_name = '0220_G1A3E7_BYK_001013.json' # filename to stop recording
write_to_file(base_dir + '/label/' + BYK_ori, base_dir + '/label', last_name)


last_name = '0347_G1A5E7_CKY_000845.json' # filename to stop recording
write_to_file(base_dir + '/label/' + CKY_ori, base_dir + '/label', last_name)

last_name = '1538_G2A1E7_JMH_000932.json' # filename to stop recording
write_to_file(base_dir + '/label/' + JMH_ori, base_dir + '/label', last_name)

last_name = '9042_G2A6E7_KSB_000930.json' # filename to stop recording
write_to_file(base_dir + '/label/' + KSB_ori, base_dir + '/label', last_name)



a = {}
a = make_lab(base_dir + '/label/' + 'BYK.txt', base_dir + '/sound/' + BYK_ori)

b = make_lab(base_dir + '/label/' + 'CKY.txt', base_dir + '/sound/' + CKY_ori)
a = a | b

b = make_lab(base_dir + '/label/' + 'JMH.txt', base_dir + '/sound/' + JMH_ori)
a = a | b

b = make_lab(base_dir + '/label/' + 'KSB.txt', base_dir + '/sound/' + KSB_ori)
a = a | b

###

# Make jamo pair dictionary
dict_name = 'aihub_korean_dict.txt'
with open(os.path.join(base_dir, dict_name), 'w', encoding='utf-8') as f:
    for key in a.keys():
        # content = '{}\t{}\n'.format(key, jamo_dict[key])
        content = f'{key}\t{a[key]}\n'
        f.write(content)