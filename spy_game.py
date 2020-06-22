# Path for file sample.txt is stored in file_path
# sample.txt contains message "This is a sample message."

# Path for file M1.txt is stored in file_path_1
# M1.txt contains message "2222"

# Path for file M2.txt is stored in file_path_2
# M2.txt contains message "2477"

# Path for file M3.txt is stored in file_path_3
# M3.txt contains message "Green"

# Path for file M4.txt is stored in file_path_4
# M4.txt contains message "I hope you are good now"

# Path for file M5.txt is stored in file_path_5
# M5.txt contains message "I hope good things happen in your life."

# Path for file M6.txt is stored in file_path_6
# M6.txt contains message "The man was one step closer towards his quest to become a spy."

def read_file(path) :
    file = open(path, 'r')
    sentence = file.readline()
    file.close()
    return sentence

sample_message = read_file(file_path)

def read_file(path) :
    file = open(path, "r")
    sentence = file.readline()
    file.close()
    return sentence

def fuse_msg(message_a, message_b) :
    quotient = (int(message_b))//(int(message_a))
    return str(quotient)

message_1 = read_file(file_path_1)
message_2 = read_file(file_path_2)
print(message_1)
print(message_2)
secret_msg_1 = fuse_msg(message_1, message_2)

def read_file (path) :
    file = open(path, "r")
    sentence = file.readline()
    file.close()
    return sentence

def substitute_msg (message_c) :
    if message_c=='Red' :
        sub = 'Army General'
    elif message_c=='Green' :
        sub = 'Data Scientist'
    elif message_c=='Blue' :
        sub = 'Marine Biologist'
    return sub

message_3 = read_file(file_path_3)
secret_msg_2 = substitute_msg(message_3)

def read_file(path) :
    file = open(path, 'r')
    sentence = file.readline()
    file.close()
    return sentence

def compare_msg(message_d, message_e) :
    a_list = message_d.split()
    b_list = message_e.split()
    c_list=[]
    for i in a_list :
        if i in b_list :
            continue
        c_list.append(i)
    final_msg = " ".join(c_list)
    return final_msg

message_4 = read_file(file_path_4)
message_5 = read_file(file_path_5)
print(message_4)
print(message_5)
secret_msg_3 = compare_msg(message_4, message_5)

def read_file(path) :
    file = open(path, 'r')
    sentence = file.readline()
    file.close()
    return sentence

def extract_msg(message_f) :
    a_list = message_f.split()
    even_word = lambda x : (len(x)%2==0)
    b_list = filter(even_word, a_list)
    final_msg = " ".join(b_list)
    return final_msg

message_6 = read_file(file_path_6)
print(message_6)
secret_msg_4 = extract_msg(message_6)

def write_file(secret_msg, path) :
    file = open(path, 'a+')
    file.write(secret_msg)
    file.close

message_parts=[secret_msg_3, secret_msg_1, secret_msg_4, secret_msg_2]
secret_msg = " ".join(message_parts)
final_path= user_data_dir + '/secret_message.txt'
write_file(secret_msg, final_path)
print(secret_msg)
