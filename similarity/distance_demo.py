import similarity

def edit_distance(s1, s2):
    return similarity.levenshtein(s1, s2)

if __name__ == "__main__":
    str1 = "公司地址是哪里"
    str2 = "公司在什么位置"
    print(edit_distance(str1, str2))