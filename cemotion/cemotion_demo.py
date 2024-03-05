from cemotion import Cemotion


def text_emotion(text, c):

    val = c.predict(text)
    #print('预测值:{:6f}'.format(val))
    if val > 0.5:
        print('"', text, '"\n', '情感分类：正向', '\n')
    else:
        print('"', text, '"\n', '情感分类：负向', '\n')


if __name__ == "__main__":

    text1 = '刘亦菲的气质像一个仙女下凡'
    text2 = '今天下班被追尾了，生气'
    text3 = '今天下雪了欸，好开心呀'

    c = Cemotion()
    text_emotion(text1,c)
    text_emotion(text2,c)
    text_emotion(text3,c)