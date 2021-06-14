import jieba.analyse

def keywords_textrank(text):
    keywords = jieba.analyse.textrank(text, topK=6)
    return keywords

if __name__ == "__main__":
    text = "来源：中国科学报本报讯（记者肖洁）又有一位中国科学家喜获小行星命名殊荣！4月19日下午，中国科学院国家天文台在京举行“周又元星”颁授仪式，" \
           "我国天文学家、中国科学院院士周又元的弟子与后辈在欢声笑语中济济一堂。国家天文台党委书记、" \
           "副台长赵刚在致辞一开始更是送上白居易的诗句：“令公桃李满天下，何须堂前更种花。”" \
           "据介绍，这颗小行星由国家天文台施密特CCD小行星项目组于1997年9月26日发现于兴隆观测站，" \
           "获得国际永久编号第120730号。2018年9月25日，经国家天文台申报，" \
           "国际天文学联合会小天体联合会小天体命名委员会批准，国际天文学联合会《小行星通报》通知国际社会，" \
           "正式将该小行星命名为“周又元星”。"

    # 基于jieba的textrank算法实现
    keywords = keywords_textrank(text)
    print(keywords)