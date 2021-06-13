# -*- coding: utf8 -*-
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.context import SQLContext
from pytoolkit import TDWProvider, TDWUtil, TDWSQLProvider
from pyspark.sql.functions import udf
import pyspark.sql.functions as f
import sys
import time
import datetime

user_name = 'tdw_weijiankong'
password = 'ra6008RAA'
db = 'g_omg_pac_app'
tb_item2vec = "shield_omg_qqcom_item2vec"
tb_user2vec = "shield_omg_qqcom_user_feat_test"
info_schema_item = ["create_date", "article_id", "article_info_name", "article_info", "value"]
info_schema_user = ["create_date", "uid", "info_name", "info_key", "value"]

user_date = time.strftime("%Y%m%d", time.localtime(time.time() - 3600 * 24))
item_date = time.strftime("%Y%m%d", time.localtime(time.time() - 3600 * 24 * 2))

item_name_array = ["item2vec_tags", "item2vec_topic", "item2vec_cat1"]
user_name_array = ["user2vec_tags", "user2vec_topic", "user2vec_cat1"]

pri_parts_user = ['p_' + user_date]
pri_parts_item = ['p_' + item_date]


def create_partition(user_name, password, db, tb, date):
    try:
        tdw1 = TDWUtil(user_name, password, db)
        tdw1.createListPartition(tb, "p_" + date, date)
    except:
        print("exit partion!")


def trip_udf(vec):
    vec_str = vec[1:-1]
    return vec_str


def compute_dimension_item(item_vec_df):
    type_function_udf = udf(lambda x: item_name_array[x / 200])
    name_df = item_vec_df.withColumn('article_info_name', type_function_udf(item_vec_df.article_info))
    pos_df = name_df.withColumn('article_info', name_df.article_info % 200)
    #pos_df.show()
    print("dimension item")
    #tdw.saveToTable(pos_df, tblName=tb_item2vec, priPart=pri_parts_item[0])
    pos_df.createOrReplaceTempView("result")
    sqlContext.sql("select * from result where article_info_name='item2vec_topic'").show()


def compute_all_item(item_vec_df):
    name_df = item_vec_df.withColumn('article_info_name', f.lit("item2vec")).cache()
    print("-----------------------------------------------------------------------")
    save_df = name_df.limit(10)
    save_df.show(10)
    print(save_df.count())
    # -*- coding: utf8 -*-
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import SparkSession
    from pyspark.sql.context import SQLContext
    from pytoolkit import TDWProvider, TDWUtil, TDWSQLProvider
    from pyspark.sql.functions import udf
    import pyspark.sql.functions as f
    import sys
    import time
    import datetime

    user_name = 'tdw_weijiankong'
    password = 'ra6008RAA'
    db = 'g_omg_pac_app'
    tb_item2vec = "shield_omg_qqcom_item2vec"
    tb_user2vec = "shield_omg_qqcom_user_feat_test"
    info_schema_item = ["create_date", "article_id", "article_info_name", "article_info", "value"]
    info_schema_user = ["create_date", "uid", "info_name", "info_key", "value"]

    user_date = time.strftime("%Y%m%d", time.localtime(time.time() - 3600 * 24))
    item_date = time.strftime("%Y%m%d", time.localtime(time.time() - 3600 * 24 * 2))

    item_name_array = ["item2vec_tags", "item2vec_topic", "item2vec_cat1"]
    user_name_array = ["user2vec_tags", "user2vec_topic", "user2vec_cat1"]

    pri_parts_user = ['p_' + user_date]
    pri_parts_item = ['p_' + item_date]

    def create_partition(user_name, password, db, tb, date):
        try:
            tdw1 = TDWUtil(user_name, password, db)
            tdw1.createListPartition(tb, "p_" + date, date)
        except:
            print("exit partion!")

    def trip_udf(vec):
        vec_str = vec[1:-1]
        return vec_str

    def compute_dimension_item(item_vec_df):
        type_function_udf = udf(lambda x: item_name_array[x / 200])
        name_df = item_vec_df.withColumn('article_info_name', type_function_udf(item_vec_df.article_info))
        pos_df = name_df.withColumn('article_info', name_df.article_info % 200)
        # pos_df.show()
        print("dimension item")
        # tdw.saveToTable(pos_df, tblName=tb_item2vec, priPart=pri_parts_item[0])
        pos_df.createOrReplaceTempView("result")
        sqlContext.sql("select * from result where article_info_name='item2vec_topic'").show()


    def compute_all_item(item_vec_df):
        name_df = item_vec_df.withColumn('article_info_name', f.lit("item2vec")).cache()
        name_df.show(5)
        print("-----------------------------------------------------------------------")
        name_df.createOrReplaceTempView("all_item_result")
        print("all item")
        sqlContext.sql("select * from all_item_result").show()
        print("save to %s %s" % (tb_item2vec, pri_parts_item[0]))
        tdw.saveToTable(save_df, tblName=tb_item2vec, priPart=pri_parts_item[0], subPart='sp_item2vec')

    def compute_dimension_user(user_vec_df):
        type_function_udf = udf(lambda x: user_name_array[x / 200])
        name_df = user_vec_df.withColumn('info_name', type_function_udf(user_vec_df.info_key))
        pos_df = name_df.withColumn('info_key', name_df.info_key % 200)
        pos_df.show()
        pos_df.createOrReplaceTempView("result")
        sqlContext.sql("select * from result where info_name='user2vec_topic'").show()
        # tdw.saveToTable(pos_df, tblName=tb_user2vec, priPart=pri_parts_user[0])

    def compute_all_user(user_vec_df):
        name_df = user_vec_df.withColumn('info_name', f.lit("user2vec"))
        name_df.createOrReplaceTempView("all_user_result")
        sqlContext.sql("select * from all_user_result where uid='0_749fd894d75aa'").show()
        print(pri_parts_user)
        tdw.saveToTable(name_df, tblName=tb_user2vec, priPart=pri_parts_user[0], subPart='sp_user2vec')

    def data_format(file_path, table_name):
        df = sqlContext.read.parquet(file_path).cache()
        df.printSchema()
        df.createOrReplaceTempView(table_name)
        sql_df = sqlContext.sql("SELECT * FROM " + table_name)
        sql_df.show()
        trip_function_udf = udf(trip_udf)
        format_df = sql_df.withColumn('vec', trip_function_udf(sql_df.vec))
        return format_df

    def handle_item(item_path, table_name):
        format_df = data_format(item_path, table_name)
        # cnt = format_df.count()
        # print("cnt: %d" % cnt)
        # if cnt < 500:
        #     return
        format_item_df = format_df.withColumnRenamed('key', 'article_id')
        format_item_df.printSchema()
        format_item_df.show()
        item_df = format_item_df.select('article_id',
                                        f.posexplode(f.split('vec', '[,]')).alias("article_info", "value"))
        item_df.printSchema()
        item_df.show()
        date_df = item_df.withColumn('create_date', f.lit(item_date))
        date_df.printSchema()
        date_df.show()
        print("compute all:")
        compute_all_item(date_df)
        print("compute dimension")
        compute_dimension_item(date_df)

    def handle_user(user_path):
        format_df = data_format(user_path, "user")
        format_item_df = format_df.withColumnRenamed('key', 'uid')
        format_item_df.printSchema()
        format_item_df.show()

        item_df = format_item_df.select('uid', f.posexplode(format_item_df.vec).alias("info_key", "value"))
        item_df.printSchema()
        item_df.show()

        date_df = item_df.withColumn('create_date', f.lit(user_date))
        date_df.printSchema()
        date_df.show()
        print("compute all:")
        compute_all_user(date_df)
        print("compute dimension")
        compute_dimension_user(date_df)

    if __name__ == '__main__':
        input_path = sys.argv[1].strip().split("=")[1]
        print("input_path:", input_path)
        # conf = SparkConf() \
        #     .setAppName("shield:fastText") \
        #     .set("spark.hadoop.validateOutputSpecs", "false")
        # sc = SparkContext(conf=conf)
        ss = SparkSession \
            .builder \
            .appName("fasttext") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        sc = ss.sparkContext
        sqlContext = SQLContext(sc)

        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')

        input_file_item = input_path + "/article500dim/20180930/*.parquet.gzip"
        input_file_user = input_path + "/user500dim/20180930/*.parquet.gzip"
        create_partition(user_name, password, db, tb_item2vec, item_date)
        create_partition(user_name, password, db, tb_user2vec, user_date)
        tdw = TDWSQLProvider(ss, user=user_name, passwd=password, db=db)
        now = datetime.datetime.now()
        h = now.hour
        print("hour: %s" % h)
        if h == 14:
            print("path: %s" % input_file_item)
            print("start handle item")
            handle_item(input_file_item, "item")
        print("path: %s" % input_file_user)
        print("start handle user")
        # handle_user(input_file_user)
        print("------end-----")
        sc.stop().createOrReplaceTempView("all_item_result")
    print("all item")
    #sqlContext.sql("select * from all_item_result").show()
    #print("save to %s %s" % (tb_item2vec, pri_parts_item[0]))
    #tdw.saveToTable(save_df, tblName=tb_item2vec, priPart=pri_parts_item[0], subPart='sp_item2vec')


def compute_dimension_user(user_vec_df):
    type_function_udf = udf(lambda x: user_name_array[x / 200])
    name_df = user_vec_df.withColumn('info_name', type_function_udf(user_vec_df.info_key))
    pos_df = name_df.withColumn('info_key', name_df.info_key % 200)
    pos_df.show()
    pos_df.createOrReplaceTempView("result")
    sqlContext.sql("select * from result where info_name='user2vec_topic'").show()
    #tdw.saveToTable(pos_df, tblName=tb_user2vec, priPart=pri_parts_user[0])


def compute_all_user(user_vec_df):
    name_df = user_vec_df.withColumn('info_name', f.lit("user2vec"))
    name_df.createOrReplaceTempView("all_user_result")
    sqlContext.sql("select * from all_user_result where uid='0_749fd894d75aa'").show()
    print(pri_parts_user)
    tdw.saveToTable(name_df, tblName=tb_user2vec, priPart=pri_parts_user[0], subPart='sp_user2vec')


def data_format(file_path, table_name):
    df = sqlContext.read.parquet(file_path).cache()
    df.printSchema()
    df.createOrReplaceTempView(table_name)
    sql_df = sqlContext.sql("SELECT * FROM "+table_name)
    sql_df.show()
    trip_function_udf = udf(trip_udf)
    format_df = sql_df.withColumn('vec', trip_function_udf(sql_df.vec))
    return format_df


def handle_item(item_path, table_name):
    format_df = data_format(item_path, table_name)
    # cnt = format_df.count()
    # print("cnt: %d" % cnt)
    # if cnt < 500:
    #     return
    format_item_df = format_df.withColumnRenamed('key', 'article_id')
    format_item_df.printSchema()
    format_item_df.show()
    item_df = format_item_df.select('article_id', f.posexplode(f.split('vec', '[,]')).alias("article_info", "value"))
    item_df.printSchema()
    item_df.show()
    date_df = item_df.withColumn('create_date', f.lit(item_date))
    date_df.printSchema()
    date_df.show()
    print("compute all:")
    compute_all_item(date_df)
    print("compute dimension")
    compute_dimension_item(date_df)


def handle_user(user_path):
    format_df = data_format(user_path, "user")
    format_item_df = format_df.withColumnRenamed('key', 'uid')
    format_item_df.printSchema()
    format_item_df.show()

    item_df = format_item_df.select('uid', f.posexplode(format_item_df.vec).alias("info_key", "value"))
    item_df.printSchema()
    item_df.show()

    date_df = item_df.withColumn('create_date', f.lit(user_date))
    date_df.printSchema()
    date_df.show()
    print("compute all:")
    compute_all_user(date_df)
    print("compute dimension")
    compute_dimension_user(date_df)


if __name__ == '__main__':
    input_path = sys.argv[1].strip().split("=")[1]
    print("input_path:", input_path)
    # conf = SparkConf() \
    #     .setAppName("shield:fastText") \
    #     .set("spark.hadoop.validateOutputSpecs", "false")
    # sc = SparkContext(conf=conf)
    ss = SparkSession \
        .builder \
        .appName("fasttext") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    sc = ss.sparkContext
    sqlContext = SQLContext(sc)

    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')

    input_file_item = input_path+"/article500dim/20180930/*.parquet.gzip"
    input_file_user = input_path+"/user500dim/20180930/*.parquet.gzip"
    create_partition(user_name, password, db, tb_item2vec, item_date)
    create_partition(user_name, password, db, tb_user2vec, user_date)
    tdw = TDWSQLProvider(ss, user=user_name, passwd=password, db=db)
    now = datetime.datetime.now()
    h = now.hour
    print("hour: %s" % h)
    if h == 14:
        print("path: %s" % input_file_item)
        print("start handle item")
        handle_item(input_file_item, "item")
    print("path: %s" % input_file_user)
    print("start handle user")
    #handle_user(input_file_user)
    print("------end-----")
    sc.stop()