# This file is part of fast-wikification.
#
# Fast-wikification is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.

# Fast-wikification is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.

# You should have received a copy of the GNU General Public License along with
# fast-wikification. If not, see <https://www.gnu.org/licenses/>.

# -*- coding: utf-8 -*-
import os
import shutil
import socket
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, BooleanType
from pyspark.sql.functions import col, element_at, udf, explode, trim, split, lower, \
    monotonically_increasing_id, lit
import pyspark.sql.functions as f
from pyspark.sql.window import Window
import pickle
from __init__ import stopwords

import re

FDIR = os.path.dirname(os.path.abspath(__file__))
os.environ["SPARK_HOME"] = "/opt/spark"
os.environ["PYTHONPATH"] = "/opt/spark/python"
os.environ["SPARK_LOCAL_IP"] = "{}".format(socket.gethostname())
SPARK_MASTER = "spark://{}:7077".format(socket.gethostname())

'''
entity_id -> entity_details:
- load_entities_lowercase: 
    * title_id: entity id
    * namespace: entity namespace
    * title: entity title (in lowercase)
    * is_redirect: is redirect flag

anchor_id -> text:
- load_anchors_ids
    * anchor_text
    * anchor_id

dataset:
- run
    * source_article_id 
    * anchor_id
    * entity_id    
'''
has_val = udf(lambda val: True if val else False, BooleanType())
second_or_fist = udf(lambda first, second: second or first)
equals = udf(lambda first, second: 1 if first == second else 0, IntegerType())
min_max_norm = udf(lambda v, mn, mx: 0.0 if mx == mn else (v - mn) / (mx - mn), FloatType())
second_max_diff = udf(lambda count, max_count: count if count == max_count else max_count - count, IntegerType())


refine_udf = udf(lambda l: [l] + [i for i in l.replace('/', ' ').split()], ArrayType(StringType()))


def warehouse(warehouse_path, refresh_warehouse=False, save=True):
    def warehouse_decorator(func):
        def func_wrapper(*args, **kwargs):
            if refresh_warehouse or not os.path.exists(warehouse_path):
                if os.path.exists(warehouse_path):
                    shutil.rmtree(warehouse_path)
                df = func(*args, **kwargs)
                if save:
                    print("saving", warehouse_path)
                    df.write.save(warehouse_path)
            else:
                df = args[0].sc.read.load(warehouse_path)
            return df

        return func_wrapper

    return warehouse_decorator


class ETL:
    def __init__(self, articles_dump_path="datasets/enwiki-20220420-pages-articles.xml.bz2"):
        try:
            conf = SparkConf()
            conf.setMaster(SPARK_MASTER)
            conf.setAppName("Word Sense Disambiguation")
            conf.set("spark.executor.memory", "60g")
            conf.set("spark.driver.memory", "60g")
            conf.set("spark.driver.maxResultSize", "55g")
            conf.set("spark.eventLog.dir", "/DATA/tmp")
            conf.set("spark.sql.crossJoin.enabled", True)
            conf.set("spark.jars.packages", "com.databricks:spark-xml_2.12:0.13.0")
            conf.set("spark.local.dir", "/DATA/tmp")
            self.articles_dump_path = articles_dump_path
            self.sc = SparkSession.builder.config(conf=conf).getOrCreate()
            print("connection succeeded with Master", conf)
        except Exception as e:
            print("unable to connect to remote server")
            print(e)

    """
    +------+---+-----------+-------------------+--------------------+
    |    id| ns|redirect_to|           text    |               title|
    +------+---+-----------+-------------------+--------------------+
    |107689|  0|      null |[/* Notable peop...|Temple City, Cali...|
    |107690|  0|      null |[Undid revision ...|Torrance, California|
    |107691|  0|      null |[Filled in 5 bar...|Val Verde, Califo...|
    |107692|  0|      null |[/* See also */,...| Valinda, California|
    |107693|  0|      null |[/* City status ...|  Vernon, California|
    +------+---+-----------+-------------------+--------------------+
    """

    @warehouse(os.path.join(FDIR, "warehouse/articles.parquet"))
    def load_articles(self):
        articles = self.sc.read.format('xml') \
            .options(rowTag='page') \
            .load(self.articles_dump_path)
        articles = articles.select(
            articles.id,
            articles.title,
            articles.ns,
            articles.redirect._title.alias("redirect_to"),
            articles.revision.text._VALUE.alias("text")
        )
        return articles

    """
        +------+------------------------------+------------------+
        |id    |entity_title                  |anchor            |
        +------+------------------------------+------------------+
        |107689|city (california)             |City              |
        |107689|los angeles county, california|Los Angeles County|
        |107689|california                    |California        |
        |107689|list of sovereign states      |Country           |
        |107689|united states                 |United States     |
        |107689|u.s. state                    |State             |
        |107689|california                    |California        |
        |107689|list of counties in california|County            |
        |107689|los angeles county, california|Los Angeles       |
        |107689|mayor                         |Mayor             |
        +------+------------------------------+------------------+
    """

    @warehouse(os.path.join(FDIR, "warehouse/articlelinks.parquet"))  # R refresh_warehouse=True
    def load_article_links(self):
        """
        :return: "id", "anchor", "entity_title"
        Links from all articles are included here. (regardless of namespace). Some of them are inter-page links,
        (and will be discarded later).
        """
        articles = self.load_articles()
        articles = articles.filter(articles.text.isNotNull())  # 1667 null articles filtered (in ns!=0)
        # Inter-Wiki link regex
        iwlinks = re.compile(r'\[\[(?!(?:File:|Image:))(.+?)\]\]')
        # convert text to iwlinks links string list
        textlinks_udf = udf(lambda text: list(iwlinks.findall(text)), ArrayType(StringType()))

        return articles \
            .select(
            articles.id,
            explode(textlinks_udf(articles.text)).alias("wikilinks")
        ).select(
            col("id"),
            split(trim(col("wikilinks")), r"\|").alias("wikilinks")
        ).select(
            col("id"),
            lower(col("wikilinks")[0]).alias("entity_title"),
            element_at(col("wikilinks"), -1).alias("anchor"),
        )

    """
    +----------+---------+
    |article_id|entity_id|
    +----------+---------+
    |  24210863| 24210863|
    |  57176541| 57176541|
    |   1103883|  1103883|
    |   1166953|  1166953|
    |    850005|   850005|
    +----------+---------+
    This method resolves redirects, and creates a map of article_ids (including redirect articles) to destination entity ids
    """

    @warehouse(os.path.join(FDIR, "warehouse/entity_map.parquet"))
    def load_entity_map(self):

        arts = self.load_articles().drop("text")
        # convert redirect title to redirect id
        right = arts.toDF("redirect_id", "title0", "ns0", "redirect0")  # right member of join
        arts = arts \
            .join(right, arts.redirect_to == right.title0, 'left') \
            .select('id', 'title', 'ns', 'redirect_id')

        # get first redir_map, (supose all entities having a redirect need further redirect investigation)
        redir_map = arts \
            .select('id', 'redirect_id') \
            .withColumn('has_redirect', has_val(arts.redirect_id))

        def iterate(redirect_map):
            right = redirect_map \
                .toDF('id0', 'redirect_id0', 'has_redirect0')  # right member of join
            redirect_map = redirect_map \
                .join(right, redirect_map.redirect_id == right.id0, 'left')

            redirect_map = redirect_map \
                .select(
                'id',
                second_or_fist(redirect_map.redirect_id0, redirect_map.redirect_id).alias('redirect_id'),
                has_val(redirect_map.has_redirect0).alias('has_redirect')
            )
            return redirect_map

        redir_map = iterate(iterate(iterate(redir_map)))
        rm = redir_map.filter(redir_map.has_redirect == False).drop('has_redirect')

        return rm.withColumn(
            'entity_id',
            second_or_fist(rm.id, rm.redirect_id)
        ).select(
            col('id').alias('article_id'),
            col('entity_id')
        )

    """
    +----------+---------+---+--------------------+
    |article_id|entity_id| ns|               title|
    +----------+---------+---+--------------------+
    |       944|      944|  0|  automated alice/ii|
    |      1075|     1075|  0|antigua and barbu...|
    |      1151|     1151|  0|                ak47|
    |      1232|     1232|  0|ashmore and carti...|
    |      1436|     1436|  0|             abraham|
    +----------+---------+---+--------------------+
    title: the article title (including redirect titles)
    entity_id: is the corresponding entity id (redirects are resolved to the final destination entity)
    """

    @warehouse(os.path.join(FDIR, "warehouse/entities_lowercase.parquet"))
    def load_entities(self):
        """
        Loads entities in lowercase (title_id, namespace, title (lowercase), is_redirect )
         * title_id: entity id
         * namespace: entity namespace
         * title: entity title (in lowercase)
         * is_redirect: is redirect flag
        :return: (title_id, ns, title, is_redirect)
        """
        e = self.load_entity_map()
        a = self.load_articles().select(col('id'), col('title'), col('ns'))
        return e \
            .join(
            a, a.id == e.article_id,
            'left'
        ) \
            .select(
            e.article_id,
            e.entity_id,
            a.ns,
            lower(a.title).alias('title')
        )

    @warehouse(os.path.join(FDIR, "warehouse/entities.parquet"))
    def load_raw_entities(self):
        """
        Loads entities (title_id, namespace, title, is_redirect )
         * title_id: entity id
         * namespace: entity namespace
         * title: entity title
         * is_redirect: is redirect flag
        :return: (title_id, ns, title, is_redirect)
        """
        e = self.load_entity_map()
        a = self.load_articles().select(col('id'), col('title'), col('ns'))
        return e \
            .join(
            a,
            a.id == e.article_id,
            'left',
        ) \
            .select(
            e.article_id,
            e.entity_id,
            a.ns,
            a.title.alias('title')
        )

    """
    Loads RedW spotMap: a table of all Wikipedia titles, including the redirect titles.
    +--------------------+---------+                                                
    |               title|entity_id|
    +--------------------+---------+
    |       norwegian sea|    21281|
    |   ursula k. le guin|    32037|
    |boulevard périphé...|    62001|
    |    vanessa redgrave|    63741|
    |         bbc radio 4|    72758|
    |            aquileia|    75839|
    |lamotte township,...|   119037|
    |pacific city, oregon|   130949|
    |    grant, wisconsin|   141714|
    |gilman city, miss...|   150743|
    +--------------------+---------+
    """

    @warehouse(os.path.join(FDIR, "warehouse/redw_spot_map.parquet"))
    def load_redw_spot_map(self):
        e = self.load_raw_entities()
        return e \
            .select(
            e.title,
            e.entity_id
        )


    """
    +--------------------+---------+---+-------+---------------+                    
    |               title|entity_id| SR|SR_norm|SR_min_max_norm|
    +--------------------+---------+---+-------+---------------+
    |Committee for Cha...| 10053367|1.0|    1.0|            0.0|
    |1998 United State...| 10078354|1.0|    1.0|            0.0|
    +--------------------+---------+---+-------+---------------+
    """
    @warehouse(os.path.join(FDIR, "warehouse/redw_sr.parquet"))
    def load_redw_sr(self):
        sm = self.load_redw_spot_map()
        entity_links = self.load_raw_entity_links().distinct().withColumnRenamed("entity_id", "anchor_entity_id")
        sr = entity_links \
            .join(
            sm,
            sm.title == entity_links.anchor,
            'left'
        ) \
            .drop(col('anchor')) \
            .filter(col('title').isNotNull())
        sr = sr.select(sr.source_article_id, sr.anchor_entity_id, sr.title, sr.entity_id).groupBy(
            sr.source_article_id,
            sr.anchor_entity_id,
            sr.title,
            sr.entity_id
        ).count()
        sr = sr.withColumn('N', f.sum(col('count')).over(Window.partitionBy('title')))
        sr = sr.withColumn('equals', equals(sr.anchor_entity_id, sr.entity_id))
        sr = sr.withColumn('K', f.sum(col('equals')).over(Window.partitionBy('title')))
        sr = sr.withColumn('SR', col('K') / col('N'))
        sr = sr.withColumn('SR_max', f.max(col('SR')).over(Window.partitionBy('entity_id')))
        sr = sr.withColumn('SR_min', f.min(col('SR')).over(Window.partitionBy('entity_id')))
        sr = sr.withColumn('SR_norm', col('SR') / col('SR_max'))  # max scaling is described on the paper
        sr = sr.withColumn('SR_min_max_norm', min_max_norm(sr.SR, sr.SR_min, sr.SR_max))  # min-max norm
        sr = sr.select(sr.title, sr.entity_id, sr.SR, sr.SR_norm, sr.SR_min_max_norm).distinct()
        sr = sr.na.fill(value=0.0)
        return sr

    """
    +-----------------+--------------------+---------+                              
    |source_article_id|              anchor|entity_id|
    +-----------------+--------------------+---------+
    |                0|       albaniapeople|    67578|
    |                0|academyawards/bes...|    61702|
    |                0|             america|  3434750|
    +-----------------+--------------------+---------+ 
    """

    @warehouse(os.path.join(FDIR, "warehouse/additional_links.parquet"))
    def load_additional_title_links(self):

        entities = self.load_entity_map()
        articles = self.load_articles()
        articles = articles.filter(articles.ns == 0)
        return articles \
            .select(
            articles.id,
            articles.title
        ) \
            .join(
            entities,
            articles.id == entities.article_id,
            'left'
        ) \
            .drop('id', 'article_id') \
            .withColumn('source_article_id', lit(0)) \
            .select(
            col('source_article_id'),
            lower(articles.title).alias('anchor'),
            col('entity_id')
        )

    """
        +-----------------+----------------+---------+
        |source_article_id|          anchor|entity_id|
        +-----------------+----------------+---------+
        |          8199349|   !Oye Esteban!| 10783753|
        |         26179117|      "B" reader| 36721838|
        |         15879409|"Pine Top" Smith| 55156192|
        |         14739135|       Ric Flair|  4460010|
        |          1053291|       Ric Flair|  4460010|
        +-----------------+----------------+---------+
        """

    @warehouse(os.path.join(FDIR, "warehouse/raw_entity_links.parquet"))
    def load_raw_entity_links(self):
        entities = self.load_entities()
        links = self.load_article_links()
        links = links.join(entities, links.entity_title == entities.title, 'left')
        return links \
            .filter(links.ns == 0) \
            .filter(links.title.isNotNull()) \
            .select(
            links.id.alias('source_article_id'),
            links.anchor.alias('anchor'),
            entities.entity_id
        )

    """
    +-----------------+--------------------+---------+                              
    |source_article_id|              anchor|entity_id|
    +-----------------+--------------------+---------+
    |         22016145|  "galway joe" dolan| 25311407|
    |         55274543|  "galway joe" dolan| 25311407|
    |         58118186|  "galway joe" dolan| 25311407|
    +-----------------+--------------------+---------+
    """

    @warehouse(os.path.join(FDIR, "warehouse/entity_links.parquet"))
    def load_entity_links(self):
        entities = self.load_entities()
        links = self.load_article_links()
        links = links.join(entities, links.entity_title == entities.title, 'left')


        links = links \
            .filter(links.ns == 0) \
            .filter(links.title.isNotNull()) \
            .select(
            links.id.alias('source_article_id'),
            lower(links.anchor).alias('anchor'),
            entities.entity_id
        )

        additional_links = self.load_additional_title_links()
        links = links.union(additional_links)
        links.show(100, False)
        return links

    """
    +----------------------------------------------------+-------------+
    |anchor_text                                         |anchor_id    |
    +----------------------------------------------------+-------------+
    |1943 Central Java earthquakes                       |1056561954817|
    |1949 National Invitation Tournament                 |1056561954818|
    |1990 Tayside Regional Council election              |1056561954820|
    |1991 Blockbuster Bowl                               |1056561954821|
    |2010 All Stars Match                                |1056561954825|
    +----------------------------------------------------+-------------+
    """

    @warehouse(os.path.join(FDIR, "warehouse/anchors_and_ids.parquet"))
    def load_anchors_ids(self):
        """
        :return:
        """
        entity_links = self.load_entity_links()
        anchors = entity_links \
            .select(entity_links.anchor) \
            .distinct().withColumn("anchor_id", monotonically_increasing_id()) \
            .toDF("anchor_text", "anchor_id")
        return anchors

    @warehouse(os.path.join(FDIR, "warehouse/filtered_anchors_and_ids.parquet"))
    def load_filtered_anchors_ids(self):
        anchors = self.load_anchors_ids()
        anchors = anchors.filter(~anchors.anchor_text.isin(*stopwords))
        return anchors

    """
    +-----------------+---------+------------+
    |source_article_id|entity_id|   anchor_id|
    +-----------------+---------+------------+
    |            19727| 31185171|506806195496|
    |          4371710| 24349911|506806160531|
    |         50689681| 38270798|506806194940|
    |         19848420| 24914788|506806199849|
    |          3219130| 50141358|506806212218|
    +-----------------+---------++------------+
    """

    @warehouse(os.path.join(FDIR, "warehouse/loaded.parquet"))
    def load(self):
        entity_links = self.load_entity_links()
        anchors = self.load_anchors_ids()

        links = entity_links \
            .join(anchors, entity_links.anchor == anchors.anchor_text, 'left')
        return links.select(links.source_article_id.cast(IntegerType()), links.entity_id.cast(IntegerType()),
                            links.anchor_id)

    """
    +---------+---------+-----+-----+-------------------+                           
    |anchor_id|entity_id|count|total|         commonness|
    +---------+---------+-----+-----+-------------------+
    |       26|  1113865|   64|   64|                1.0|
    |       29|  1113899|   36|   36|                1.0|
    |       65| 18740534|    2|    3| 0.6666666666666666|
    |       65| 22645402|    1|    3| 0.3333333333333333|
    +---------+---------+-----+-----+-------------------+ 
    """
    @warehouse(os.path.join(FDIR, "warehouse/commonness_counts.parquet"))
    def load_counted_commonness(self):
        e = self.load()
        e = e.select(e.anchor_id, e.entity_id).groupBy(
            e.anchor_id,
            e.entity_id
        ).count()
        e = e.withColumn('total', f.sum(col('count')).over(Window.partitionBy('anchor_id')))
        e = e.withColumn('commonness', col('count') / col('total'))
        return e

    @warehouse(os.path.join(FDIR, "warehouse/commonness.parquet"))
    def load_commonness(self):
        e = self.load_counted_commonness()
        e = e.select(e.entity_id, e.anchor_id, e.commonness)
        return e

    """
    +---------+-----+-----+----------+---------+--------------------+               
    |entity_id|count|total|commonness|max_count|         anchor_text|
    +---------+-----+-----+----------+---------+--------------------+
    | 46984499|    1|    1|       1.0|        1|adagio city apart...|
    +---------+-----+-----+----------+---------+--------------------+
    """
    @warehouse(os.path.join(FDIR, "warehouse/max_commonness.parquet"))
    def load_max_commonness(self):
        e = self.load_counted_commonness()
        e = e.withColumn('max_count', f.max(col('count')).over(Window.partitionBy('anchor_id')))
        e = e.filter(col('count') == e.max_count)
        e.drop('count', 'max_count')
        a = self.load_anchors_ids().withColumnRenamed("anchor_id", "a_id")
        j = e.join(a, e.anchor_id == a.a_id, 'left')
        j = j.drop('a_id', 'anchor_id')
        return j


    """
    +---------+-----+------------------+--------+-------------------+--------------------+
    |entity_id|total|        commonness|max_diff|relative_commonness|         anchor_text|
    +---------+-----+------------------+--------+-------------------+--------------------+
    |   243074|   12|               1.0|      12|                1.0|          (s)-malate|
    +---------+-----+------------------+--------+-------------------+--------------------+
    """
    @warehouse(os.path.join(FDIR, "warehouse/max_relative_commonness.parquet"))
    def load_max_relative_commonness(self):
        e = self.load_counted_commonness()
        e = e.withColumn('max_count', f.max(col('count')).over(Window.partitionBy('anchor_id')))
        e = e.withColumn('second_max_count', second_max_diff(col('count'), e.max_count))
        e = e.withColumn('max_diff', f.min(col('second_max_count')).over(Window.partitionBy('anchor_id')))
        e = e.withColumn('relative_commonness', col('max_diff') / col('total'))
        e = e.filter(col('count') == e.max_count)
        a = self.load_anchors_ids().withColumnRenamed("anchor_id", "a_id")
        j = e.join(a, e.anchor_id == a.a_id, 'left')
        j = j\
            .drop('a_id', 'anchor_id', 'count', 'max_count', 'second_max_count')\
            .filter(j.anchor_text != '')\
            .filter(j.entity_id.isNotNull())
        return j

    def pickle_spot_map(self):
        sm = self.load_redw_spot_map()
        sm.show(10)
        rows = [list(row) for row in sm.collect()]
        d = {}
        for i in rows:
            if i[0] in d:
                d[i[0]].append(i[1])
            else:
                d[i[0]] = [i[1]]
        with open("warehouse/spotMap.pickle", 'wb') as fp:
            pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_spot_map_sr(self):
        sm = self.load_redw_sr().na.fill(value=0.0)
        sm.show(10)
        print('Pickling warehouse/spotMapSR.pickle')
        rows = [list(row) for row in sm.collect()]
        d = {}
        for i in rows:
            d[i[0]] = {
                'id': i[1],
                'SR': i[2],
                'SR_norm': i[3],
                'SR_min_max_norm': i[4],
            }
        with open("warehouse/spotMapSR.pickle", 'wb') as fp:
            pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)

    @warehouse(os.path.join(FDIR, "warehouse/anchor_commonness.parquet"))
    def load_anchor_commonness(self):
        c = self.load_commonness()
        a = self.load_anchors_ids()
        return c.join(a, c.anchor_id == a.anchor_id, 'left').drop('anchor_id')

    def pickle_commonness(self):
        j = self.load_commonness()
        c = j.rdd.toLocalIterator(prefetchPartitions=True)
        d = {}
        for row in c:
            if row['anchor_id'] in d:
                d[row['anchor_id']][row['entity_id']] = row['commonness']
            else:
                d[row['anchor_id']] = {row['entity_id']: row['commonness']}
        print('Pickling warehouse/commonness.pickle')
        with open("warehouse/commonness.pickle", 'wb') as fp:
            pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_anchor_commonness(self):
        j = self.load_anchor_commonness()
        c = j.rdd.toLocalIterator(prefetchPartitions=True)
        d = {}
        for row in c:
            if row['anchor_text'] in d:
                d[row['anchor_text']][row['entity_id']] = row['commonness']
            else:
                d[row['anchor_text']] = {row['entity_id']: row['commonness']}
        print('Pickling warehouse/commonness.pickle')
        with open("warehouse/anchor_commonness.pickle", 'wb') as fp:
            pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_max_commonness(self):
        j = self.load_max_commonness()
        c = j.collect()
        d = {}
        for row in c:
            d[row['anchor_text']] = {'id': int(row['entity_id']), 'commonness': float(row['commonness']), 'total': int(row['total'])}
        print('Pickling warehouse/commonness.pickle')
        with open("warehouse/max_commonness.pickle", 'wb') as fp:
            pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_max_relative_commonness(self):
        j = self.load_max_relative_commonness()
        c = j.collect()
        d = {}
        for row in c:
            d[row['anchor_text']] = {'id': int(row['entity_id']), 'commonness': float(row['commonness']),'relative_commonness': float(row['relative_commonness']), 'total': int(row['total'])}
        print('Pickling warehouse/max_relative_commonness.pickle')
        with open("warehouse/max_relative_commonness.pickle", 'wb') as fp:
            pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_anchors_ids(self):
        a = self.load_anchors_ids()
        c = a.collect()
        rows = [list(row) for row in c]
        d = {}
        for i in rows:
            d[i[0]] = i[1]
        with open("warehouse/anchors_ids.pickle", 'wb') as fp:
            pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    etl = ETL()
    etl.pickle_spot_map_sr()

