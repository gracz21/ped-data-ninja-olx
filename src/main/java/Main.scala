import java.util
import java.util.Properties

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Dataset, Row, SaveMode, SparkSession}
import org.apache.spark.sql.functions.{asc, desc, col}
import pl.sgjp.morfeusz.{Morfeusz, MorfeuszUsage, MorphInterpretation}

import scala.collection.mutable.ArrayBuffer
import scala.collection.{JavaConversions, Set, mutable}

case class Advertisement(id: Long, title: String, category1: Long, category2: Option[Long], category3: Option[Long])

case class BagOfWords(advertisementId: Long, word: String)

case class Category(id: Long, name: String)

case class Neighbour(idA: Long, category1A: Long, category2A: Option[Long], category3A: Option[Long],
                     idB: Long, category1B: Long, category2B: Option[Long], category3B: Option[Long], similarity: Double)

object Main extends java.io.Serializable {
    disableLogging()

    val sparkSession = SparkSession.builder.
            master("local[*]")
            .appName("NinjaCosTam")
            .config("spark.ui.showConsoleProgress", false)
            .config("spark.sql.crossJoin.enabled", true)
            .config("spark.sql.broadcastTimeout", 999999999)
            .getOrCreate()
    val sc = sparkSession.sparkContext
    val sqlContext = sparkSession.sqlContext

    import sqlContext.implicits._

    val morfeusz = Morfeusz.createInstance(MorfeuszUsage.ANALYSE_ONLY)

    val stopWords = sc.textFile("data/polish_stop_words.txt").collect()

    def main(args: Array[String]): Unit = {
        time {
//            val trainingRDD = sc.textFile("data/test.tsv")
                    val trainingRDD = sc.textFile("data/training.[0-9]*.tsv")
//                            val trainingRDD = sc.textFile("data/training.0001.tsv")

            //
            case class AdvertisementTmp(id: Long, words: Array[String])
            val bagOfWordsDS = trainingRDD.map(string => string.split("\t"))
                    .map(stringArray =>
                        AdvertisementTmp(stringArray(0).toLong, processTitle(stringArray(1)))
                    )
                    .flatMap(
                        advertisementTmp => advertisementTmp.words.map(
                            word => BagOfWords(advertisementTmp.id, word)
                        )
                    )
                    .toDS()

            //
            val advertisementDS = trainingRDD.map(string => string.split("\t"))
                    .map(
                        stringArray => Advertisement(
                            stringArray(0).toLong, stringArray(1),
                            stringArray(4).toLong,
                            if (!stringArray(5).isEmpty) Option(stringArray(5).toLong) else None,
                            if (stringArray.length == 7) Option(stringArray(6).toLong) else None
                        )
                    )
                    .toDS()

            //
            val categoryRDD = sc.textFile("data/categories.tsv")
            val categoryDS = categoryRDD
                    .zipWithIndex().filter((tuple: (String, Long)) => tuple._2 >= 2)
                    .map(string => string._1.split("\t"))
                    .map(
                        stringArray => Category(stringArray(0).toLong, stringArray(2))
                    )
                    .toDS()

            //
            bagOfWordsDS.createOrReplaceTempView("bag_of_words")
            advertisementDS.createOrReplaceTempView("advertisement")
            categoryDS.createOrReplaceTempView("category")
            sqlContext.cacheTable("advertisement")
            sqlContext.cacheTable("bag_of_words")
            sqlContext.cacheTable("category")

            val filteredAdvertisementDS = sqlContext.sql("SELECT * FROM advertisement WHERE id IN(SELECT DISTINCT advertisementId FROM bag_of_words WHERE word IN (SELECT word FROM bag_of_words GROUP BY word ORDER BY COUNT(word) DESC LIMIT 100))")
            filteredAdvertisementDS.createOrReplaceTempView("filtered_advertisement")
            sqlContext.cacheTable("filtered_advertisement")

            val filteredBagOfWordsDS = sqlContext.sql("SELECT * FROM bag_of_words WHERE advertisementId IN (SELECT id FROM filtered_advertisement) AND word IN (SELECT word FROM bag_of_words GROUP BY word ORDER BY COUNT(word) DESC LIMIT 100)")
            filteredBagOfWordsDS.createOrReplaceTempView("filtered_bag_of_words")
            sqlContext.cacheTable("filtered_bag_of_words")

            sqlContext.uncacheTable("advertisement")
            sqlContext.uncacheTable("bag_of_words")
        }

        // odpowiedzi
        time {
            exercise2_5_1_a()
        }
        time {
            exercise2_5_1_b()
        }
        time {
          exercise2_5_2()
        }
    }

    def exercise2_5_1_a(): Unit = {
        val features = sqlContext.sql("SELECT word, COUNT(word) FROM filtered_bag_of_words GROUP BY word ORDER BY COUNT(word) DESC LIMIT 30")
        features.foreach(row => {
            val feature = row.getString(0)
            val numberOfAdvertisements = row.getLong(1)

            val advertisementsDS = sqlContext.sql(s"SELECT category1, category2, category3 FROM filtered_bag_of_words JOIN filtered_advertisement ON advertisementId=id WHERE word='$feature'")

            val categoryCounter: Counter = new Counter()
            categoryCounter.addCategory(advertisementsDS.collect())

            val categories = categoryCounter.getKeys
            val sumOfCategories = categoryCounter.getSumOfValues
            val sortedCategories = categoryCounter.getSorted

            val categoriesNames: Set[String] = getCategoriesNames(categories)

            println(f"$feature | $sumOfCategories | ${categoriesNames.mkString(", ")} | ${getCategoryName(sortedCategories.head._1)} ${sortedCategories.head._2.toDouble / numberOfAdvertisements}, ${getCategoryName(sortedCategories(1)._1)} ${sortedCategories(1)._2.toDouble / numberOfAdvertisements}, ${getCategoryName(sortedCategories(2)._1)} ${sortedCategories(2)._2.toDouble / numberOfAdvertisements}")
        })
    }

    def getCategoriesNames(categories: Set[Long]): Set[String] = {
        categories.map(category => getCategoryName(category))
    }

    def getCategoryName(category: Long): String = {
        sqlContext.sql(s"SELECT name FROM category WHERE id='$category'").collect()(0).getString(0)
    }

    def exercise2_5_1_b(): Unit = {
        val featuresDS = sqlContext.sql("SELECT DISTINCT word FROM filtered_bag_of_words")
        val features = featuresDS.collect().map(row => row.getString(0))

        val featuresCombinations = features.combinations(2).toList

        val combinationsWithCounter: mutable.Map[(String, String), Seq[Long]] = mutable.Map()

        var i = 0
        println(s"Combinations: ${featuresCombinations.length}")
        featuresCombinations.foreach(combination => {
            i = i + 1
            println(s"Combination: $i")

            val firstWord = combination(0)
            val secondWord = combination(1)
            val filteredBagOfWordsDS = sqlContext.sql(s"SELECT advertisementId FROM filtered_bag_of_words WHERE word = '$firstWord' OR word = '$secondWord'")

            val counter = new Counter()
            counter.addBagOfWords(filteredBagOfWordsDS.collect())

            val advertisementsIdsWithCounterEqual2 = counter.getKeysWithCounterEqual2

            combinationsWithCounter((combination(0), combination(1))) = advertisementsIdsWithCounterEqual2
        })

        val combinationsWithCounterSorted30 = combinationsWithCounter
                .toSeq
                .sortWith((tuple: ((String, String), Seq[Long]), tuple0: ((String, String), Seq[Long])) => tuple._2.length > tuple0._2.length)
                .take(30)

        combinationsWithCounterSorted30.foreach(combinationMapItem => {
            val combinationWords = combinationMapItem._1
            val combinationAdvertisementId = combinationMapItem._2

            val advertisementsDS = sqlContext.sql(s"SELECT category1, category2, category3 FROM filtered_advertisement WHERE id IN (${combinationAdvertisementId.mkString(", ")})")
            val advertisementsRows = advertisementsDS.collect()
            val numberOfAdvertisements = advertisementsRows.length

            val categoryCounter: Counter = new Counter()
            categoryCounter.addCategory(advertisementsRows)

            val categories = categoryCounter.getKeys
            val sumOfCategories = categoryCounter.getSumOfValues
            val sortedCategories = categoryCounter.getSorted

            val categoriesNames: Set[String] = getCategoriesNames(categories)

            println(f"$combinationWords._1, $combinationWords._2 | $sumOfCategories | ${categoriesNames.mkString(", ")} | ${getCategoryName(sortedCategories.head._1)} ${sortedCategories.head._2.toDouble / numberOfAdvertisements}, ${getCategoryName(sortedCategories(1)._1)} ${sortedCategories(1)._2.toDouble / numberOfAdvertisements}, ${getCategoryName(sortedCategories(2)._1)} ${sortedCategories(2)._2.toDouble / numberOfAdvertisements}")
        })
    }

    def exercise2_5_2(): Unit = {
      sqlContext.sql("SELECT advertisementId id, COUNT(*) summ FROM filtered_bag_of_words GROUP BY advertisementId").createOrReplaceTempView("words_sum")
      sqlContext.cacheTable("words_sum")

      sqlContext.sql("SELECT id FROM filtered_advertisement LIMIT 1000").createOrReplaceTempView("advertisementA")
      sqlContext.cacheTable("advertisementA")

      sqlContext.sql("SELECT idA, idB, sim FROM " +
        "(SELECT idA, idB, sim, ROW_NUMBER() OVER (PARTITION BY idA ORDER BY sim DESC) rank FROM " +
        "(SELECT a.id idA, b.id idB, COALESCE(allInter.inter, 0)/(summA.summ + summB.summ - COALESCE(allInter.inter, 0)) sim FROM " +
        "filtered_advertisement b JOIN advertisementA a ON a.id <> b.id LEFT JOIN " +
        "(SELECT c.advertisementId idA, d.advertisementId idB, COUNT(*) inter FROM (SELECT * FROM filtered_bag_of_words WHERE advertisementId in (SELECT id FROM advertisementA)) c JOIN filtered_bag_of_words d ON c.word = d.word GROUP BY c.advertisementId, d.advertisementId) allInter " +
        "ON a.id = allInter.idA AND b.id = allInter.idB JOIN " +
        "words_sum summA ON a.id = summA.id JOIN " +
        "words_sum summB ON b.id = summB.id)) " +
        "WHERE rank <= 10")
          .createOrReplaceTempView("similarity")

        sqlContext.cacheTable("similarity")

      sqlContext.sql(s"SELECT a.id idA, a.category1 category1A, a.category2 category2A, a.category3 category3A, " +
        s"b.id idB, b.category1 category1B, b.category2 category2B, b.category3 category3B, c.sim similarity FROM similarity c JOIN " +
        s"filtered_advertisement a ON c.idA = a.id JOIN filtered_advertisement b ON c.idB = b.id")
        .as[Neighbour]
        .foreach(neighbour => println(f"${neighbour.idA} | ${neighbour.category1A} | ${neighbour.category2A.getOrElse("None")} | ${neighbour.category3A.getOrElse("None")}" +
          f" | ${neighbour.idB} | ${neighbour.category1B} | ${neighbour.category2B.getOrElse("None")} | ${neighbour.category3B.getOrElse("None")} | ${neighbour.similarity}"))
  }

    class Counter {
        val map: mutable.Map[Long, Long] = mutable.Map()

        def addCategory(category: Array[Row]): Unit = {
            category.foreach((row: Row) => {
                val categories = ArrayBuffer[Long]()
                if (!row.isNullAt(0)) categories += row.getLong(0)
                if (!row.isNullAt(1)) categories += row.getLong(1)
                if (!row.isNullAt(2)) categories += row.getLong(2)

                categories.foreach((category: Long) => {
                    val counter: Long = map.getOrElse(category, 0L)
                    val newCounter = counter + 1
                    map(category) = newCounter
                })
            })
        }

        def addBagOfWords(bagOfWords: Array[Row]): Unit = {
            bagOfWords.foreach((row: Row) => {
                val advertisementId = row.getLong(0)

                val counter: Long = map.getOrElse(advertisementId, 0L)
                val newCounter = counter + 1
                map(advertisementId) = newCounter
            })
        }

        def getSorted: Seq[(Long, Long)] = {
            map.toSeq.sortWith((tuple: (Long, Long), tuple0: (Long, Long)) => tuple._2 > tuple0._2)
        }

        def getKeysWithCounterEqual2: Seq[Long] = {
            map.toSeq.filter((tuple: (Long, Long)) => tuple._2 == 2).map((tuple: (Long, Long)) => tuple._1)
        }

        def getKeys: Set[Long] = {
            map.keySet
        }

        def getSumOfValues: Long = {
            map.values.sum
        }
    }


    // znaczenie http://nkjp.pl/poliqarp/help/plse2.html
    // plik w morfeusz-sgjp.tagset w zrodlach (zawiera id tagow)
    // qub - 592 - np moglbym dzielione jest na moc i by (wywalamy by)
    // dig - 151
    // ign - 0
    // interp - 235
    // ppron12 - <429, 461> i 800
    // ppron3 - <462, 508>
    // winien - <705, 723>
    // siebie - <594, 598>
    // pred - 573
    // prep - <578, 591>
    // conj - 148
    // comp - 99
    // brev - 97, 801
    // TODO moze przyslowki i przymiotniki tez wypierdolic?
    def processTitle(title: String): Array[String] = {
        var result: util.List[MorphInterpretation] = null
        this.synchronized {
            result = morfeusz.analyseAsList(title)
        }

        val withoutTags = removeByTags(result)

        val withoutStopWords = withoutTags.filter(res => !stopWords.contains(res.getOrth))

        val lemmas = getFirstLemmaFromEveryNode(withoutStopWords)

        lemmas.distinct // usuniecie duplikatow - w tytule pojawiaja sie dwa razy te same slowo
    }

    def removeByTags(result: util.List[MorphInterpretation]): mutable.Buffer[MorphInterpretation] = {
        val filteredResult = JavaConversions.asScalaBuffer(result)
                .filter(res =>
                    !(res.getTagId == 592 ||
                            res.getTagId == 151 ||
                            res.getTagId == 0 ||
                            res.getTagId == 235 ||
                            (res.getTagId >= 429 && res.getTagId <= 461) || res.getTagId == 800 ||
                            (res.getTagId >= 462 && res.getTagId <= 508) ||
                            (res.getTagId >= 705 && res.getTagId <= 723) ||
                            (res.getTagId >= 594 && res.getTagId <= 598) ||
                            res.getTagId == 573 ||
                            (res.getTagId >= 578 && res.getTagId <= 591) ||
                            res.getTagId == 148 ||
                            res.getTagId == 99 ||
                            res.getTagId == 97 || res.getTagId == 801)
                )

        filteredResult
    }

    def getFirstLemmaFromEveryNode(filteredResult: mutable.Buffer[MorphInterpretation]): Array[String] = {
        var lemmas: Map[Int, String] = Map()

        filteredResult.foreach(morph => {
            val startNode = morph.getStartNode

            if (!lemmas.contains(startNode)) {
                lemmas += (startNode -> morph.getLemma.split(":")(0)) // wywalam to co po dwukropku
            }
        })

        lemmas.values.toArray
    }

    def saveInDatabase(bagOfWordsDS: Dataset[BagOfWords], advertisementDS: Dataset[Advertisement]): Unit = {
        //        Class.forName("org.postgresql.Driver")
        //        val connectionProperties = new Properties()
        //        connectionProperties.put("user", "postgres")
        //        connectionProperties.put("password", "postgres")

        Class.forName("org.sqlite.JDBC")

        advertisementDS.write.mode(SaveMode.Append).jdbc("jdbc:sqlite:data_ninja.db", "advertisement", new Properties())
        //        bagOfWordsDS.write.jdbc("jdbc:postgresql://localhost:5432/data_ninja", "public.bag_of_words", connectionProperties)
    }

    def disableLogging(): Unit = {
        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)
    }

    def time[R](block: => R): R = {
        val t0 = System.nanoTime()
        val result = block // call-by-name
        val t1 = System.nanoTime()
        println("Elapsed time: " + (t1 - t0) + "ns")
        result
    }
}