import java.util
import java.util.Properties

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Dataset, Row, SaveMode, SparkSession}
import pl.sgjp.morfeusz.{Morfeusz, MorfeuszUsage, MorphInterpretation}

import scala.collection.mutable.ArrayBuffer
import scala.collection.{JavaConversions, mutable}

case class Advertisement(id: Long, title: String, category1: Long, category2: Option[Long], category3: Option[Long])

case class BagOfWords(advertisementId: Long, word: String)

object Main extends java.io.Serializable {
    disableLogging()

    val sparkSession = SparkSession.builder.
            master("local[*]")
            .appName("NinjaCosTam")
            .getOrCreate()
    val sc = sparkSession.sparkContext
    val sqlContext = sparkSession.sqlContext

    import sqlContext.implicits._

    val morfeusz = Morfeusz.createInstance(MorfeuszUsage.ANALYSE_ONLY)

    val stopWords = sc.textFile("data/polish_stop_words.txt").collect()

    def main(args: Array[String]): Unit = {
        //        val trainingRDD = sc.textFile("data/test.tsv")
        val trainingRDD = sc.textFile("data/training.[0-9]*.tsv")

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
        bagOfWordsDS.createOrReplaceTempView("bag_of_words")
        advertisementDS.createOrReplaceTempView("advertisement")

        // odpowiedzi
        exercise5()
        exercise7()

        //        saveInDatabase(bagOfWordsDS, advertisementDS)
    }

    def exercise5(): Unit = {
        println("Zadanie 5")

        val numberOfFeatures = sqlContext.sql("SELECT COUNT(1) AS LiczbaCech FROM (SELECT DISTINCT word FROM bag_of_words)")
                .collect()(0).getLong(0)
        println(s"Liczba cech: $numberOfFeatures")

        val density = sqlContext.sql(s"SELECT AVG(gestosc) FROM (SELECT advertisementId, COUNT(1)/$numberOfFeatures AS gestosc FROM bag_of_words GROUP BY advertisementId)")
                .collect()(0).getDouble(0)
        println(f"Gestosc zbioru: $density%.6f")
    }

    def exercise7(): Unit = {
        println("Zadanie 7")

        // pierwsze zdanie
        sqlContext.sql("SELECT word, COUNT(word) AS LiczbaOgloszen FROM bag_of_words GROUP BY word ORDER BY COUNT(word) DESC").show(50)

        // drugie zdanie
        val features = sqlContext.sql("SELECT word, COUNT(word) AS LiczbaOgloszen FROM bag_of_words GROUP BY word ORDER BY COUNT(word) DESC LIMIT 50")
        features.foreach(row => {
            val feature = row.getString(0)
            val numberOfAdvertisements = row.getString(0)

            val advertisementsDS = sqlContext.sql(s"SELECT category1, category2, category3 FROM bag_of_words JOIN advertisement ON advertisementId=id WHERE word='$feature'")

            val categoryCounter: CategoryCounter = new CategoryCounter()
            val advertisements: Array[Row] = advertisementsDS.collect()
            categoryCounter.add(advertisements)

            val theMostFrequentCategory = categoryCounter.getTheMostFrequentCategory
            val categoryId = theMostFrequentCategory._1
            val counter = theMostFrequentCategory._2

            val frequency = counter / advertisements.length.toDouble

            println(f"Slowo: $feature | najczesciej wystepujaca kategoria: $categoryId | liczba wystapien $counter | czestotliwosc wystepowania kategorii: $frequency%.6f")
        })
    }

    class CategoryCounter {
        val map: mutable.Map[Long, Long] = mutable.Map()

        def add(category: Array[Row]): Unit = {
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

        def getTheMostFrequentCategory: (Long, Long) = {
            var theMostFrequentItem: (Long, Long) = null

            map.foreach((tuple: (Long, Long)) => {
                if (theMostFrequentItem == null) {
                    theMostFrequentItem = tuple
                } else {
                    if (tuple._2 > theMostFrequentItem._2) {
                        theMostFrequentItem = tuple
                    }
                }
            })

            theMostFrequentItem
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
}