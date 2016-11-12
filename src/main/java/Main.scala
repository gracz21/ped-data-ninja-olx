import java.util
import java.util.Properties

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Dataset, SaveMode, SparkSession}
import pl.sgjp.morfeusz.{Morfeusz, MorfeuszUsage, MorphInterpretation}

import scala.collection.{JavaConversions, mutable}

case class Advertisement(id: Long, title: String)

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
                    stringArray => Advertisement(stringArray(0).toLong, stringArray(1))
                )
                .toDS()
        //
        bagOfWordsDS.createOrReplaceTempView("bag_of_words")
        advertisementDS.createOrReplaceTempView("advertisement")

        // 5 zadanie
        val liczbaCech = sqlContext.sql("SELECT COUNT(1) AS LiczbaCech FROM (SELECT DISTINCT word FROM bag_of_words)")
                .collect()(0).getLong(0)
        println(s"Liczba cech: $liczbaCech")

        val gestosc = sqlContext.sql(s"SELECT AVG(gestosc) FROM (SELECT advertisementId, COUNT(1)/$liczbaCech AS gestosc FROM bag_of_words GROUP BY advertisementId)")
                .collect()(0).getDouble(0)
        println(f"Gestosc zbioru: $gestosc%.6f")

//        saveInDatabase(bagOfWordsDS, advertisementDS)
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

        getFirstLemmaFromEveryNode(withoutStopWords)
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