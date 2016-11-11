import java.util

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import pl.sgjp.morfeusz.{Morfeusz, MorfeuszUsage, MorphInterpretation}

import scala.collection.{JavaConversions, mutable}

case class AdvertisementAndBagOfWords(advertisementId: Long, bagOfWordsId: Long)

case class Advertisement(id: Long, category1: Long, category2: Long, category3: Option[Long])

case class BagOfWords(id: Long, word: String)

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
        val trainingRDD = sc.textFile("data/training.[0-9]*.tsv")

        //
        var i = 0
        val bagOfWordsDS = trainingRDD.map(string => string.split("\t"))
                .map(stringArray => processTitle(stringArray(1)))
                .flatMap(words => words)
                .map(word => {
                    i += 1
                    BagOfWords(i, word)
                })
                .toDS()
                .dropDuplicates("word")
        val bagOfWords = bagOfWordsDS.collect()

        //
        case class AdvertisementTmp(id: Long, words: Array[String])
        val advertisementBagOfWordsDS = trainingRDD.map(string => string.split("\t"))
                .map(stringArray => AdvertisementTmp(stringArray(0).toLong, processTitle(stringArray(1))))
                .flatMap(
                    advertisementTmp => advertisementTmp.words.map(
                        word => AdvertisementAndBagOfWords(
                            advertisementTmp.id, bagOfWords.find(bagOfWords => bagOfWords.word == word).get.id
                        )
                    )
                )
                .toDS()

        //
        val advertisementDS = trainingRDD.map(string => string.split("\t"))
                .map(
                    stringArray => Advertisement(stringArray(0).toLong, stringArray(4).toLong, stringArray(5).toLong, if (stringArray.length == 7) Option(stringArray(6).toLong) else None)
                )
                .toDS()
        //
        bagOfWordsDS.createOrReplaceTempView("bag_of_words")
        advertisementBagOfWordsDS.createOrReplaceTempView("advertisement_bag_of_words")
        advertisementDS.createOrReplaceTempView("advertisement")

        sqlContext.sql("SELECT DISTINCT COUNT(1) FROM bag_of_words").show()
        sqlContext.sql("SELECT word FROM bag_of_words").show()
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
        this.synchronized {
            val result = morfeusz.analyseAsList(title)

            val withoutTags = removeByTags(result)

            val withoutStopWords = withoutTags.filter(res => !stopWords.contains(res.getOrth))

            getFirstLemmaFromEveryNode(withoutStopWords)
        }
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

    def disableLogging(): Unit = {
        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)
    }
}