import java.time.LocalDateTime
import java.util

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import pl.sgjp.morfeusz.{Morfeusz, MorfeuszUsage, MorphInterpretation}

import scala.collection.{JavaConversions, Map, mutable}
import scalax.io.{Output, Resource}

case class Advertisement(title: String, category: Long)

object Main extends java.io.Serializable {
    val output: Output = Resource.fromFile("results/result_" + LocalDateTime.now().toString)

    //    disableLogging()

    val sparkSession = SparkSession.builder.
            master("local[*]")
            .appName("NinjaCosTam")
            .config("spark.executor.memory", "6g")
            .getOrCreate()
    val sc = sparkSession.sparkContext

    val morfeusz = Morfeusz.createInstance(MorfeuszUsage.ANALYSE_ONLY)

    val stopWords = sc.textFile("data/polish_stop_words.txt").collect()

    def main(args: Array[String]): Unit = {
        disableLogging()

        //        val data = sc.textFile("data/test.tsv")
                val data = sc.textFile("data/training.[0-9]*.tsv")
//        val data = sc.textFile("data/training.0001.tsv")

        val categoriesMap = data
                .map(string => string.split("\t"))
                .map(stringArray => stringArray(4).toInt)
                .distinct()
                .zipWithIndex()
                .collectAsMap()

        val convertedData = data
                .map(string => string.split("\t"))
                .map(stringArray =>
                    Advertisement(stringArray(1), categoriesMap(stringArray(4).toInt))
                )

        splitDataAndExecute(convertedData, categoriesMap.size, Seq(Array(0.6, 0.4)))
    }

    private def splitDataAndExecute(convertedData: RDD[Advertisement], categories: Int, splits: Seq[Array[Double]]) = {
        for (split <- splits) {
            output.write("Train=" + split(0) + ";Test=" + split(1) + "\n")
            val Array(training, test) = convertedData.randomSplit(split, 44)

            output.write("With TFIDF:\n")
            execute(training, test, categories, tfidf = true)
            output.write("Without TFIDF:\n")
            execute(training, test, categories, tfidf = false)
        }
    }

    private def execute(training: RDD[Advertisement], test: RDD[Advertisement], categories: Int, tfidf: Boolean) = {
        val trainDataSet = createLabeledPointDataSet(training, tfidf)
        val testDataSet = createLabeledPointDataSet(test, tfidf)

        naiveBayes(trainDataSet, testDataSet, Seq(0.1, 0.2, 0.3))
        logisticRegression(trainDataSet, testDataSet, categories, Seq((10, 0.01)))
    }

    private def naiveBayes(trainDataSet: RDD[LabeledPoint], testDataSet: RDD[LabeledPoint], lambdas: Seq[Double]) = {
        for (lambda <- lambdas) {
            var model: NaiveBayesModel = null
            val resultTime = time {
                model = NaiveBayes.train(trainDataSet, lambda = lambda)
            }

            output.write("NaiveBayes;label=" + lambda + ";")
            saveMetricsToFile(model, testDataSet)
            output.write(";" + resultTime)
            output.write("\n")
        }
    }

    private def logisticRegression(trainDataSet: RDD[LabeledPoint], testDataSet: RDD[LabeledPoint], categories: Int, parameters: Seq[(Int, Double)]) = {
        for (parameter <- parameters) {
            var model: LogisticRegressionModel = null
            val resultTime = time {
                val modelSettings = new LogisticRegressionWithLBFGS()
                        .setNumClasses(categories)
                modelSettings.optimizer.setNumIterations(parameter._1)
                modelSettings.optimizer.setRegParam(parameter._2)

                model = modelSettings.run(trainDataSet)
            }

            output.write("LogisticRegressionWithLBFGS;numberOfIterations=" + parameter._1 + "|regularization=" + parameter._2 + ";")
            saveMetricsToFile(model, testDataSet)
            output.write(";" + resultTime)
            output.write("\n")
        }
    }

    private def saveMetricsToFile(model: ClassificationModel, testDataSet: RDD[LabeledPoint]) = {
        val predictionAndLabel = testDataSet.map(p => (model.predict(p.features), p.label))
        val metrics = new MulticlassMetrics(predictionAndLabel)

        output.write(metrics.accuracy + ";"
                + metrics.weightedFMeasure + ";"
                + metrics.weightedPrecision + ";"
                + metrics.weightedRecall + ";"
                + metrics.weightedTruePositiveRate + ";"
                + metrics.weightedFalsePositiveRate)
    }

    def createLabeledPointDataSet(dataSet: RDD[Advertisement], tfidf: Boolean): RDD[LabeledPoint] = {
        val categories = dataSet
                .map(advertisement => advertisement.category)

        val tokens = dataSet
                .map(advertisement => processTitle(advertisement.title))

        val hashingTF = new HashingTF()
        var tf = hashingTF.transform(tokens)

        if (tfidf) {
            val idf = new IDF().fit(tf)
            tf = idf.transform(tf)
        }

        val zipped = categories.zip(tf)
        val labelPointDataSet = zipped.map { case (category, vector) => LabeledPoint(category, vector) }
        labelPointDataSet.cache

        labelPointDataSet
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
    def processTitle(title: String): Seq[String] = {
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

    def getFirstLemmaFromEveryNode(filteredResult: mutable.Buffer[MorphInterpretation]): Seq[String] = {
        var lemmas: Map[Int, String] = Map()

        filteredResult.foreach(morph => {
            val startNode = morph.getStartNode

            if (!lemmas.contains(startNode)) {
                lemmas += (startNode -> morph.getLemma.split(":")(0)) // wywalam to co po dwukropku
            }
        })

        lemmas.values.toSeq
    }

    def disableLogging(): Unit = {
        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)
    }

    def time[R](block: => R): Double = {
        val t0 = System.nanoTime()
        val result = block
        // call-by-name
        val t1 = System.nanoTime()
        //        println("Elapsed time: " + (t1 - t0) + "ns")
        //        result

        (t1 - t0) / 1000000000d
    }
}