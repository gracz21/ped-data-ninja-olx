import java.text.SimpleDateFormat
import java.util
import java.util.Date

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, SparseVector => SV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import pl.sgjp.morfeusz.{Morfeusz, MorfeuszUsage, MorphInterpretation}

import scala.collection.{JavaConversions, Map, mutable}
import scalax.io.{Output, Resource}

object MainCompetition extends java.io.Serializable {
    val output: Output = Resource.fromFile("result_" + new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss-SSS").format(new Date()))

    //    disableLogging()

    val sparkSession = SparkSession.builder
            .master("local[*]")
            .appName("NinjaCosTam")
            .config("spark.executor.memory", "32g")
            .config("spark.driver.memory", "32g")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.port", "36485")
            //                                    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            //                        .config("spark.kryoserializer.buffer.max.mb", "2047m")
            .config("spark.default.parallelism", "12")
            //            .config("spark.cores.max", "56")
            //            .config("spark.storage.memoryFraction", "0.7")
            //            .config("spark.io.compression.codec", "lzf")
            //            .config("spark.shuffle.consolidateFiles", "true")
            //            .config("spark.shuffle.service.enabled", "true")
            //            .config("spark.rdd.compress", "true")
            .config("spark.memory.fraction", "1")
            .config("spark.storage.memoryFraction", "0.1")
            .getOrCreate()
    val sc = sparkSession.sparkContext

    val morfeusz = Morfeusz.createInstance(MorfeuszUsage.ANALYSE_ONLY)

    val stopWords = sc.textFile("data/polish_stop_words.txt").collect()

    def main(args: Array[String]): Unit = {
        disableLogging()

        val training = sc.textFile("data/training.[0-9]*.tsv")
        val test = sc.textFile("data/test.[0-9]*.tsv")

        val testaa = test.first()

        val categoriesFile = sc.textFile("data/categories.tsv")
        val categoriesHierarchyMap = categoriesFile
                .zipWithIndex().filter((tuple: (String, Long)) => tuple._2 > 1)
                .map(string => string._1.split("\t"))
                .map(
                    stringArray => (stringArray(0).toInt, stringArray(1).toInt)
                )
                .collectAsMap()
        val categories = categoriesHierarchyMap.keySet.toArray.sorted

        val errorMatrix = createErrorMatrix(categoriesHierarchyMap, categories)
        val frequencies = createFrequencyList(training, categoriesHierarchyMap, categories)
        val frequenciesWithWeight = calculateWeightList(errorMatrix, frequencies)

        val allCategoriesMap = categories
                .zipWithIndex
                .toMap

        val modelCategoriesMap = Map(
            0 -> createCategoryMap(training, 0),
            1 -> createCategoryMap(training, 1),
            2 -> createCategoryMap(training, 2)
        )
        val convertedTraining = training
                .map(string => string.split("\t"))
                .filter(stringArray => stringArray.length == 7)
                .filter(stringArray => !stringArray(5).isEmpty) // 4
                .map(stringArray =>
            Advertisement(
                stringArray(0).toLong,
                stringArray(1),
                Seq(
                    modelCategoriesMap(0)(stringArray(4).toInt),
                    modelCategoriesMap(1)(stringArray(5).toInt),
                    modelCategoriesMap(2)(stringArray(6).toInt)
                )
            )
        )
        val convertedTest = test
                .map(string => string.split("\t"))
                .map(stringArray =>
                    Advertisement(
                        stringArray(0).toLong,
                        stringArray(1),
                        Seq(0, 0, 0)
                    )
                )

        executeForAllCategories(convertedTraining, convertedTest, allCategoriesMap, modelCategoriesMap, errorMatrix)
    }

    private def createCategoryMap(data: RDD[String], category: Int) = {
        data
                .map(string => string.split("\t"))
                .filter(stringArray => stringArray.length == 7)
                .filter(stringArray => !stringArray(4 + category).isEmpty)
                .map(stringArray => stringArray(4 + category).toInt)
                .distinct()
                .zipWithIndex()
                .collectAsMap()
    }

    private def createFrequencyList(data: RDD[String], categoriesHierarchyMap: Map[Int, Int], categories: Array[Int]): Array[Double] = {
        val frequencyList = Array.ofDim[Int](categoriesHierarchyMap.size)

        val dataCategories = data
                .map(string => string.split("\t"))
                .map(stringArray => stringArray.last.toInt)
                .collect()

        dataCategories
                .foreach(category => {
                    var currentCategory = -1
                    do {
                        if (currentCategory == -1)
                            currentCategory = category
                        else
                            currentCategory = categoriesHierarchyMap(currentCategory)

                        frequencyList(categories.indexOf(currentCategory)) += 1
                    } while (categoriesHierarchyMap(currentCategory) != 0)
                })

        frequencyList
                .map(value => value.toDouble / dataCategories.length)
    }

    private def calculateWeightList(errorMatrix: Array[Array[Int]], frequencyList: Array[Double]): Array[Double] = {
        frequencyList
                .zipWithIndex
                .map(row => row._1 * errorMatrix(row._2).length / errorMatrix(row._2).sum)
    }

    private def createErrorMatrix(categoriesHierarchyMap: Map[Int, Int], categories: Array[Int]): Array[Array[Int]] = {
        val errorMatrix = Array.ofDim[Int](categoriesHierarchyMap.size, categoriesHierarchyMap.size)

        categories.foreach(
            category => {
                val idx1 = categories.indexOf(category)
                errorMatrix(idx1)(idx1) = 0
                var path = Array.empty[Int]
                var currentCategory = category

                //Generate path from "predicted" category to root
                path = path :+ currentCategory
                while (categoriesHierarchyMap(currentCategory) != 0) {
                    path = path :+ currentCategory
                    currentCategory = categoriesHierarchyMap(currentCategory)
                }
                path = path :+ 0

                //Find start of common part with path from "predicted" category to root
                categories.filter(int => int != category).foreach(
                    correctCategory => {
                        val idx2 = categories.indexOf(correctCategory)
                        var errorValue = 0
                        var currentCategory2 = correctCategory
                        while (!path.contains(currentCategory2)) {
                            errorValue += 1
                            currentCategory2 = categoriesHierarchyMap(currentCategory2)
                        }
                        errorValue += path.indexOf(currentCategory2) * 2
                        errorMatrix(idx1)(idx2) = errorValue
                    }
                )
            }
        )
        errorMatrix
    }

    private def executeForAllCategories(training: RDD[Advertisement], test: RDD[Advertisement], allCategoriesMap: Map[Int, Int], modelCategoriesMap: Map[Int, Map[Int, Long]], errorMatrix: Array[Array[Int]]) = {
        val categoryIndex = 2

        val model = execute(training, categoryIndex, modelCategoriesMap(categoryIndex).size, tfidf = true)
        val testDataSet = createLabeledPointDataSet(test, categoryIndex, tfidf = true)

        val zipped = test.zip(testDataSet)

        val evaluatedTest = zipped.map(item => {
            (item._1.id, getCategoryIndex(modelCategoriesMap, categoryIndex, model.predict(item._2.features).toInt))
        })

        evaluatedTest.foreach(item => {
            println(s"${item._1}\t${item._2}")
        })
    }

    private def getCategoryIndex(modelCategoriesMap: Map[Int, Map[Int, Long]], categoryIndex: Int, category: Double) = {
        modelCategoriesMap(categoryIndex).map(_.swap).get(category.toInt).get
    }

    def calculateAccuracy(rdd: RDD[(Long, Double, Double)]): Double = {
        val accuracy = rdd.filter(x => x._2 == x._3).count().toDouble / rdd.count()
        accuracy
    }

    private def execute(training: RDD[Advertisement], categoryIndex: Int, categoriesSize: Int, tfidf: Boolean) = {
        val trainDataSet = createLabeledPointDataSet(training, categoryIndex, tfidf)

        val model = logisticRegression(trainDataSet, categoriesSize, (10, 0.01))

        model
    }

    private def logisticRegression(trainDataSet: RDD[LabeledPoint], categoriesSize: Int, parameter: (Int, Double)) = {
        var model: LogisticRegressionModel = null
        val resultTime = time {
            val modelSettings = new LogisticRegressionWithLBFGS()
                    .setNumClasses(categoriesSize)
            modelSettings.optimizer.setNumIterations(parameter._1)
            modelSettings.optimizer.setRegParam(parameter._2)

//            model = modelSettings.run(trainDataSet)
//            model.save(sc, s"model/logistic_regression_${categoriesSize}_${parameter._1}_${parameter._2}")
            model = LogisticRegressionModel.load(sc, s"model/logistic_regression_${categoriesSize}_${parameter._1}_${parameter._2}")
        }

        model
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

    def createLabeledPointDataSet(dataSet: RDD[Advertisement], categoryIndex: Int, tfidf: Boolean): RDD[LabeledPoint] = {
        val categories = dataSet
                .map(advertisement => advertisement.categories(categoryIndex))

        val tokens = dataSet
                .map(advertisement => processTitle(advertisement.title))

        //        val hashingTF = new HashingTF(524288)
        //        val hashingTF = new HashingTF(1024 * 100) // 2 kateogira Bayess
        //        val hashingTF = new HashingTF(1024 * 50) // 3 kateogira Bayess

        var hashingTF: HashingTF = null
        categoryIndex match {
            case 0 => hashingTF = new HashingTF()
            case 1 => hashingTF = new HashingTF(1024 * 100)
            case 2 => hashingTF = new HashingTF(1024 * 60)
        }
        // 1 kategoria Regression
        //        val hashingTF = new HashingTF(1024 * 100)// 2 kategoria Regression
        //        val hashingTF = new HashingTF(1024 * 60) // 3 kategoria Regression
        //2 kategoria
        //        val hashingTF = new HashingTF(524288/2)
        var tf = hashingTF.transform(tokens)

        if (tfidf) {
            val idf = new IDF().fit(tf)
            tf = idf.transform(tf)
        }

        val zipped = categories.zip(tf)
        val labelPointDataSet = zipped.map { case (category, vector) => LabeledPoint(category, vector) }
        //        labelPointDataSet.cache

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

    object ClassificationUtility {
        def predictPoint(dataMatrix: Vector, model: LogisticRegressionModel) = {
            require(dataMatrix.size == model.numFeatures)
            val dataWithBiasSize: Int = model.weights.size / (model.numClasses - 1)
            val weightsArray: Array[Double] = model.weights match {
                case dv: DenseVector => dv.values
                case _ =>
                    throw new IllegalArgumentException(
                        s"weights only supports dense vector but got type ${model.weights.getClass}.")
            }
            var bestClass = 0
            var maxMargin = 0.0
            val withBias = dataMatrix.size + 1 == dataWithBiasSize
            val classProbabilities: Array[Double] = new Array[Double](model.numClasses)
            (0 until model.numClasses - 1).foreach { i =>
                var margin = 0.0
                dataMatrix.foreachActive { (index, value) =>
                    if (value != 0.0) margin += value * weightsArray((i * dataWithBiasSize) + index)
                }
                // Intercept is required to be added into margin.
                if (withBias) {
                    margin += weightsArray((i * dataWithBiasSize) + dataMatrix.size)
                }
                if (margin > maxMargin) {
                    maxMargin = margin
                    bestClass = i + 1
                }
                classProbabilities(i + 1) = 1.0 / (1.0 + Math.exp(-(maxMargin - margin)))
            }

            (bestClass, classProbabilities(bestClass))
        }
    }
}