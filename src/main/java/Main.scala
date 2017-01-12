import java.nio.file.{Files, Paths}
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

case class Advertisement(id: Long, title: String, categories: Seq[Long])

object Main extends java.io.Serializable {
  val output: Output = Resource.fromFile("results/result_" + new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss-SSS").format(new Date()))
  val testsr: Output = Resource.fromFile("results/result_" + new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss-SSS").format(new Date()) + ".tsv")

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

    val data = sc.textFile("data/training.[0-9]*.tsv")
    //val data = sc.textFile("data/training.[0-2]*.tsv")
    //        val data = sc.textFile("data/test.tsv")

    val categoriesFile = sc.textFile("data/categories.tsv")
    val categiresRDD = categoriesFile
      .zipWithIndex().filter((tuple: (String, Long)) => tuple._2 > 1)
      .map(string => string._1.split("\t"))
      .map(
        stringArray => (stringArray(0).toInt, stringArray(1).toInt)
      )

    /*val categories = categoriesHierarchyMap.keySet.toArray.sorted*/

    /*val allCategoriesMap = categories
      .zipWithIndex
      .toMap*/

    val L1Categories = categiresRDD
      .filter(row => row._2 == 0)
      .map(row => row._1)

    val L1Cat = L1Categories.zipWithIndex().collectAsMap()

    val L2Categories = categiresRDD
      .groupBy(row => row._2)
      .join(L1Categories.keyBy(row => row))
      .map(row => (L1Cat(row._1), row._2._1.map(row => row._1).zipWithIndex.toMap))
      .collectAsMap()


    val convertedData = data
      .map(string => string.split("\t"))
      .filter(row => row.size >= 6)
      .map(stringArray =>
        Advertisement(
          stringArray(0).toLong,
          stringArray(1),
          Seq(
            L1Cat(stringArray(4).toInt),
            L2Categories(L1Cat(stringArray(4).toInt))(stringArray(5).toInt),
            0
          )
        )
      )

    val tests = sc.textFile("data/test.[0-9]*.tsv")
      .map(string => string.split("\t"))
      .map(stringArray =>
        Advertisement(
          stringArray(0).toLong,
          stringArray(1),
          Seq(-1, -1, -1)
        )
      )

    createLabeledPointDataSet(convertedData.union(tests), 0, true)

    val models = executeForAllCategories(convertedData, L1Cat, L2Categories, tests)

    val test = predict(tests, convertedData, models._1, models._2, models._3, true)

    test
      .collect()
      .foreach(row => {
        var category = 0

        if (row._3 == 1)
          category = L1Cat.filter(r => r._2 == row._2._2).head._1
        else if (row._3 == 2)
          category = L2Categories(row._2._2).filter(r => r._2 == row._2._3).head._1

        testsr.write(s"${row._1.id}\t$category\n")
      })

    val test2 = predict(convertedData, tests, models._1, models._2, models._3, false)
    output.write(calculateAccuracy(test2) + "\n")
  }

  private def createCategoryMap(data: RDD[String], category: Int, cat: Int) = {
    data
      .map(string => string.split("\t"))
      .filter(stringArray => !stringArray(4 + category).isEmpty)
      .filter(row => cat == -1 || row(3 + category).toInt == cat)
      .map(stringArray => stringArray(4 + category).toInt)
      .distinct()
      .zipWithIndex()
      .collectAsMap()
  }

  private def executeForAllCategories(dataSet: RDD[Advertisement], L1: Map[Int, Long], L2: Map[Long, Map[Int, Int]], tests: RDD[Advertisement]) = {
    val modelL1Map = scala.collection.mutable.Map[Long, LogisticRegressionModel]()
    val modelL2Map = scala.collection.mutable.Map[Long, LogisticRegressionModel]()

    val model = execute(dataSet, 0, L1.size, tfidf = true, 0, tests)

    for (category <- L1) {

      val L2set = dataSet
        .filter(row => row.categories(0) == category._2)

      modelL1Map.put(
        category._2,
        execute(L2set, 1, L2(category._2).size, tfidf = true, category._1, tests)
      )
    }

    (model, modelL1Map, modelL2Map)
  }

  private def predict(dataSet: RDD[Advertisement], tests: RDD[Advertisement], model: LogisticRegressionModel, L1Models: Map[Long, LogisticRegressionModel], L2Models: Map[Long, LogisticRegressionModel], test: Boolean) = {
    val labeled = dataSet.zip(createLabeledPointDataSet(dataSet, 0, tfidf = true))
    val labeledL1 = dataSet.zip(createLabeledPointDataSet(dataSet, 1, tfidf = true))

    var conc = labeled
      .join(labeledL1)

    val L1predictions = conc
      .map(row => (row, ClassificationUtility.predictPoint(row._2._1.features, model)))

    val L2predictions = L1predictions
      .map(row => {
        if (row._2._2 < 0.9)
          (row._1, row._2, (-1.toDouble, -1.toDouble))
        else
          (row._1, row._2, ClassificationUtility.predictPoint(row._1._2._2.features, L1Models(row._2._1.toLong)))
      })

    L2predictions
      .map(row => {
        if (row._3._1 == -1)
          (row._1._1, (0.toLong, -1.toLong, -1.toLong, -1.toLong), 0)
        else if (row._3._2 < 0.9)
          (row._1._1, (0.toLong, row._2._1.toLong, -1.toLong, -1.toLong), 1)
        else
          (row._1._1, (0.toLong, row._2._1.toLong, row._3._1.toLong, -1.toLong), 2)
      })
  }

  private def getCategoryIndexInErrorMatrix(allCategoriesMap: Map[Int, Int], modelCategoriesMap: Map[Int, Map[Int, Long]], categoryIndex: Int, category: Double) = {
    allCategoriesMap(modelCategoriesMap(categoryIndex).map(_.swap).get(category.toInt).get)
  }

  def calculateAccuracy(rdd: RDD[(Advertisement, (Long, Long, Long, Long), Int)]): Double = {
    val accuracy = rdd.filter(x => {
      x._3 match {
        case 0 => false
        case 1 => x._1.categories(0) == x._2._2
        case 2 => x._1.categories(1) == x._2._3
        case 3 => x._1.categories(2) == x._2._4
      }
    }).count().toDouble / rdd.filter(r => r._3 != 0).count()
    accuracy
  }

  private def execute(dataSet: RDD[Advertisement], categoryIndex: Int, categoriesSize: Int, tfidf: Boolean, category: Int, tests: RDD[Advertisement]) = {

    var model: LogisticRegressionModel = null
    val parameter = (10, 0.01)
    val path = s"model/logistic_regression_${categoriesSize}_${parameter._1}_${parameter._2}_$category"

    if (!Files.exists(Paths.get(path))) {
      val labeledDataSet = createLabeledPointDataSet(dataSet, categoryIndex, tfidf)
      val labeledTrainDataSet = labeledDataSet/*dataSet.keyBy(r => r.id)
        .join(labeledDataSet.keyBy(r => r._4))
        .map(r => {
          categoryIndex match {
            case 0 => r._2._2._1
            case 1 => r._2._2._2
            case 2 => r._2._2._3
          }
        })*/

      output.write(path + " " + time {
        val modelSettings = new LogisticRegressionWithLBFGS()
          .setNumClasses(categoriesSize)
        modelSettings.optimizer.setNumIterations(parameter._1)
        modelSettings.optimizer.setRegParam(parameter._2)

        model = modelSettings.run(labeledTrainDataSet)
        model.save(sc, path)
      } + "\n")
    }

    model = LogisticRegressionModel.load(sc, path)
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
      case 0 => hashingTF = new HashingTF(1024 * 30)
      //            case 1 => hashingTF = new HashingTF(1024 * 100)
      case 1 => hashingTF = new HashingTF(1024 * 30)
      case 2 => hashingTF = new HashingTF(1024 * 30)
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



  /*var points: RDD[(LabeledPoint, LabeledPoint, LabeledPoint, Long)] = null

  def createLabeledPointDataSet(dataSet: RDD[Advertisement], categoryIndex: Int, tfidf: Boolean): RDD[(LabeledPoint, LabeledPoint, LabeledPoint, Long)] = {
    if (points != null)
      return points

    val tokens = dataSet
      .map(advertisement => processTitle(advertisement.title))

    //        val hashingTF = new HashingTF(524288)
    //        val hashingTF = new HashingTF(1024 * 100) // 2 kateogira Bayess
    //        val hashingTF = new HashingTF(1024 * 50) // 3 kateogira Bayess

    var hashingTF: HashingTF = null
    categoryIndex match {
      case 0 => hashingTF = new HashingTF(1024 * 30)
      //            case 1 => hashingTF = new HashingTF(1024 * 100)
      case 1 => hashingTF = new HashingTF(1024 * 30)
      case 2 => hashingTF = new HashingTF(1024 * 30)
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

    val zipped = dataSet.zip(tf)
    val labelPointDataSet: RDD[(LabeledPoint, LabeledPoint, LabeledPoint, Long)] = zipped.map { case (all, vector) => (LabeledPoint(all.categories(0), vector),
      LabeledPoint(all.categories(1), vector), LabeledPoint(all.categories(2), vector), all.id)
    }
    //        labelPointDataSet.cache

    points = labelPointDataSet

    points
  }*/

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
      }
      (bestClass.toDouble, 1.0 / (1.0 + Math.exp(-maxMargin)))
    }
  }

}