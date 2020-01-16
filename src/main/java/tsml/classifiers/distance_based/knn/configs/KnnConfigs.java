package tsml.classifiers.distance_based.knn.configs;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import machine_learning.classifiers.tuned.incremental.IncTuner;
import tsml.classifiers.distance_based.distances.Ddtw;
import tsml.classifiers.distance_based.distances.DistanceMeasureConfigs;
import tsml.classifiers.distance_based.distances.Dtw;
import tsml.classifiers.distance_based.knn.KnnLoocv;
import tsml.classifiers.distance_based.knn.neighbour_iteration.LinearNeighbourIteratorBuilder;
import tsml.classifiers.distance_based.knn.neighbour_iteration.RandomNeighbourIteratorBuilder;
import utilities.ClassifierTools;
import utilities.params.ParamSpace;
import weka.core.Instances;

import java.util.function.Function;
import java.util.function.Supplier;

import static utilities.ArrayUtilities.incrementalRange;

public class KnnConfigs {
    private KnnConfigs() {}

    public static KnnLoocv build1nnV1() {
        KnnLoocv classifier = new KnnLoocv();
        classifier.setEarlyAbandon(true);
        classifier.setK(1);
        classifier.setNeighbourLimit(-1);
        classifier.setNeighbourIteratorBuilder(new LinearNeighbourIteratorBuilder(classifier));
        classifier.setRandomTieBreak(false);
        return classifier;
    }

    public static KnnLoocv build1nnV2() {
        KnnLoocv classifier = new KnnLoocv();
        classifier.setEarlyAbandon(true);
        classifier.setK(1);
        classifier.setNeighbourLimit(-1);
        classifier.setNeighbourIteratorBuilder(new RandomNeighbourIteratorBuilder(classifier));
        classifier.setRandomTieBreak(true);
        return classifier;
    }

    public static KnnLoocv buildEd1nnV1() {
        KnnLoocv knn = build1nnV1();
        knn.setDistanceFunction(new Dtw(0));
        return knn;
    }

    public static KnnLoocv buildDtw1nnV1() {
        KnnLoocv knn = build1nnV1();
        knn.setDistanceFunction(new Dtw(-1));
        return knn;
    }

    public static KnnLoocv buildDdtw1nnV1() {
        KnnLoocv knn = build1nnV1();
        knn.setDistanceFunction(new Ddtw(-1));
        return knn;
    }

    public static KnnLoocv buildEd1nnV2() {
        KnnLoocv knn = build1nnV2();
        knn.setDistanceFunction(new Dtw(0));
        return knn;
    }

    public static KnnLoocv buildDtw1nnV2() {
        KnnLoocv knn = build1nnV2();
        knn.setDistanceFunction(new Dtw(-1));
        return knn;
    }

    public static KnnLoocv buildDdtw1nnV2() {
        KnnLoocv knn = build1nnV2();
        knn.setDistanceFunction(new Ddtw(-1));
        return knn;
    }


    public static void main(String[] args) throws Exception {
        int seed = 0;
        Instances[] data = DatasetLoading.sampleGunPoint(seed);
        IncTuner classifier = buildTunedDtw1nnV1();
        classifier.setSeed(seed); // set seed
        classifier.setEstimateOwnPerformance(true);
        ClassifierResults results = ClassifierTools.trainAndTest(data, classifier);
        results.setDetails(classifier, data[1]);
        ClassifierResults trainResults = classifier.getTrainResults();
        trainResults.setDetails(classifier, data[0]);
        System.out.println(trainResults.writeSummaryResultsToString());
        System.out.println(results.writeSummaryResultsToString());
    }

    public static IncTuner buildTunedDtw1nnV1() {
        return buildTuned1nnV1(DistanceMeasureConfigs::buildDtwSpaceV1);
    }

    public static IncTuner buildTunedDdtw1nnV1() {
        return buildTuned1nnV1(DistanceMeasureConfigs::buildDdtwSpaceV1);
    }

    public static IncTuner buildTunedWdtw1nnV1() {
        return buildTuned1nnV1(i -> DistanceMeasureConfigs.buildWdtwSpaceV1());
    }

    public static IncTuner buildTunedWddtw1nnV1() {
        return buildTuned1nnV1(i -> DistanceMeasureConfigs.buildWddtwSpaceV1());
    }

    public static IncTuner buildTunedDtw1nnV2() {
        return buildTuned1nnV1(DistanceMeasureConfigs::buildDtwSpaceV2);
    }

    public static IncTuner buildTunedDdtw1nnV2() {
        return buildTuned1nnV1(DistanceMeasureConfigs::buildDdtwSpaceV2);
    }

    public static IncTuner buildTunedWdtw1nnV2() {
        return buildTuned1nnV1(i -> DistanceMeasureConfigs.buildWdtwSpaceV2());
    }

    public static IncTuner buildTunedWddtw1nnV2() {
        return buildTuned1nnV1(i -> DistanceMeasureConfigs.buildWddtwSpaceV2());
    }

    public static IncTuner buildTunedMsm1nnV1() {
        return buildTuned1nnV1(i -> DistanceMeasureConfigs.buildMsmSpace());
    }

    public static IncTuner buildTunedTwed1nnV1() {
        return buildTuned1nnV1(i -> DistanceMeasureConfigs.buildTwedSpace());
    }

    public static IncTuner buildTunedErp1nnV1() {
        return buildTuned1nnV1(DistanceMeasureConfigs::buildErpSpace);
    }

    public static IncTuner buildTunedLcss1nnV1() {
        return buildTuned1nnV1(DistanceMeasureConfigs::buildLcssSpace);
    }

    public static IncTuner buildTunedMsm1nnV2() {
        return buildTuned1nnV1(i -> DistanceMeasureConfigs.buildMsmSpace());
    }

    public static IncTuner buildTunedTwed1nnV2() {
        return buildTuned1nnV1(i -> DistanceMeasureConfigs.buildTwedSpace());
    }

    public static IncTuner buildTunedErp1nnV2() {
        return buildTuned1nnV1(DistanceMeasureConfigs::buildErpSpace);
    }

    public static IncTuner buildTunedLcss1nnV2() {
        return buildTuned1nnV1(DistanceMeasureConfigs::buildLcssSpace);
    }

    public static IncTuner buildTunedKnn(Function<Instances, ParamSpace> paramSpaceFunction,
                                         Supplier<KnnLoocv> supplier) {
        IncTuner incTunedClassifier = new IncTuner();
        incTunedClassifier.setInitFunction(new IncKnnTunerBuilder().setIncTunedClassifier(incTunedClassifier).setParamSpace(paramSpaceFunction).setKnnSupplier(supplier));
        return incTunedClassifier;
    }

    public static IncTuner buildTuned1nnV1(Function<Instances, ParamSpace> paramSpaceFunction) {
        return buildTunedKnn(paramSpaceFunction, KnnConfigs::build1nnV1);
    }

    public static IncTuner buildTuned1nnV1(ParamSpace paramSpace) {
        return buildTunedKnn(i -> paramSpace, KnnConfigs::build1nnV1);
    }

    public static IncTuner buildTuned1nnV2(Function<Instances, ParamSpace> paramSpaceFunction) {
        return buildTunedKnn(paramSpaceFunction, KnnConfigs::build1nnV2);
    }

    public static IncTuner buildTuned1nnV2(ParamSpace paramSpace) {
        return buildTunedKnn(i -> paramSpace, KnnConfigs::build1nnV2);
    }
}
