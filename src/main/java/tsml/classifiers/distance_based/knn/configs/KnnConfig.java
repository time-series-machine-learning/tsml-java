package tsml.classifiers.distance_based.knn.configs;

import com.google.common.collect.ImmutableList;
import evaluation.storage.ClassifierResults;
import experiments.ClassifierBuilderFactory;
import experiments.ClassifierBuilderFactory.ClassifierBuilder;
import experiments.ClassifierBuilderFactory.Tag;
import experiments.data.DatasetLoading;
import machine_learning.classifiers.tuned.incremental.IncTuner;
import tsml.classifiers.distance_based.distances.DDTWDistance;
import tsml.classifiers.distance_based.distances.DistanceMeasureConfigs;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.knn.KnnLoocv;
import tsml.classifiers.distance_based.knn.neighbour_iteration.LinearNeighbourIteratorBuilder;
import tsml.classifiers.distance_based.knn.neighbour_iteration.RandomNeighbourIteratorBuilder;
import utilities.ClassifierTools;
import utilities.iteration.LinearListIterator;
import utilities.iteration.RandomListIterator;
import utilities.params.ParamSpace;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;


public enum KnnConfig implements ClassifierBuilder {

    ED_1NN_V1(KnnConfig::buildEd1nnV1, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    DTW_1NN_V1(KnnConfig::buildDtw1nnV1, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    DDTW_1NN_V1(KnnConfig::buildDdtw1nnV1, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_DTW_1NN_V1(KnnConfig::buildTunedDtw1nnV1, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_DDTW_1NN_V1(KnnConfig::buildTunedDdtw1nnV1, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_WDTW_1NN_V1(KnnConfig::buildTunedWdtw1nnV1, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_WDDTW_1NN_V1(KnnConfig::buildTunedWddtw1nnV1, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_ERP_1NN_V1(KnnConfig::buildTunedErp1nnV1, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_MSM_1NN_V1(KnnConfig::buildTunedMsm1nnV1, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_LCSS_1NN_V1(KnnConfig::buildTunedLcss1nnV1, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_TWED_1NN_V1(KnnConfig::buildTunedTwed1nnV1, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),

    ED_1NN_V2(KnnConfig::buildEd1nnV2, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    DTW_1NN_V2(KnnConfig::buildDtw1nnV2, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    DDTW_1NN_V2(KnnConfig::buildDdtw1nnV2, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_DTW_1NN_V2(KnnConfig::buildTunedDtw1nnV2, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_DDTW_1NN_V2(KnnConfig::buildTunedDdtw1nnV2, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_WDTW_1NN_V2(KnnConfig::buildTunedWdtw1nnV2, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_WDDTW_1NN_V2(KnnConfig::buildTunedWddtw1nnV2, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_ERP_1NN_V2(KnnConfig::buildTunedErp1nnV2, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_MSM_1NN_V2(KnnConfig::buildTunedMsm1nnV2, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_LCSS_1NN_V2(KnnConfig::buildTunedLcss1nnV2, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),
    TUNED_TWED_1NN_V2(KnnConfig::buildTunedTwed1nnV2, KnnTag.DISTANCE, KnnTag.UNIVARIATE, KnnTag.SIMILARITY),

    ;

    public String toString() {
        return classifierBuilder.toString();
    }

    @Override
    public String getName() {
        return classifierBuilder.getName();
    }

    @Override
    public Classifier build() {
        return classifierBuilder.build();
    }

    @Override
    public ImmutableList<? extends Tag> getTags() {
        return classifierBuilder.getTags();
    }

    private final ClassifierBuilder classifierBuilder;

    KnnConfig(Supplier<? extends Classifier> supplier, Tag... tags) {
        classifierBuilder = new ClassifierBuilderFactory.SuppliedClassifierBuilder(name(), supplier, tags);
    }

    public static List<ClassifierBuilder> all() {
        return Arrays.stream(values()).map(i -> (ClassifierBuilder) i).collect(Collectors.toList());
    }
    
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
        classifier.setCvSearcherIteratorBuilder(new RandomNeighbourIteratorBuilder(classifier));
        classifier.setRandomTieBreak(true);
        return classifier;
    }

    public static KnnLoocv buildEd1nnV1() {
        KnnLoocv knn = build1nnV1();
        knn.setDistanceFunction(new DTWDistance(0));
        return knn;
    }

    public static KnnLoocv buildDtw1nnV1() {
        KnnLoocv knn = build1nnV1();
        knn.setDistanceFunction(new DTWDistance(-1));
        return knn;
    }

    public static KnnLoocv buildDdtw1nnV1() {
        KnnLoocv knn = build1nnV1();
        knn.setDistanceFunction(new DDTWDistance(-1));
        return knn;
    }

    public static KnnLoocv buildEd1nnV2() {
        KnnLoocv knn = build1nnV2();
        knn.setDistanceFunction(new DTWDistance(0));
        return knn;
    }

    public static KnnLoocv buildDtw1nnV2() {
        KnnLoocv knn = build1nnV2();
        knn.setDistanceFunction(new DTWDistance(-1));
        return knn;
    }

    public static KnnLoocv buildDdtw1nnV2() {
        KnnLoocv knn = build1nnV2();
        knn.setDistanceFunction(new DDTWDistance(-1));
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
        return buildTuned1nnV2(DistanceMeasureConfigs::buildDtwSpaceV2);
    }

    public static IncTuner buildTunedDdtw1nnV2() {
        return buildTuned1nnV2(DistanceMeasureConfigs::buildDdtwSpaceV2);
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
        return buildTuned1nnV2(i -> DistanceMeasureConfigs.buildMsmSpace());
    }

    public static IncTuner buildTunedTwed1nnV2() {
        return buildTuned1nnV2(i -> DistanceMeasureConfigs.buildTwedSpace());
    }

    public static IncTuner buildTunedErp1nnV2() {
        return buildTuned1nnV2(DistanceMeasureConfigs::buildErpSpace);
    }

    public static IncTuner buildTunedLcss1nnV2() {
        return buildTuned1nnV2(DistanceMeasureConfigs::buildLcssSpace);
    }


    public static IncTuner buildTuned1nnV1(Function<Instances, ParamSpace> paramSpaceFunction) {
        IncTuner incTunedClassifier = new IncTuner();
        IncKnnTunerSetup incKnnTunerSetup = new IncKnnTunerSetup();
        incKnnTunerSetup
                .setIncTunedClassifier(incTunedClassifier)
                .setParamSpace(paramSpaceFunction)
                .setKnnSupplier(KnnConfig::build1nnV1).setImproveableBenchmarkIteratorBuilder(LinearListIterator::new);
        incTunedClassifier.setTrainSetupFunction(incKnnTunerSetup);
        return incTunedClassifier;
    }

    public static IncTuner buildTuned1nnV1(ParamSpace paramSpace) {
        return buildTuned1nnV1(i -> paramSpace);
    }

    public static IncTuner buildTuned1nnV2(Function<Instances, ParamSpace> paramSpaceFunction) {
        IncTuner incTunedClassifier = new IncTuner();
        IncKnnTunerSetup incKnnTunerSetup = new IncKnnTunerSetup();
        incKnnTunerSetup
                .setIncTunedClassifier(incTunedClassifier)
                .setParamSpace(paramSpaceFunction)
                .setKnnSupplier(KnnConfig::build1nnV1).setImproveableBenchmarkIteratorBuilder(benchmarks -> new RandomListIterator<>(benchmarks, incTunedClassifier.getSeed()));
        incTunedClassifier.setTrainSetupFunction(incKnnTunerSetup);
        return incTunedClassifier;
    }

    public static IncTuner buildTuned1nnV2(ParamSpace paramSpace) {
        return buildTuned1nnV2(i -> paramSpace);
    }
}
