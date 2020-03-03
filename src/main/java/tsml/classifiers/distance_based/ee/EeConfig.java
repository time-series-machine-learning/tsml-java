package tsml.classifiers.distance_based.ee;

import com.google.common.collect.ImmutableList;
import tsml.classifiers.distance_based.utils.classifier_building.ClassifierBuilderFactory;
import tsml.classifiers.distance_based.knn.configs.IncKnnTunerSetup;
import tsml.classifiers.distance_based.knn.configs.KnnTag;
import tsml.classifiers.distance_based.rltuning.RLTunedClassifier;
import weka.classifiers.Classifier;

import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import static tsml.classifiers.distance_based.knn.configs.KnnConfig.*;
import static tsml.classifiers.distance_based.knn.configs.KnnConfig.TUNED_TWED_1NN_V2;

public enum EeConfig implements ClassifierBuilderFactory.ClassifierBuilder {
    EE_V1(EeConfig::buildEeV1, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
    EE_V2(EeConfig::buildEeV2, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
    CEE_V1(EeConfig::buildCeeV1, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
    CEE_V2(EeConfig::buildCeeV2, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
    LEE(EeConfig::buildLee, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
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
    public ImmutableList<? extends ClassifierBuilderFactory.Tag> getTags() {
        return classifierBuilder.getTags();
    }

    private final ClassifierBuilderFactory.ClassifierBuilder classifierBuilder;

    EeConfig(Supplier<? extends Classifier> supplier, ClassifierBuilderFactory.Tag... tags) {
        classifierBuilder = new ClassifierBuilderFactory.SuppliedClassifierBuilder(name(), supplier, tags);
    }

    public static List<ClassifierBuilderFactory.ClassifierBuilder> all() {
        return Arrays.stream(values()).map(i -> (ClassifierBuilderFactory.ClassifierBuilder) i).collect(Collectors.toList());
    }

    // -----------------------------------------------------------------------------------------------------------------

    public static ImmutableList<Classifier> buildV1Constituents() {
        return ImmutableList.of(
                ED_1NN_V1.build(),
                DTW_1NN_V1.build(),
                DDTW_1NN_V1.build(),
                TUNED_DTW_1NN_V1.build(),
                TUNED_DDTW_1NN_V1.build(),
                TUNED_WDTW_1NN_V1.build(),
                TUNED_WDDTW_1NN_V1.build(),
                TUNED_ERP_1NN_V1.build(),
                TUNED_MSM_1NN_V1.build(),
                TUNED_LCSS_1NN_V1.build(),
                TUNED_TWED_1NN_V1.build()
        );
    }

    public static ImmutableList<Classifier> buildV2Constituents() {
        return ImmutableList.of(
                ED_1NN_V2.build(),
                DTW_1NN_V2.build(),
                DDTW_1NN_V2.build(),
                TUNED_DTW_1NN_V2.build(),
                TUNED_DDTW_1NN_V2.build(),
                TUNED_WDTW_1NN_V2.build(),
                TUNED_WDDTW_1NN_V2.build(),
                TUNED_ERP_1NN_V2.build(),
                TUNED_MSM_1NN_V2.build(),
                TUNED_LCSS_1NN_V2.build(),
                TUNED_TWED_1NN_V2.build()
        );
    }

    public static Ee buildEeV1() {
        Ee ee = new Ee();
        ee.setConstituents(buildV1Constituents());
        setTrainSelectedBenchmarksFully(ee,false);
        return ee; // todo set full ee?
    }

    public static Ee buildEeV2() {
        Ee ee = new Ee();
        ee.setConstituents(buildV2Constituents());
        setTrainSelectedBenchmarksFully(ee,false);
        return ee; // todo set full ee?
    }

    public static Ee buildCeeV1() {
        return buildEeV1(); // todo turn off full ee?
    }

    public static Ee buildCeeV2() {
        return buildEeV2(); // todo turn off full ee?
    }

    private static Ee forEachTunedConstituent(Ee ee, Consumer<IncKnnTunerSetup> consumer) {
        for(Classifier classifier : ee.getConstituents()) {
            if(!(classifier instanceof RLTunedClassifier)) {
                continue;
            }
            RLTunedClassifier tuner = (RLTunedClassifier) classifier;
            IncKnnTunerSetup config = (IncKnnTunerSetup) tuner.getTrainSetupFunction();
            consumer.accept(config);
        }
        return ee;
    }

    public static Ee setLimitedParameters(Ee ee, int limit) {
        return forEachTunedConstituent(ee, incKnnTunerSetup -> incKnnTunerSetup.setParamSpaceSizeLimit(limit));
    }

    public static Ee setLimitedParametersPercentage(Ee ee, double limit) {
        return forEachTunedConstituent(ee, incKnnTunerSetup -> incKnnTunerSetup.setParamSpaceSizeLimitPercentage(limit));
    }

    public static Ee setLimitedNeighbours(Ee ee, int limit) {
        return forEachTunedConstituent(ee, incKnnTunerSetup -> incKnnTunerSetup.setNeighbourhoodSizeLimit(limit));
    }

    public static Ee setLimitedNeighboursPercentage(Ee ee, double limit) { // todo params from cmdline in experiment + append to cls name
        return forEachTunedConstituent(ee, incKnnTunerSetup -> incKnnTunerSetup.setParamSpaceSizeLimitPercentage(limit));
    }

    public static Ee setTrainSelectedBenchmarksFully(Ee ee, boolean state) { // todo params from cmdline in experiment + append to cls name
        return forEachTunedConstituent(ee, incKnnTunerSetup -> incKnnTunerSetup.setTrainSelectedBenchmarksFully(state));
    }

    private static Ee buildLee() {
        Ee ee = new Ee();
        ImmutableList<Classifier> constituents = buildV2Constituents();
        ee.setConstituents(constituents);
        setLimitedNeighboursPercentage(ee, 0.1);
        setLimitedParametersPercentage(ee, 0.5);
        setTrainSelectedBenchmarksFully(ee,true);
        return ee;
    }
}
