package tsml.classifiers.distance_based.ee;

import com.google.common.collect.ImmutableList;
import experiments.ClassifierBuilderFactory;
import machine_learning.classifiers.tuned.incremental.IncTuner;
import tsml.classifiers.distance_based.knn.configs.IncKnnTunerBuilder;
import tsml.classifiers.distance_based.knn.configs.KnnTag;
import weka.classifiers.Classifier;

import java.util.function.Consumer;
import java.util.function.Supplier;

public enum EeConfig implements Supplier<ClassifierBuilderFactory.ClassifierBuilder<?>> {
    EE_V1(EeConfig::buildEeV1, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
    EE_V2(EeConfig::buildEeV2, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
    CEE_V1(EeConfig::buildCeeV1, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
    CEE_V2(EeConfig::buildCeeV2, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
    LEE(EeConfig::buildLee, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
    ;

    private final ClassifierBuilderFactory.ClassifierBuilder<?> classifierBuilder;

    EeConfig(final Supplier<? extends Classifier> supplier, final Supplier<String>... tags) {
        classifierBuilder = new ClassifierBuilderFactory.ClassifierBuilder<>(name(), supplier, tags);
    }

    public ClassifierBuilderFactory.ClassifierBuilder<?> getClassifierBuilder() {
        return classifierBuilder;
    }

    @Override public ClassifierBuilderFactory.ClassifierBuilder<?> get() {
        return getClassifierBuilder();
    }

    // -----------------------------------------------------------------------------------------------------------------

    public static EE buildEeV1() {
        EE ee = new EE();
        ee.setConstituents(EE.getV1Constituents());
        return ee; // todo set full ee?
    }

    public static EE buildEeV2() {
        EE ee = new EE();
        ee.setConstituents(EE.getV2Constituents());
        return ee; // todo set full ee?
    }

    public static EE buildCeeV1() {
        return buildEeV1(); // todo turn off full ee?
    }

    public static EE buildCeeV2() {
        return buildEeV2(); // todo turn off full ee?
    }

    private static EE forEachTunedConstituent(EE ee, Consumer<IncKnnTunerBuilder> consumer) {
        for(Classifier classifier : ee.getConstituents()) {
            if(!(classifier instanceof IncTuner)) {
                continue;
            }
            IncTuner tuner = (IncTuner) classifier;
            IncKnnTunerBuilder config = (IncKnnTunerBuilder) tuner.getInitFunction();
            consumer.accept(config);
        }
        return ee;
    }

    public static EE setLimitedParameters(EE ee, int limit) {
        return forEachTunedConstituent(ee, incKnnTunerBuilder -> incKnnTunerBuilder.setMaxParamSpaceSize(limit));
    }

    public static EE setLimitedParametersPercentage(EE ee, double limit) {
        return forEachTunedConstituent(ee, incKnnTunerBuilder -> incKnnTunerBuilder.setMaxParamSpaceSizePercentage(limit));
    }

    public static EE setLimitedNeighbours(EE ee, int limit) {
        return forEachTunedConstituent(ee, incKnnTunerBuilder -> incKnnTunerBuilder.setMaxNeighbourhoodSize(limit));
    }

    public static EE setLimitedNeighboursPercentage(EE ee, double limit) { // todo params from cmdline in experiment + append to cls name
        return forEachTunedConstituent(ee, incKnnTunerBuilder -> incKnnTunerBuilder.setMaxParamSpaceSizePercentage(limit));
    }

    private static EE buildLee() {
        EE ee = new EE();
        ImmutableList<Classifier> constituents = EE.getV2Constituents();
        ee.setConstituents(constituents);
        setLimitedNeighboursPercentage(ee, 0.1);
        setLimitedParametersPercentage(ee, 0.5);
        return ee;
    }
}
