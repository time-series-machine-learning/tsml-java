package tsml.classifiers.distance_based.ee;

import com.google.common.collect.ImmutableList;
import experiments.ClassifierBuilderFactory;
import tsml.classifiers.distance_based.distances.DistanceMeasureConfigs;
import tsml.classifiers.distance_based.knn.configs.IncKnnTunerBuilder;
import tsml.classifiers.distance_based.knn.configs.KnnConfig;
import tsml.classifiers.distance_based.knn.configs.KnnTag;
import weka.classifiers.Classifier;

import java.util.function.Supplier;

import static tsml.classifiers.distance_based.knn.configs.KnnConfig.*;

public enum EEConfig implements Supplier<ClassifierBuilderFactory.ClassifierBuilder<?>> {
    EE_V1(EEConfig::buildEeV1, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
    EE_V2(EEConfig::buildEeV2, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
    CEE_V1(EEConfig::buildCeeV1, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
    CEE_V2(EEConfig::buildCeeV2, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
    LEE(EEConfig::buildLee, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
    ;

    private final ClassifierBuilderFactory.ClassifierBuilder<?> classifierBuilder;

    EEConfig(final Supplier<? extends Classifier> supplier, final Supplier<String>... tags) {
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

    public static EE buildLee() {
        EE ee = new EE();
        ee.setConstituents(EE.getV2Constituents());
        ee.setLimitedVersion(true); // todo actually set limits for neighbours + params
        return ee;
    }

//    public static EE buildV1() {
//        EE EE = new EE();
//        EE.setConstituents(ImmutableList.of(
//                ED_1NN_V1.getClassifierBuilder().build(),
//                DTW_1NN_V1.getClassifierBuilder().build(),
//                DDTW_1NN_V1.getClassifierBuilder().build(),
//                TUNED_DTW_1NN_V1.getClassifierBuilder().build(),
//                TUNED_DDTW_1NN_V1.getClassifierBuilder().build(),
//                TUNED_WDTW_1NN_V1.getClassifierBuilder().build(),
//                TUNED_WDDTW_1NN_V1.getClassifierBuilder().build(),
//                TUNED_ERP_1NN_V1.getClassifierBuilder().build(),
//                TUNED_MSM_1NN_V1.getClassifierBuilder().build(),
//                TUNED_LCSS_1NN_V1.getClassifierBuilder().build(),
//                TUNED_TWED_1NN_V1.getClassifierBuilder().build()
//        ));
//        return EE;
//    }
//
//    public static EE buildV2() {
//        EE EE = new EE();
//        EE.setConstituents(ImmutableList.of(
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildDtwSpaceV2).build(),
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildDdtwSpaceV2).build(),
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildErpSpace).build(),
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildLcssSpace).build(),
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildMsmSpace).build(),
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildTwedSpace).build(),
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildWdtwSpaceV2).build(),
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildWddtwSpaceV2).build()
//        ));
//        return EE;
//    }
//
//    public static EE buildLee() { // AKA limited EE
//        EE EE = new EE();
//        EE.setConstituents(ImmutableList.of(
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildDtwSpaceV1).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build(),
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildDdtwSpaceV1).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build(),
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildErpSpace).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build(),
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildLcssSpace).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build(),
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildMsmSpace).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build(),
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildTwedSpace).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build(),
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildWdtwSpaceV1).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build(),
//                new IncKnnTunerBuilder().setKnnSupplier(KnnConfig::build1nnV2).setParamSpace(
//                        DistanceMeasureConfigs::buildWddtwSpaceV1).setMaxNeighbourhoodSizePercentage(0.1).setMaxParamSpaceSizePercentage(0.5).build()
//        ));
//        return EE;
//    }
}
