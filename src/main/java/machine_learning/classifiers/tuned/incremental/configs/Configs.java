package machine_learning.classifiers.tuned.incremental.configs;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import machine_learning.classifiers.tuned.incremental.*;
import tsml.classifiers.distance_based.knn.KNNCV;
import utilities.Utilities;
import utilities.params.ParamSpace;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.function.Supplier;

import static tsml.classifiers.distance_based.knn.Configs.build1nnV1;

public class Configs {

    public static void main(String[] args) throws Exception {
        int seed = 0;
        Instances[] instances = DatasetLoading.sampleGunPoint(seed);
        Instances train = instances[0];
        Instances test = instances[1];
        IncTunedClassifier incTunedClassifier = buildTunedDtw1nnV1();
        incTunedClassifier.setTrainTimeLimit(10, TimeUnit.SECONDS);//TimeUnit.MINUTES);
        incTunedClassifier.buildClassifier(train);
        ClassifierResults trainResults = incTunedClassifier.getTrainResults();
        ClassifierResults results = new ClassifierResults();
        results.setDetails(incTunedClassifier, train);
        for(Instance testCase : test) {
            long startTime = System.nanoTime();
            double[] distribution = incTunedClassifier.distributionForInstance(testCase);
            long timeTaken = System.nanoTime() - startTime;
            int prediction = Utilities.argMax(distribution, new Random(seed));
            results.addPrediction(testCase.classValue(), distribution, prediction, timeTaken, "");
        }
        System.out.println(results.getAcc());
        System.out.println(trainResults.getBuildTime());
    }

    public static IncTunedClassifier buildTunedDtw1nnV1() {
        return buildTuned1nnV1(tsml.classifiers.distance_based.distances.Configs::buildDtwSpaceV1);
    }

    public static IncTunedClassifier buildTunedDdtw1nnV1() {
        return buildTuned1nnV1(tsml.classifiers.distance_based.distances.Configs::buildDdtwSpaceV1);
    }

    public static IncTunedClassifier buildTunedWdtw1nnV1() {
        return buildTuned1nnV1(i -> tsml.classifiers.distance_based.distances.Configs.buildWdtwSpaceV1());
    }

    public static IncTunedClassifier buildTunedWddtw1nnV1() {
        return buildTuned1nnV1(i -> tsml.classifiers.distance_based.distances.Configs.buildWddtwSpaceV1());
    }

    public static IncTunedClassifier buildTunedDtw1nnV2() {
        return buildTuned1nnV1(tsml.classifiers.distance_based.distances.Configs::buildDtwSpaceV2);
    }

    public static IncTunedClassifier buildTunedDdtw1nnV2() {
        return buildTuned1nnV1(tsml.classifiers.distance_based.distances.Configs::buildDdtwSpaceV2);
    }

    public static IncTunedClassifier buildTunedWdtw1nnV2() {
        return buildTuned1nnV1(i -> tsml.classifiers.distance_based.distances.Configs.buildWdtwSpaceV2());
    }

    public static IncTunedClassifier buildTunedWddtw1nnV2() {
        return buildTuned1nnV1(i -> tsml.classifiers.distance_based.distances.Configs.buildWddtwSpaceV2());
    }

    public static IncTunedClassifier buildTunedMsm1nnV1() {
        return buildTuned1nnV1(i -> tsml.classifiers.distance_based.distances.Configs.buildMsmSpace());
    }

    public static IncTunedClassifier buildTunedTwed1nnV1() {
        return buildTuned1nnV1(i -> tsml.classifiers.distance_based.distances.Configs.buildTwedSpace());
    }

    public static IncTunedClassifier buildTunedErp1nnV1() {
        return buildTuned1nnV1(tsml.classifiers.distance_based.distances.Configs::buildErpSpace);
    }

    public static IncTunedClassifier buildTunedLcss1nnV1() {
        return buildTuned1nnV1(tsml.classifiers.distance_based.distances.Configs::buildLcssSpace);
    }

    public static IncTunedClassifier buildTunedMsm1nnV2() {
        return buildTuned1nnV1(i -> tsml.classifiers.distance_based.distances.Configs.buildMsmSpace());
    }

    public static IncTunedClassifier buildTunedTwed1nnV2() {
        return buildTuned1nnV1(i -> tsml.classifiers.distance_based.distances.Configs.buildTwedSpace());
    }

    public static IncTunedClassifier buildTunedErp1nnV2() {
        return buildTuned1nnV1(tsml.classifiers.distance_based.distances.Configs::buildErpSpace);
    }

    public static IncTunedClassifier buildTunedLcss1nnV2() {
        return buildTuned1nnV1(tsml.classifiers.distance_based.distances.Configs::buildLcssSpace);
    }

    public static IncTunedClassifier buildTunedKnn(Function<Instances, ParamSpace> paramSpaceFunction,
                                                    Supplier<KNNCV> supplier) {
        IncTunedClassifier incTunedClassifier = new IncTunedClassifier();
        incTunedClassifier.setOnTrainDataAvailable(new Inc1nnTuningSetup(incTunedClassifier,
                                                                         paramSpaceFunction,
                                                                         supplier));
        return incTunedClassifier;
    }

    public static IncTunedClassifier buildTuned1nnV1(Function<Instances, ParamSpace> paramSpaceFunction) {
        return buildTunedKnn(paramSpaceFunction, tsml.classifiers.distance_based.knn.Configs::build1nnV1);
    }
    
    public static IncTunedClassifier buildTuned1nnV1(ParamSpace paramSpace) {
        return buildTunedKnn(i -> paramSpace, tsml.classifiers.distance_based.knn.Configs::build1nnV1);
    }
    
    public static IncTunedClassifier buildTuned1nnV2(Function<Instances, ParamSpace> paramSpaceFunction) {
        return buildTunedKnn(paramSpaceFunction, tsml.classifiers.distance_based.knn.Configs::build1nnV2);
    }

    public static IncTunedClassifier buildTuned1nnV2(ParamSpace paramSpace) {
        return buildTunedKnn(i -> paramSpace, tsml.classifiers.distance_based.knn.Configs::build1nnV2);
    }
}
