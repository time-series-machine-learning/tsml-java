package machine_learning.classifiers.tuned.incremental.configs;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import machine_learning.classifiers.tuned.incremental.*;
import org.apache.commons.collections4.Transformer;
import org.apache.commons.collections4.iterators.AbstractIteratorDecorator;
import org.apache.commons.collections4.iterators.TransformIterator;
import org.apache.commons.lang3.StringUtils;
import tsml.classifiers.distance_based.knn.KNNCV;
import utilities.ClassifierTools;
import utilities.Utilities;
import utilities.collections.Best;
import utilities.collections.Utils;
import utilities.collections.box.Box;
import utilities.iteration.RandomIterator;
import utilities.params.ParamSet;
import utilities.params.ParamSpace;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

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
        IncTunedClassifier incTunedClassifier = new IncTunedClassifier();
        incTunedClassifier.setOnTrainDataAvailable(new Inc1nnTuningSetup(incTunedClassifier,
                                                                         tsml.classifiers.distance_based.distances.Configs::buildDtwSpaceV1));
        return incTunedClassifier;
    }

    public static IncTunedClassifier buildTuned1nn(Function<Instances, ParamSpace> paramSpaceFunction) {
        IncTunedClassifier incTunedClassifier = new IncTunedClassifier();
        incTunedClassifier.setOnTrainDataAvailable(new Inc1nnTuningSetup(incTunedClassifier,
                                                                         paramSpaceFunction));
        return incTunedClassifier;
    }

    public static IncTunedClassifier buildTuned1nn(ParamSpace paramSpaceFunction) {
        IncTunedClassifier incTunedClassifier = new IncTunedClassifier();
        incTunedClassifier.setOnTrainDataAvailable(new Inc1nnTuningSetup(incTunedClassifier,
                                                                         paramSpaceFunction));
        return incTunedClassifier;
    }

}
