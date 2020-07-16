package evaluation.evaluators;

import evaluation.storage.ClassifierResults;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.classifiers.Utils;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import tsml.classifiers.distance_based.utils.system.random.RandomSource;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.logging.Logger;

public class OutOfBagEvaluator extends Evaluator implements Loggable {

    private static final Logger DEFAULT_LOGGER = LogUtils.buildLogger(OutOfBagEvaluator.class);
    private transient Logger logger = DEFAULT_LOGGER;
    private Instances baggedTrainData;
    private Instances outOfBagTestData;

    public OutOfBagEvaluator() {
        super(-1, false, false);
    }

    public OutOfBagEvaluator(int seed) {
        this();
        setSeed(seed);
    }

    @Override public Logger getLogger() {
        return logger;
    }

    @Override public void setLogger(final Logger logger) {
        this.logger = logger;
    }

    @Override public ClassifierResults evaluate(final Classifier classifier, Instances data) throws Exception {
        final Random random = new Random(seed);
        // build a new oob train / test data
        baggedTrainData = new Instances(data, 0);
        final Set<Instance> oobTestSet = new HashSet<>(data.size());
        oobTestSet.addAll(data);
        // pick n instances from train data, where n is the size of train data
        for(int i = 0; i < data.size(); i++) {
            int index = random.nextInt(data.size());
            Instance instance = data.get(index);
            // add instance to the train bag
            baggedTrainData.add(instance);
            // remove the train instance from the test bag (if no already)
            oobTestSet.remove(instance);
        }
        // quick check that oob test / train are independent
        for(Instance i : baggedTrainData) {
            Assert.assertFalse(oobTestSet.contains(i));
        }
        // convert test data from set to instances
        outOfBagTestData = new Instances(data, 0);
        outOfBagTestData.addAll(oobTestSet);
        // build the tree on the oob train
        getLogger().info("training on bagged train data");
        classifier.buildClassifier(baggedTrainData);
        // test tree on the oob test
        ClassifierResults results = new ClassifierResults();
        getLogger().info("testing on out-of-bag test data");
        Utils.addPredictions(classifier, outOfBagTestData, results);
        return results;
    }

    public Instances getBaggedTrainData() {
        return baggedTrainData;
    }

    public Instances getOutOfBagTestData() {
        return outOfBagTestData;
    }

}
