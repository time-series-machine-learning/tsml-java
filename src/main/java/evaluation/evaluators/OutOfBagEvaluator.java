package evaluation.evaluators;

import evaluation.storage.ClassifierResults;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;
import tsml.classifiers.distance_based.utils.classifiers.Utils;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import utilities.ArrayUtilities;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.logging.Logger;

public class OutOfBagEvaluator extends Evaluator implements Loggable {

    private static final Logger DEFAULT_LOGGER = LogUtils.buildLogger(OutOfBagEvaluator.class);
    private transient Logger logger = DEFAULT_LOGGER;
    private Instances inBagTrainData;
    private List<Integer> inBagTrainDataIndices;
    private Instances outOfBagTestData;
    private List<Integer> outOfBagTestDataIndices;
    private boolean cloneClassifier = false;

    public OutOfBagEvaluator() {
        super(-1, false, false);
    }

    @Override public Logger getLogger() {
        return logger;
    }

    @Override public void setLogger(final Logger logger) {
        this.logger = logger;
    }

    @Override public ClassifierResults evaluate(Classifier classifier, Instances data) throws Exception {
        final Random random = new Random(seed);
        // build a new oob train / test data
        inBagTrainData = new Instances(data, data.size());
        inBagTrainDataIndices = new ArrayList<>(data.size());
        final Set<Instance> oobTestSet = new HashSet<>(data.size());
        final Set<Integer> oobTestSetIndices = new HashSet<>(data.size());
        oobTestSet.addAll(data);
        oobTestSetIndices.addAll(ArrayUtilities.sequence(data.size()));
        // pick n instances from train data, where n is the size of train data
        for(int i = 0; i < data.size(); i++) {
            int index = random.nextInt(data.size());
            Instance instance = data.get(index);
            inBagTrainDataIndices.add(index);
            // add instance to the train bag
            inBagTrainData.add(instance);
            // remove the train instance from the test bag (if no already)
            oobTestSet.remove(instance);
            oobTestSetIndices.remove(index);
        }
        // quick check that oob test / train are independent
        for(Instance i : inBagTrainData) {
            Assert.assertFalse(oobTestSet.contains(i));
        }
        // convert test data from set to instances
        outOfBagTestData = new Instances(data, oobTestSet.size());
        outOfBagTestData.addAll(oobTestSet);
        outOfBagTestDataIndices = new ArrayList<>(oobTestSetIndices);
        // build the tree on the oob train
        getLogger().info("training on bagged train data");
        if(cloneClassifier) {
            classifier = (Classifier) CopierUtils.deepCopy(classifier);
        }
        classifier.buildClassifier(inBagTrainData);
        // test tree on the oob test
        ClassifierResults results = new ClassifierResults();
        getLogger().info("testing on out-of-bag test data");
        Utils.addPredictions(classifier, outOfBagTestData, results);
        return results;
    }

    public Instances getInBagTrainData() {
        return inBagTrainData;
    }

    public Instances getOutOfBagTestData() {
        return outOfBagTestData;
    }

    public List<Integer> getInBagTrainDataIndices() {
        return inBagTrainDataIndices;
    }

    public boolean isCloneClassifier() {
        return cloneClassifier;
    }

    public void setCloneClassifier(final boolean cloneClassifier) {
        this.cloneClassifier = cloneClassifier;
    }

    public List<Integer> getOutOfBagTestDataIndices() {
        return outOfBagTestDataIndices;
    }
}
