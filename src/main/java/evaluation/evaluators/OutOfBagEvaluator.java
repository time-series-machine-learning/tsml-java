package evaluation.evaluators;

import evaluation.storage.ClassifierResults;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import utilities.ArrayUtilities;
import utilities.ClassifierTools;
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
        inBagTrainDataIndices = new ArrayList<>(data.size());
        final Set<Integer> oobTestSetIndices = new HashSet<>(data.size());
        oobTestSetIndices.addAll(ArrayUtilities.sequence(data.size()));
        // pick n instances from train data, where n is the size of train data
        for(int i = 0; i < data.size(); i++) {
            int index = random.nextInt(data.size());
            Instance instance = data.get(index);
            inBagTrainDataIndices.add(index);
            // remove the train instance from the test bag (if not already)
            oobTestSetIndices.remove(index);
        }
        // populate in-bag train data
        inBagTrainData = new Instances(data, inBagTrainDataIndices.size());
        for(Integer i : inBagTrainDataIndices) {
            // quick check that oob test / train are independent
            Assert.assertFalse(oobTestSetIndices.contains(i));
            Instance instance = data.get(i);
            inBagTrainData.add(instance);
        }
        // populate out-of-bag test data
        outOfBagTestData = new Instances(data, oobTestSetIndices.size());
        outOfBagTestDataIndices = new ArrayList<>(oobTestSetIndices);
        for(Integer i : outOfBagTestDataIndices) {
            Instance instance = data.get(i);
            outOfBagTestData.add(instance);
        }
        // build the tree on the oob train
        getLogger().info("training on bagged train data");
        if(cloneClassifier) {
            classifier = (Classifier) CopierUtils.deepCopy(classifier);
        }
        classifier.buildClassifier(inBagTrainData);
        // test tree on the oob test
        ClassifierResults results = new ClassifierResults();
        getLogger().info("testing on out-of-bag test data");
        ClassifierTools.addPredictions(classifier, outOfBagTestData, results, random);
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
