package evaluation.evaluators;

import evaluation.storage.ClassifierResults;
import org.junit.Assert;
import tsml.classifiers.TSClassifier;
import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import utilities.ArrayUtilities;
import utilities.ClassifierTools;

import java.util.*;
import java.util.logging.Logger;

public class OutOfBagEvaluator extends Evaluator {

    private static final Logger DEFAULT_LOGGER = LogUtils.getLogger(OutOfBagEvaluator.class);
    private transient Logger log = DEFAULT_LOGGER;
    private TimeSeriesInstances inBagTrainData;
    private List<Integer> inBagTrainDataIndices;
    private TimeSeriesInstances outOfBagTestData;
    private List<Integer> outOfBagTestDataIndices;
    private boolean cloneClassifier = false;

    public OutOfBagEvaluator() {
        super(-1, false, false);
    }

    @Override public ClassifierResults evaluate(TSClassifier classifier, TimeSeriesInstances data) throws Exception {
        final Random random = new Random(seed);
        // build a new oob train / test data
        inBagTrainDataIndices = new ArrayList<>(data.numInstances());
        final Set<Integer> oobTestSetIndices = new HashSet<>(data.numInstances());
        oobTestSetIndices.addAll(ArrayUtilities.sequence(data.numInstances()));
        // pick n instances from train data, where n is the size of train data
        for(int i = 0; i < data.numInstances(); i++) {
            int index = random.nextInt(data.numInstances());
            TimeSeriesInstance instance = data.get(index);
            inBagTrainDataIndices.add(index);
            // remove the train instance from the test bag (if not already)
            oobTestSetIndices.remove(index);
        }
        // populate in-bag train data
        inBagTrainData = new TimeSeriesInstances(data.getClassLabels());
        for(Integer i : inBagTrainDataIndices) {
            // quick check that oob test / train are independent
            Assert.assertFalse(oobTestSetIndices.contains(i));
            TimeSeriesInstance instance = data.get(i);
            inBagTrainData.add(instance);
        }
        // populate out-of-bag test data
        outOfBagTestData = new TimeSeriesInstances(data.getClassLabels());
        outOfBagTestDataIndices = new ArrayList<>(oobTestSetIndices);
        for(Integer i : outOfBagTestDataIndices) {
            TimeSeriesInstance instance = data.get(i);
            outOfBagTestData.add(instance);
        }
        // build the tree on the oob train
        if(cloneClassifier) {
            classifier = CopierUtils.deepCopy(classifier);
        }
        classifier.buildClassifier(inBagTrainData);
        // test tree on the oob test
        ClassifierResults results = new ClassifierResults();
        ClassifierTools.addPredictions(classifier, outOfBagTestData, results, random);
        return results;
    }

    public TimeSeriesInstances getInBagTrainData() {
        return inBagTrainData;
    }

    public TimeSeriesInstances getOutOfBagTestData() {
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
