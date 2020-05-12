package tsml.classifiers.distance_based.proximity;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import java.util.ArrayList;
import java.util.List;
import tsml.classifiers.distance_based.utils.classifier_mixins.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifier_mixins.Utils;
import utilities.ArrayUtilities;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ProximityForest extends BaseClassifier {

    public static void main(String[] args) throws Exception {
        int seed = 1;
        ProximityForest classifier = new ProximityForest();
        classifier.setEstimateOwnPerformance(true);
        classifier.setSeed(seed);
        classifier.setConfig100TreeLimit();
        Utils.trainTestPrint(classifier, DatasetLoading.sampleGunPoint(seed));
    }

    // -------------------- configs --------------------

    public ProximityForest setConfig100TreeLimit() {
        return setNumTreeLimit(100);
    }

    public ProximityForest setConfig200TreeLimit() {
        return setNumTreeLimit(200);
    }

    public ProximityForest setConfig500TreeLimit() {
        return setNumTreeLimit(500);
    }

    public ProximityForest setConfig1000TreeLimit() {
        return setNumTreeLimit(1000);
    }

    // -------------------- end configs --------------------

    public ProximityForest() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
    }

    private final List<ProximityTree> trees = new ArrayList<>();
    private int numTreeLimit = 100;
    private final TimeContracter trainTimeContracter = new TimeContracter();
    private final TimeContracter testTimeContracter = new TimeContracter();
    private final MemoryContracter trainMemoryContracter = new MemoryContracter();
    private final MemoryContracter testMemoryContracter = new MemoryContracter();
    private long longestTreeBuildTimeNanos = 0;
    private ConstituentConfig constituentConfig = ProximityTree::setConfigR1;
    private boolean useDistributionInVoting = false;

    @Override
    public void buildClassifier(final Instances trainData) throws Exception {
        trainMemoryContracter.getWatcher().enable();
        trainTimeContracter.getTimer().enable();
        if(isRebuild()) {
            trainMemoryContracter.getWatcher().resetAndEnable();
            trainTimeContracter.getTimer().resetAndEnable();
            super.buildClassifier(trainData);
            trees.clear();
            longestTreeBuildTimeNanos = 0;
        }
        while(
            (!hasNumTreeLimit() || trees.size() < numTreeLimit)
            &&
            (!trainTimeContracter.hasTimeLimit() || trainTimeContracter.getRemainingTrainTime() > longestTreeBuildTimeNanos)
        ) {
            getLogger().info("building tree " + trees.size());
            ProximityTree tree = new ProximityTree();
            trees.add(tree);
            constituentConfig.setConfig(tree);
            tree.setSeed(seed);
            tree.setEstimateOwnPerformance(getEstimateOwnPerformance());
            // todo should resource monitors be disabled and then retrospectively added after build?
            tree.buildClassifier(trainData);
        }
        if(getEstimateOwnPerformance()) {
            trainResults = new ClassifierResults();
            double[][] distributions = new double[trainData.size()][getNumClasses()];
            for(ProximityTree tree : trees) {
                Instances oobTest = tree.getOobTest();
                for()
            }
        }
        getLogger().info("build complete");
    }

    @Override
    public double[] distributionForInstance(final Instance instance) throws Exception {
        final double[] finalDistribution = new double[getNumClasses()];
        for(ProximityTree tree : trees) {
            final double[] distribution = tree.distributionForInstance(instance);
            if(useDistributionInVoting) {
                ArrayUtilities.addInPlace(finalDistribution, distribution);
            } else {
                // majority vote
                double index = Utilities.argMax(distribution, rand);
                finalDistribution[(int) index]++;
            }
        }
        return finalDistribution;
    }

    public boolean hasNumTreeLimit() {
        return numTreeLimit > 0;
    }

    public int getNumTreeLimit() {
        return numTreeLimit;
    }

    public ProximityForest setNumTreeLimit(final int numTreeLimit) {
        this.numTreeLimit = numTreeLimit;
        return this;
    }

    public ConstituentConfig getConstituentConfig() {
        return constituentConfig;
    }

    public ProximityForest setConstituentConfig(
        final ConstituentConfig constituentConfig) {
        this.constituentConfig = constituentConfig;
        return this;
    }

    public boolean isUseDistributionInVoting() {
        return useDistributionInVoting;
    }

    public ProximityForest setUseDistributionInVoting(final boolean useDistributionInVoting) {
        this.useDistributionInVoting = useDistributionInVoting;
        return this;
    }

    public interface ConstituentConfig {
        ProximityTree setConfig(ProximityTree tree);
    }
}
