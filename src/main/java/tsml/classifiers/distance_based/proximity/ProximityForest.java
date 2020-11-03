package tsml.classifiers.distance_based.proximity;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.Evaluator;
import evaluation.evaluators.OutOfBagEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import tsml.classifiers.TrainEstimateTimeable;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.Factory;
import tsml.classifiers.distance_based.utils.classifiers.ClassifierFromEnum;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointed;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import utilities.ArrayUtilities;
import utilities.ClassifierTools;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;

import static utilities.ArrayUtilities.*;
import static utilities.Utilities.argMax;

/**
 * Proximity Forest
 * <p>
 * Contributors: goastler
 */
public class ProximityForest extends BaseClassifier implements ContractedTrain, ContractedTest, TrainEstimateTimeable, Checkpointed {

    public static void main(String[] args) throws Exception {
        for(int i = 1; i < 2; i++) {
            int seed = i;
            ProximityForest classifier = Config.PF_R5.build();
//            classifier.setEstimateOwnPerformance(true);
            classifier.setSeed(seed);
//            classifier.setNumTreeLimit(2);
//            classifier.setCheckpointPath("checkpoints/PF");
//            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
            classifier.setTrainTimeLimit(30, TimeUnit.SECONDS);
            ClassifierTools
                    .trainTestPrint(classifier, DatasetLoading.sampleDataset("/bench/phd/data/all", "SyntheticControl", seed), seed);
        }
        //        Thread.sleep(10000);

//        String root = "/bench/phd/experiments";
//        String expName = "pf_correctness";
//        String expDir = root + "/" + expName;
//        String analysisName = "analysis_v3";
//        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(expDir + "/", analysisName, 10);
//        mce.setDatasets("/bench/phd/datasets/lists/2015.txt");
//        mce.readInClassifier("ORIG_PF", "orig", expDir + "/v1/results/");
//        mce.readInClassifier("PF_R5", "v3", expDir + "/v3/results/");
//        mce.readInClassifier("PF_WRAPPED", "wrap", expDir + "/v3/results/");
//        mce.readInClassifier("PF_R5", "v2", expDir + "/v2/results/");
//        mce.setTestResultsOnly(true);
//        mce.setUseAllStatistics();
//        mce.setIgnoreMissingResults(true);
//        mce.setBuildMatlabDiagrams(true, true);
//        mce.runComparison();
    }

    public enum Config implements ClassifierFromEnum<ProximityForest> {
        PF_R1() {
            @Override
            public <B extends ProximityForest> B configure(B proximityForest) {
                proximityForest.setClassifierName(name());
                proximityForest.setTrainTimeLimit(-1);
                proximityForest.setTestTimeLimit(-1);
                proximityForest.setEstimatorMethod("none");
                proximityForest.setNumTreeLimit(100);
                proximityForest.setProximityTreeFactory(ProximityTree.Config.PT_R1);
                return proximityForest;
            }
        },
        PF_R1_OOB() {
            @Override
            public <B extends ProximityForest> B configure(B proximityForest) {
                proximityForest = PF_R1.configure(proximityForest);
                proximityForest.setClassifierName(name());
                proximityForest.setEstimatorMethod("OOB");
                return proximityForest;
            }
        },
        PF_R1_CV() {
            @Override
            public <B extends ProximityForest> B configure(B proximityForest) {
                proximityForest = PF_R1.configure(proximityForest);
                proximityForest.setClassifierName(name());
                proximityForest.setEstimatorMethod("CV");
                return proximityForest;
            }
        },
        PF_R5() {
            @Override
            public <B extends ProximityForest> B configure(B proximityForest) {
                proximityForest = PF_R1.configure(proximityForest);
                proximityForest.setClassifierName(name());
                proximityForest.setProximityTreeFactory(ProximityTree.Config.PT_R5);
                return proximityForest;
            }
        },
        PF_R5_OOB() {
            @Override
            public <B extends ProximityForest> B configure(B proximityForest) {
                proximityForest = PF_R5.configure(proximityForest);
                proximityForest.setClassifierName(name());
                proximityForest.setEstimatorMethod("OOB");
                return proximityForest;
            }
        },
        PF_R5_CV() {
            @Override
            public <B extends ProximityForest> B configure(B proximityForest) {
                proximityForest = PF_R5.configure(proximityForest);
                proximityForest.setClassifierName(name());
                proximityForest.setEstimatorMethod("CV");
                return proximityForest;
            }
        },
        PF_R10() {
            @Override
            public <B extends ProximityForest> B configure(B proximityForest) {
                proximityForest = PF_R1.configure(proximityForest);
                proximityForest.setClassifierName(name());
                proximityForest.setProximityTreeFactory(ProximityTree.Config.PT_R10);
                return proximityForest;
            }
        },
        PF_R10_OOB() {
            @Override
            public <B extends ProximityForest> B configure(B proximityForest) {
                proximityForest = PF_R10.configure(proximityForest);
                proximityForest.setClassifierName(name());
                proximityForest.setEstimatorMethod("OOB");
                return proximityForest;
            }
        },
        PF_R10_CV() {
            @Override
            public <B extends ProximityForest> B configure(B proximityForest) {
                proximityForest = PF_R10.configure(proximityForest);
                proximityForest.setClassifierName(name());
                proximityForest.setEstimatorMethod("CV");
                return proximityForest;
            }
        },
        ;
    }

    public ProximityForest() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        Config.PF_R1.configure(this);
    }
    
    private static final long serialVersionUID = 1;
    // the list of trees in this forest
    private List<Constituent> constituents;
    // the number of trees
    private int numTreeLimit;
    // the train time limit / contract
    private long trainTimeLimit;
    // the test time limit / contract
    private long testTimeLimit;
    // how long this took to build. THIS INCLUDES THE TRAIN ESTIMATE!
    private final StopWatch trainTimer = new StopWatch();
    // how long the train estimate took
    private final StopWatch trainEstimateTimer = new StopWatch();
    // how long testing took
    private final StopWatch testTimer = new StopWatch();
    // the longest tree build time for predicting train time requirements
    private long longestTrainStageTimeNanos;
    // the method of setting the config of the trees
    private Factory<ProximityTree> proximityTreeFactory;
    // checkpoint config
    private long lastCheckpointTimeStamp = -1;
    private String checkpointPath;
    private String checkpointFileName = Checkpointed.DEFAULT_CHECKPOINT_FILENAME;
    private boolean checkpointLoadingEnabled = true;
    private long checkpointInterval = Checkpointed.DEFAULT_CHECKPOINT_INTERVAL;

    private static class Constituent implements Serializable {
        private ProximityTree proximityTree;
        private Evaluator evaluator;
        private ClassifierResults evaluationResults;

        public Evaluator getEvaluator() {
            return evaluator;
        }

        public void setEvaluator(final Evaluator evaluator) {
            this.evaluator = evaluator;
        }

        public ProximityTree getProximityTree() {
            return proximityTree;
        }

        public void setProximityTree(final ProximityTree proximityTree) {
            this.proximityTree = proximityTree;
        }

        public ClassifierResults getEvaluationResults() {
            return evaluationResults;
        }

        public void setEvaluationResults(final ClassifierResults evaluationResults) {
            this.evaluationResults = evaluationResults;
        }
    }
    
    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        // kick off resource monitors
//        // track the time from start to after loading the checkpoint. Loading checkpoints overwrites the trainTimer, not recording any time taken to load the checkpoint / do things before loading the checkpoint. This timer records that.
//        final StopWatch loadCheckpointTimer = new StopWatch(true);
        trainTimer.start();
        trainEstimateTimer.checkStopped();
        // IDE may say this var is redundant - it isn't because may be overwritten in load checkpoint
        final StopWatch trainTimerBeforeLoadCheckpoint = trainTimer;
        // load from checkpoint
        // if checkpoint exists then skip initialisation
        if(loadCheckpoint()) {
            getLog().info("loaded from checkpoint");
            // train timer has been replaced with one from checkpoint. Need to add on any time spent between the start of this func and here
            trainTimer.add(trainTimerBeforeLoadCheckpoint);
        } else {
            // no checkpoint exists
            super.buildClassifier(trainData);
            // if rebuilding (i.e. building from scratch) initialise the classifier
            if(isRebuild()) {
                // reset resources
                trainTimer.resetElapsedTime();
                trainEstimateTimer.resetElapsedTime();
                // no constituents to start with
                constituents = new ArrayList<>();
                // zero tree build time so the first tree build will always set the bar
                longestTrainStageTimeNanos = 0;
            }
        }
        LogUtils.logTimeContract(trainTimer.lap(), trainTimeLimit, getLog(), "train");
        // whether work has been done in this call to buildClassifier
        boolean rebuildTrainEstimate = false;
        // maintain a timer for how long trees take to build
        final StopWatch trainStageTimer = new StopWatch();
        // while remaining time / more trees need to be built
        if(estimateOwnPerformance && estimator.equals(EstimatorMethod.CV)) {
            // if there's a train contract then need to spend half the time CV'ing
            trainEstimateTimer.start();
            getLog().info("cross validating");
            ProximityForest pf = new ProximityForest();
            // copy over the same configuration as this instance
            pf.deepCopyFrom(this);
            // evaluate pf using cross validation
            CrossValidationEvaluator cv = new CrossValidationEvaluator();
            int numFolds = 10;
            cv.setNumFolds(numFolds);
            cv.setCloneData(false);
            cv.setSeed(getSeed());
            cv.setSetClassMissing(false);
            if(hasTrainTimeLimit()) {
                // must set PF train contract. The total CV time should be half of the train contract. This must be divided by numFolds+1 for a per-fold train contract. Note the +1 is to account for testing, as numFold test batches will be used which is the same size as the data, i.e. equivalent to another fold. Adds a single nano on in case the contract divides up to zero time.
                pf.setTrainTimeLimit(findRemainingTrainTime(trainTimer.lap()) / 2 / (numFolds + 1) + 1);
            }
            // turn off train estimation otherwise infinite recursion!
            pf.setEstimateOwnPerformance(false);
            // reset the state of pf to build from scratch
            pf.setRebuild(true);
            // reset the timers (as these are copies of our currently running timers)
            pf.trainTimer.resetAndStop();
            pf.trainEstimateTimer.resetAndStop();
            // disable checkpointing on pf
            pf.setCheckpointPath(null);
            // evaluate
            trainResults = cv.evaluate(pf, trainData);
            // stop timer and set meta info in results
            trainEstimateTimer.stop();
            LogUtils.logTimeContract(trainTimer.lap(), trainTimeLimit, getLog(), "train");
        }
        while(
                insideNumTreeLimit()
                &&
                insideTrainTimeLimit(trainTimer.lap() + longestTrainStageTimeNanos)
        ) {
            // reset the tree build timer
            trainStageTimer.resetAndStart();
            final int treeIndex = constituents.size();
            // setup a new tree
            final ProximityTree tree = proximityTreeFactory.build();
            final int constituentSeed = rand.nextInt();
            tree.setSeed(constituentSeed);
            // setup the constituent
            final Constituent constituent = new Constituent();
            constituent.setProximityTree(tree);
            constituents.add(constituent);
            // estimate the performance of the tree
            if(estimator.equals(EstimatorMethod.OOB)) {
                // the timer for contracting the estimate of train error
                final StopWatch trainEstimateTimer = new StopWatch();
                trainEstimateTimer.start();
                // build train estimate based on method
                final OutOfBagEvaluator oobe = new OutOfBagEvaluator();
                oobe.setCloneClassifier(false);
                oobe.setSeed(constituentSeed);
                constituent.setEvaluator(oobe);
                getLog().info(() -> "oob evaluating tree " + treeIndex);
                // evaluate the tree
                final ClassifierResults results = oobe.evaluate(tree, trainData);
                constituent.setEvaluationResults(results);
                // rebuild the train results as the train estimate has been changed
                rebuildTrainEstimate = true;
                results.setErrorEstimateMethod(getEstimatorMethod());
                trainEstimateTimer.stop();
            }
            // build the tree if not producing train estimate OR rebuild after evaluation
            getLog().info(() -> "building tree " + treeIndex);
            tree.setRebuild(true);
            tree.buildClassifier(trainData);
            // tree fully built
            trainStageTimer.stop();
            // update longest tree build time
            longestTrainStageTimeNanos = Math.max(longestTrainStageTimeNanos, trainStageTimer.getElapsedTimeStopped());
            // optional checkpoint
            saveCheckpoint();
            // update train timer
            LogUtils.logTimeContract(trainTimer.lap(), trainTimeLimit, getLog(), "train");
        }
        // if work has been done towards estimating the train error via OOB
        if(estimateOwnPerformance && rebuildTrainEstimate && estimator.equals(EstimatorMethod.OOB)) {
            // must format the OOB errors into classifier results
            trainEstimateTimer.start();
            getLog().info("finalising train estimate");
            // init final distributions for each train instance
            final double[][] finalDistributions = new double[trainData.size()][getNumClasses()];
            // time for each train instance prediction
            final long[] times = new long[trainData.size()];
            // go through every constituent
            for(int j = 0; j < constituents.size(); j++) {
                // add the output of that constituent to the train estimate
                final Constituent constituent = constituents.get(j);
                final Evaluator evaluator = constituent.getEvaluator();
                final ProximityTree tree = constituent.getProximityTree();
                // the indices of dataInTrainEstimate to trainData. I.e. the 0th instance in dataInTrainEstimate is the trainDataIndices.get(0) 'th instance in the train data.
                final List<Integer> trainDataIndices;
                final Instances dataInTrainEstimate;
                // the train estimate data may be different depending on the evaluation method
                if(estimator.equals(EstimatorMethod.OOB)) {
                    dataInTrainEstimate = ((OutOfBagEvaluator) evaluator).getOutOfBagTestData();
                    trainDataIndices = ((OutOfBagEvaluator) evaluator).getOutOfBagTestDataIndices();
                } else if(estimator.equals(EstimatorMethod.CV)) {
                    dataInTrainEstimate = trainData;
                    trainDataIndices = ArrayUtilities.sequence(trainData.size());
                } else {
                    throw new UnsupportedOperationException("cannot get train data from evaluator: " + evaluator);
                }
                final ClassifierResults constituentEvaluationResults = constituent.getEvaluationResults();
                // add each prediction to the results weighted by the evaluation of the constituent
                for(int i = 0; i < dataInTrainEstimate.size(); i++) {
                    long time = System.nanoTime();
                    final Instance instance = dataInTrainEstimate.get(i);
                    final int instanceIndexInTrainData = trainDataIndices.get(i);
                    double[] distribution = constituentEvaluationResults.getProbabilityDistribution(i);
                    // weight the vote of this constituent
                    distribution = vote(constituent, instance, distribution);
                    add(finalDistributions[instanceIndexInTrainData], distribution);
                    // add onto the prediction time for this instance
                    time = System.nanoTime() - time;
                    time += constituentEvaluationResults.getPredictionTime(i);
                    times[instanceIndexInTrainData] = time;
                }
            }
            // add the final predictions into the results
            for(int i = 0; i < trainData.size(); i++) {
                long time = System.nanoTime();
                double[] distribution = finalDistributions[i];
                // normalise the distribution as sum of votes has likely pushed sum of distribution >1
                normalise(distribution, true);
                double prediction = argMax(distribution, rand);
                double classValue = trainData.get(i).classValue();
                time = System.nanoTime() - time;
                times[i] += time;
                trainResults.addPrediction(classValue, distribution, prediction, times[i], null);
            }
            trainEstimateTimer.stop();
        }
        trainTimer.stop();
        trainEstimateTimer.checkStopped();
        ResultUtils.setInfo(trainResults, this, trainData);
        forceSaveCheckpoint();
    }

    @Override
    public double[] distributionForInstance(final Instance instance) throws Exception {
        testTimer.resetAndStart();
        long longestTestStageTimeNanos = 0;
        final StopWatch testStageTimer = new StopWatch();
        final double[] finalDistribution = new double[getNumClasses()];
        for(int i = 0;
            i < constituents.size()
            &&
            (testTimeLimit <= 0 || testTimer.lap() + longestTestStageTimeNanos < testTimeLimit)
                ; i++) {
            testStageTimer.resetAndStart();
            final double[] distribution = vote(constituents.get(i), instance);
            add(finalDistribution, distribution);
            testStageTimer.stop();
            longestTestStageTimeNanos = Math.max(longestTestStageTimeNanos, testStageTimer.getElapsedTimeStopped());
        }
        normalise(finalDistribution);
        testTimer.stop();
        return finalDistribution;
    }
    
    private double[] vote(Constituent constituent, Instance instance) throws Exception {
        ProximityTree tree = constituent.getProximityTree();
        double[] distribution = tree.distributionForInstance(instance);
        return vote(constituent, instance, distribution);
    }
    
    private double[] vote(Constituent constituent, Instance instance, double[] distribution) {
        // vote for the highest probability class
        final int index = argMax(distribution, getRandom());
        return oneHot(distribution.length, index);
    }

    public boolean insideNumTreeLimit() {
        return !hasNumTreeLimit() || constituents.size() < numTreeLimit;
    }

    public boolean hasNumTreeLimit() {
        return numTreeLimit > 0;
    }

    public int getNumTreeLimit() {
        return numTreeLimit;
    }

    public void setNumTreeLimit(final int numTreeLimit) {
        this.numTreeLimit = numTreeLimit;
    }

    public Factory<ProximityTree> getProximityTreeFactory() {
        return proximityTreeFactory;
    }

    public void setProximityTreeFactory(final Factory<ProximityTree> proximityTreeFactory) {
        this.proximityTreeFactory = Objects.requireNonNull(proximityTreeFactory);
    }

    @Override
    public long getTrainTimeLimit() {
        return trainTimeLimit;
    }

    @Override
    public void setTrainTimeLimit(final long nanos) {
        trainTimeLimit = nanos;
    }

    @Override
    public long getTestTimeLimit() {
        return testTimeLimit;
    }

    @Override
    public void setTestTimeLimit(final long nanos) {
        testTimeLimit = nanos;
    }

    @Override public long getTrainTime() {
        // train time is the overall build time minus any time spent estimating the train error
        return trainTimer.getElapsedTimeStopped() - getTrainEstimateTime();
    }

    @Override public long getTrainEstimateTime() {
        return trainEstimateTimer.getElapsedTimeStopped();
    }

    @Override public long getTestTime() {
        return testTimer.getElapsedTimeStopped();
    }

    @Override public long getLastCheckpointTimeStamp() {
        return lastCheckpointTimeStamp;
    }

    @Override public void setLastCheckpointTimeStamp(final long lastCheckpointTimeStamp) {
        this.lastCheckpointTimeStamp = lastCheckpointTimeStamp;
    }

    @Override public String getCheckpointFileName() {
        return checkpointFileName;
    }

    @Override public void setCheckpointFileName(final String checkpointFileName) {
        this.checkpointFileName = checkpointFileName;
    }

    @Override public String getCheckpointPath() {
        return checkpointPath;
    }

    @Override public boolean setCheckpointPath(final String checkpointPath) {
        this.checkpointPath = checkpointPath;
        return true;
    }

    @Override public void setCheckpointLoadingEnabled(final boolean checkpointLoadingEnabled) {
        this.checkpointLoadingEnabled = checkpointLoadingEnabled;
    }

    @Override public boolean isCheckpointLoadingEnabled() {
        return checkpointLoadingEnabled;
    }

    @Override public long getCheckpointInterval() {
        return checkpointInterval;
    }

    @Override public void setCheckpointInterval(final long checkpointInterval) {
        this.checkpointInterval = checkpointInterval;
    }
}
