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
import utilities.ClassifierTools;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;

import static utilities.ArrayUtilities.*;
import static utilities.Utilities.argMax;

/**
 * Proximity Forest
 * <p>
 * Contributors: goastler
 */
public class ProximityForest extends BaseClassifier implements ContractedTrain, ContractedTest, TrainEstimateTimeable, Checkpointed {

    public static void main(String[] args) throws Exception {
//        Thread.sleep(10000);
        for(int i = 1; i < 2; i++) {
            int seed = i;
            ProximityForest classifier = Config.PF_R5.build();
            classifier.setEstimateOwnPerformance(true);
            classifier.setEstimatorMethod("oob");
            classifier.setSeed(seed);
            classifier.setNumTreeLimit(14);
//            classifier.setCheckpointPath("checkpoints/PF");
//            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
//            classifier.setTrainTimeLimit(30, TimeUnit.SECONDS);
            ClassifierTools
                    .trainTestPrint(classifier, DatasetLoading.sampleDataset("/bench/phd/data/all", "ProximalPhalanxTW", seed), seed);
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
    private StopWatch buildTimer = new StopWatch();
    // how long testing took
    private StopWatch testTimer = new StopWatch();
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
    private boolean checkpointLoaded = false;
    private StopWatch checkpointTimer = new StopWatch();
    // train estimate variables
    private double[][] trainEstimateDistributions;
    private StopWatch trainEstimateTimer = new StopWatch();
    private long[] trainEstimatePredictionTimes;

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

    @Override public void copyFromSerObject(final Object obj) throws Exception {
        // keep a ref to each of the current timers
        final StopWatch origBuildTimer = this.buildTimer;
        final StopWatch origTrainEstimateTimer = this.trainEstimateTimer;
        final StopWatch origCheckpointTimer = this.checkpointTimer;
        final StopWatch origTestTimer = this.testTimer;
        // copy from another obj
        Checkpointed.super.copyFromSerObject(obj);
        // update the new timers with the originals. I.e. the buildTimer would be replaced with the one from the obj which, say, has a timing of 12s already. Suppose the orig build timer has a timing of 5s. This 5s would be discarded if left to just shallow copy. Instead, must shallow copy to obtain the timer with 12s and add on the current timer containing the 5s to finish with a timer containing 17s, i.e. adding the build time from the checkpoint onto the current build time, consolidating into one timer
        // note that the timer may not change if copying from a partial or the same object, therefore need to check for the change on each timer
        if(origBuildTimer != buildTimer) {
            // add the current time onto the new timer
            buildTimer.add(origBuildTimer);
            // reset the new timer to time from here onwards
            buildTimer.resetClock();
        }
        if(origTrainEstimateTimer != trainEstimateTimer) {
            // add the current time onto the new timer
            trainEstimateTimer.add(origTrainEstimateTimer);
            // reset the new timer to time from here onwards
            trainEstimateTimer.resetClock();
        }
        if(origCheckpointTimer != checkpointTimer) {
            // add the current time onto the new timer
            checkpointTimer.add(origCheckpointTimer);
            // reset the new timer to time from here onwards
            checkpointTimer.resetClock();
        }
        if(origTestTimer != testTimer) {
            // add the current time onto the new timer
            testTimer.add(origTestTimer);
            // reset the new timer to time from here onwards
            testTimer.resetClock();
        }
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        // various timers:
            // build timer tracks the entire run time, irrelevant of what's done during that time
            // checkpoint timer tracks the time spent loading / saving / handling checkpoints
            // train estimate timer tracks the time spent conducting evaluation to produce an estimate of train data error
        // these produce various times:
            // train time - the amount of time spent training the classifier. This excludes non-train related operations, i.e. checkpointing efforts. This is calculated from buildTime minus everything non train related
            // train estimate time - the amount of time spent evaluating the classifier to obtain a train data error
            // checkpoint time - the amount of time spent checkpointing
            // build time - the total time taken to build the classifier. This includes absolutely everything.
        
        // start the build timer to record the entire time spent building, irrelevant of whatever that may be
        buildTimer.start();
        // check the other timers are disabled
        checkpointTimer.checkStopped();
        trainEstimateTimer.checkStopped();
        testTimer.checkStopped();
        // attempt to load a checkpoint
            // 4 scenarios:
                // 1) loads from checkpoint AND rebuild off
                    // this should just load from the checkpoint and carry on where left off
                // 2) does not load from checkpoint AND rebuild off
                    // this should not load from a checkpoint but still carry on from where left off
                // 3) loads from checkpoint AND rebuild on
                    // same as (1)
                // 4) does not load from checkpoint AND rebuild on
                    // this should just rebuild and begin build from scratch
        
        // load from a checkpoint
        checkpointTimer.start();
        checkpointLoaded = loadCheckpoint();
        checkpointTimer.stop();
        // if there was a checkpoint and it was loaded        
        if(checkpointLoaded) {
            // (3) and (1) land here
            // just carry on with build as loaded from a checkpoint
            getLog().info("checkpoint loaded");
            // sanity check timer states
            buildTimer.checkStarted();
            checkpointTimer.checkStopped();
            trainEstimateTimer.checkStopped();
            testTimer.checkStopped();
        } else {
            // (2) and (4) land here
            // let super build anything necessary (will handle isRebuild accordingly in super class)
            super.buildClassifier(trainData);
            // if rebuilding
            if(isRebuild()) {
                // then init vars
                // build timer is already started so just clear any time already accrued from previous builds. I.e. keep the time stamp of when the timer was started, but clear any record of accumulated time
                buildTimer.resetElapsed();
                // clear other timers entirely
                testTimer.resetAndStop();
                trainEstimateTimer.resetAndStop();
                // checkpoint time is equal to the time spent attempting to load a checkpoint during this buildClassifier call
                final long singleCheckpointTime = checkpointTimer.lapTime();
                checkpointTimer.resetAndStop();
                // add the time taken attempting to load a checkpoint during this build classifier call
                checkpointTimer.add(singleCheckpointTime);
                // no constituents to start with
                constituents = new ArrayList<>();
                // zero tree build time so the first tree build will always set the bar
                longestTrainStageTimeNanos = 0;
                // init the running train estimate variables if using OOB
                if(estimateOwnPerformance && estimator.equals(EstimatorMethod.OOB)) {
                    trainEstimatePredictionTimes = new long[trainData.size()];
                    trainEstimateDistributions = new double[trainData.size()][trainData.numClasses()];
                }
            }
        }
        LogUtils.logTimeContract(buildTimer.lap(), trainTimeLimit, getLog(), "train");
        // whether work has been done in this call to buildClassifier
        boolean rebuildTrainEstimate = false;
        // maintain a timer for how long trees take to build
        final StopWatch trainStageTimer = new StopWatch();
        // while remaining time / more trees need to be built
        if(estimateOwnPerformance && estimator.equals(EstimatorMethod.CV)) {
            // if there's a train contract then need to spend half the time CV'ing
            trainEstimateTimer.start();
            getLog().info("cross validating");
            final ProximityForest pf = new ProximityForest();
            // copy over the same configuration as this instance
            pf.deepCopyFrom(this);
            // turn off train estimation otherwise infinite recursion!
            pf.setEstimateOwnPerformance(false);
            // reset the state of pf to build from scratch
            pf.setRebuild(true);
            // reset the timers (as these are copies of our currently running timers)
            pf.buildTimer.resetAndStop();
            pf.trainEstimateTimer.resetAndStop();
            pf.checkpointTimer.resetAndStop();
            pf.testTimer.resetAndStop();
            // disable checkpointing on pf
            pf.setCheckpointPath(null);
            // evaluate pf using cross validation
            final CrossValidationEvaluator cv = new CrossValidationEvaluator();
            final int numFolds = 10;
            cv.setNumFolds(numFolds);
            cv.setCloneData(false);
            cv.setSeed(getSeed());
            cv.setSetClassMissing(false);
            if(hasTrainTimeLimit()) {
                // must set PF train contract. The total CV time should be half of the train contract. This must be divided by numFolds+1 for a per-fold train contract. Note the +1 is to account for testing, as numFold test batches will be used which is the same size as the data, i.e. equivalent to another fold. Adds a single nano on in case the contract divides up to zero time.
                pf.setTrainTimeLimit(findRemainingTrainTime(buildTimer.lap()) / 2 / (numFolds + 1) + 1);
            }
            // evaluate
            trainResults = cv.evaluate(pf, trainData);
            // stop timer and set meta info in results
            trainEstimateTimer.stop();
            LogUtils.logTimeContract(buildTimer.lap(), trainTimeLimit, getLog(), "train");
        }
        while(
                // there's remaining trees to be built
                insideNumTreeLimit()
                &&
                // and there's remaining time left to build more trees
                insideTrainTimeLimit(buildTimer.lap() + longestTrainStageTimeNanos)
        ) {
            // reset the tree build timer
            trainStageTimer.resetAndStart();
            // setup a new tree
            final int treeIndex = constituents.size();
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
                trainEstimateTimer.start();
                // build train estimate based on method
                final OutOfBagEvaluator oobe = new OutOfBagEvaluator();
                oobe.setCloneClassifier(false);
                oobe.setSeed(constituentSeed);
                constituent.setEvaluator(oobe);
                getLog().info(() -> "oob evaluating tree " + treeIndex);
                // evaluate the tree
                final ClassifierResults treeEvaluationResults = oobe.evaluate(tree, trainData);
                constituent.setEvaluationResults(treeEvaluationResults);
                // for each index in the test data of the oobe
                final List<Integer> outOfBagTestDataIndices = oobe.getOutOfBagTestDataIndices();
                // for each instance in the oobe test data, add the distribution and prediction time to the corresponding instance predictions in the train estimate results
                for(int oobeIndex = 0; oobeIndex < outOfBagTestDataIndices.size(); oobeIndex++) {
                    final int trainDataIndex = outOfBagTestDataIndices.get(oobeIndex);
                    // get the corresponding distribution from the oobe results
                    double[] distribution = treeEvaluationResults.getProbabilityDistribution(oobeIndex);
                    distribution = vote(treeIndex, distribution);
                    // get the corresponding distribution from the train estimate distribution
                    // add tree's distribution for this instance onto the overall train estimate distribution for this instance
                    add(trainEstimateDistributions[trainDataIndex], distribution);
                    // add the prediction time from the oobe to the time for this instance in the train estimate
                    trainEstimatePredictionTimes[trainDataIndex] += treeEvaluationResults.getPredictionTime(oobeIndex);
                }
                // rebuild the train results as the train estimate has been changed
                rebuildTrainEstimate = true;
                treeEvaluationResults.setErrorEstimateMethod(getEstimatorMethod());
                trainEstimateTimer.stop();
            }
            // build the tree if not producing train estimate OR rebuild after evaluation
            getLog().info(() -> "building tree " + treeIndex);
            tree.setRebuild(true);
            tree.buildClassifier(trainData);
            // tree fully built
            trainStageTimer.stop();
            // update longest tree build time
            longestTrainStageTimeNanos = Math.max(longestTrainStageTimeNanos, trainStageTimer.elapsedTimeStopped());
            // optional checkpoint
            checkpointTimer.start();
            saveCheckpoint();
            checkpointTimer.stop();
            // update train timer
            LogUtils.logTimeContract(buildTimer.lap(), trainTimeLimit, getLog(), "train");
        }
        // if work has been done towards estimating the train error via OOB
        if(estimateOwnPerformance && rebuildTrainEstimate && estimator.equals(EstimatorMethod.OOB)) {
            // must format the OOB errors into classifier results
            trainEstimateTimer.start();
            getLog().info("finalising train estimate");
            // add the final predictions into the results
            for(int i = 0; i < trainData.size(); i++) {
                final long timeStamp = System.nanoTime();
                double[] distribution = trainEstimateDistributions[i];
                // i.e. [71, 29] --> [0.71, 0.29]
                // copies as to not alter the original distribution. This is helpful if more trees are added in the future to avoid having to denormalise
                // set ignoreZeroSum to true to produce a uniform distribution if there is no evaluation of the instance (i.e. it never ended up in an out-of-bag test data set because too few trees were built, say)
                distribution = normalise(copy(distribution), true);
                // get the prediction, rand tie breaking if necessary
                final double prediction = argMax(distribution, rand);
                final double classValue = trainData.get(i).classValue();
                trainResults.addPrediction(classValue, distribution, prediction, trainEstimatePredictionTimes[i] + (System.nanoTime() - timeStamp), null);
            }
            trainEstimateTimer.stop();
        }
        LogUtils.logTimeContract(buildTimer.lap(), trainTimeLimit, getLog(), "train");
        // sanity check that all timers have been stopped
        trainEstimateTimer.checkStopped();
        testTimer.checkStopped();
        checkpointTimer.checkStopped();
        buildTimer.stop();
        // update the results info
        ResultUtils.setInfo(trainResults, this, trainData);
        forceSaveCheckpoint();
        // print the trees
//        System.out.println(Utilities.apply(constituents, Constituent::getProximityTree));
    }

    @Override
    public double[] distributionForInstance(final Instance instance) throws Exception {
        // start timer
        testTimer.resetAndStart();
        // sanity check the other timers are stopped
        buildTimer.checkStopped();
        trainEstimateTimer.checkStopped();
        checkpointTimer.checkStopped();
        long longestTestStageTimeNanos = 0;
        // time each stage of the prediction
        final StopWatch testStageTimer = new StopWatch();
        final double[] finalDistribution = new double[getNumClasses()];
        // while there's remaining constituents to be examined and remaining test time
        for(int i = 0;
            i < constituents.size()
            &&
            (testTimeLimit <= 0 || testTimer.lap() + longestTestStageTimeNanos < testTimeLimit)
                ; i++) {
            testStageTimer.resetAndStart();
            // let the constituent vote
            final double[] distribution = vote(i, instance);
            // add the vote to the total votes
            add(finalDistribution, distribution);
            // update timings
            testStageTimer.stop();
            longestTestStageTimeNanos = Math.max(longestTestStageTimeNanos, testStageTimer.elapsedTimeStopped());
        }
        // normalise the final vote, i.e. [71,29] --> [.71,.29]
        normalise(finalDistribution);
        // sanity check the other timers are stopped
        buildTimer.checkStopped();
        trainEstimateTimer.checkStopped();
        checkpointTimer.checkStopped();
        // finished prediction so stop timer
        testTimer.stop();
        return finalDistribution;
    }
    
    private double[] vote(int constituentIndex, Instance instance) throws Exception {
        ProximityTree tree = constituents.get(constituentIndex).getProximityTree();
        double[] distribution = tree.distributionForInstance(instance);
        return vote(constituentIndex, distribution);
    }
    
    private double[] vote(int constituentIndex, double[] distribution) {
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
        return buildTimer.elapsedTimeStopped() - getTrainEstimateTime();
    }

    @Override public long getTrainEstimateTime() {
        return trainEstimateTimer.elapsedTimeStopped();
    }

    @Override public long getTestTime() {
        return testTimer.elapsedTimeStopped();
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
