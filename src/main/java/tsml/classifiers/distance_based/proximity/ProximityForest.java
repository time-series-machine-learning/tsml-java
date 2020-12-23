package tsml.classifiers.distance_based.proximity;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.Evaluator;
import evaluation.evaluators.OutOfBagEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import tsml.classifiers.TrainEstimateTimeable;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.Builder;
import tsml.classifiers.distance_based.utils.classifiers.ClassifierFromEnum;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointed;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
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
//            classifier.setEstimateOwnPerformance(true);
//            classifier.setEstimatorMethod("oob");
            classifier.setSeed(seed);
//            classifier.setNumTreeLimit(3);
//            classifier.setCheckpointPath("checkpoints");
//            classifier.setNumTreeLimit(14);
//            classifier.setCheckpointPath("checkpoints/PF");
//            classifier.setTrainTimeLimit(10, TimeUnit.SECONDS);
//            classifier.setTrainTimeLimit(30, TimeUnit.SECONDS);
            ClassifierTools
                    .trainTestPrint(classifier, DatasetLoading.sampleDataset("/bench/phd/data/all", "ItalyPowerDemand", seed), seed);
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
        PF() {
            @Override public <B extends ProximityForest> B configure(final B classifier) {
                return PF_R5.configure(classifier);
            }
        },
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

        @Override public ProximityForest newInstance() {
            return new ProximityForest();
        }
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
    private StopWatch runTimer = new StopWatch();
    // how long testing took
    private StopWatch testTimer = new StopWatch();
    // the longest tree build time for predicting train time requirements
    private long longestTrainStageTimeNanos;
    // the method of setting the config of the trees
    private Builder<ProximityTree> proximityTreeBuilder;
    // checkpoint config
    private long lastCheckpointTimeStamp = -1;
    private String checkpointPath;
    private String checkpointFileName = Checkpointed.DEFAULT_CHECKPOINT_FILENAME;
    private boolean checkpointLoadingEnabled = true;
    private long checkpointInterval = Checkpointed.DEFAULT_CHECKPOINT_INTERVAL;
    private StopWatch checkpointTimer = new StopWatch();
    // train estimate variables
    private double[][] trainEstimateDistributions;
    private StopWatch evaluationTimer = new StopWatch();
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

    @Override
    public void buildClassifier(TimeSeriesInstances trainData) throws Exception {
        // timings:
            // train time tracks the time spent processing the algorithm. This should not be used for contracting.
            // run time tracks the entire time spent processing, whether this is work towards the algorithm or otherwise (e.g. saving checkpoints to disk). This should be used for contracting.
            // evaluation time tracks the time spent evaluating the quality of the classifier, i.e. producing an estimate of the train data error.
            // checkpoint time tracks the time spent loading / saving the classifier to disk.
        // record the start time
        final long startTimeStamp = System.nanoTime();
        runTimer.start(startTimeStamp);
        // check the other timers are disabled
        checkpointTimer.checkStopped();
        evaluationTimer.checkStopped();
        // several scenarios for entering this method:
            // 1) from scratch: isRebuild() is true
                // 1a) checkpoint found and loaded, resume from wherever left off
                // 1b) checkpoint not found, therefore initialise classifier and build from scratch
            // 2) rebuild off, i.e. buildClassifier has been called before and already handled 1a or 1b. We can safely continue building from current state. This is often the case if a smaller contract has been executed (e.g. 1h), then the contract is extended (e.g. to 2h) and we enter into this method again. There is no need to reinitialise / discard current progress - simply continue on under new constraints.
        if(isRebuild()) {
            // case (1)
            // load from a checkpoint
            // use a separate timer to handle this as the instance-based timer variables get overwritten with the ones from the checkpoint
            StopWatch loadCheckpointTimer = new StopWatch(true);
            boolean checkpointLoaded = loadCheckpoint();
            // finished loading the checkpoint
            loadCheckpointTimer.stop();
            // if there was a checkpoint and it was loaded        
            if(checkpointLoaded) {
                // case (1a)
                // update the run timer with the start time of this session
                runTimer.start(startTimeStamp);
                // just carry on with build as loaded from a checkpoint
                getLog().info("checkpoint loaded");
                // sanity check timer states
                runTimer.checkStarted();
                checkpointTimer.checkStopped();
                evaluationTimer.checkStopped();
            } else {
                // case (1b)
                // let super build anything necessary (will handle isRebuild accordingly in super class)
                super.buildClassifier(trainData);
                // if rebuilding
                // then init vars
                // build timer is already started so just clear any time already accrued from previous builds. I.e. keep the time stamp of when the timer was started, but clear any record of accumulated time
                runTimer.resetElapsedTime();
                // clear other timers entirely
                evaluationTimer.stopAndReset();
                checkpointTimer.stopAndReset();
                // no constituents to start with
                constituents = new ArrayList<>();
                // zero tree build time so the first tree build will always set the bar
                longestTrainStageTimeNanos = 0;
                // init the running train estimate variables if using OOB
                if(estimateOwnPerformance && estimator.equals(EstimatorMethod.OOB)) {
                    trainEstimatePredictionTimes = new long[trainData.numInstances()];
                    trainEstimateDistributions = new double[trainData.numInstances()][trainData.numClasses()];
                }
            }
            // add the time to load the checkpoint onto the checkpoint timer (irrelevant of whether rebuilding or not)
            checkpointTimer.add(loadCheckpointTimer.elapsedTime());
        } // else case (2)
        LogUtils.logTimeContract(runTimer.elapsedTime(), trainTimeLimit, getLog(), "train");
        // whether work has been done in this call to buildClassifier
        boolean workDone = false;
        // maintain a timer for how long trees take to build
        final StopWatch trainStageTimer = new StopWatch();
        // while remaining time / more trees need to be built
        if(estimateOwnPerformance && estimator.equals(EstimatorMethod.CV)) {
            // if there's a train contract then need to spend half the time CV'ing
            evaluationTimer.start();
            getLog().info("cross validating");
            final ProximityForest pf = new ProximityForest();
            // copy over the same configuration as this instance
            pf.deepCopyFrom(this);
            // turn off train estimation otherwise infinite recursion!
            pf.setEstimateOwnPerformance(false);
            // reset the state of pf to build from scratch
            pf.setRebuild(true);
            // reset the timers (as these are copies of our currently running timers in this obj)
            pf.runTimer.stopAndReset();
            pf.evaluationTimer.stopAndReset();
            pf.checkpointTimer.stopAndReset();
            pf.testTimer.stopAndReset();
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
                pf.setTrainTimeLimit(findRemainingTrainTime(runTimer.elapsedTime()) / 2 / (numFolds + 1) + 1);
            }
            // evaluate
            trainResults = cv.evaluate(pf, trainData);
            // stop timer and set meta info in results
            evaluationTimer.stop();
            LogUtils.logTimeContract(runTimer.elapsedTime(), trainTimeLimit, getLog(), "train");
        }
        while(
                // there's remaining trees to be built
                insideNumTreeLimit()
                &&
                // and there's remaining time left to build more trees
                insideTrainTimeLimit(runTimer.elapsedTime() + longestTrainStageTimeNanos)
        ) {
            // reset the tree build timer
            trainStageTimer.resetAndStart();
            // setup a new tree
            final int treeIndex = constituents.size();
            final ProximityTree tree = proximityTreeBuilder.build();
            final int constituentSeed = rand.nextInt();
            tree.setSeed(constituentSeed);
            // setup the constituent
            final Constituent constituent = new Constituent();
            constituent.setProximityTree(tree);
            constituents.add(constituent);
            // estimate the performance of the tree
            if(estimator.equals(EstimatorMethod.OOB)) {
                // the timer for contracting the estimate of train error
                evaluationTimer.start();
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
                treeEvaluationResults.setErrorEstimateMethod(getEstimatorMethod());
                evaluationTimer.stop();
            }
            // build the tree if not producing train estimate OR rebuild after evaluation
            getLog().info(() -> "building tree " + treeIndex);
            tree.setRebuild(true);
            tree.buildClassifier(trainData);
            // tree fully built
            trainStageTimer.stop();
            workDone = true;
            // update longest tree build time
            longestTrainStageTimeNanos = Math.max(longestTrainStageTimeNanos, trainStageTimer.elapsedTime());
            // optional checkpoint
            checkpointTimer.start();
            if(saveCheckpoint()) getLog().info("saved checkpoint");
            checkpointTimer.stop();
            // update train timer
            LogUtils.logTimeContract(runTimer.elapsedTime(), trainTimeLimit, getLog(), "train");
        }
        // if work has been done towards estimating the train error via OOB
        if(estimateOwnPerformance && workDone && estimator.equals(EstimatorMethod.OOB)) {
            // must format the OOB errors into classifier results
            evaluationTimer.start();
            getLog().info("finalising train estimate");
            // add the final predictions into the results
            for(int i = 0; i < trainData.numInstances(); i++) {
                final long timeStamp = System.nanoTime();
                double[] distribution = trainEstimateDistributions[i];
                // i.e. [71, 29] --> [0.71, 0.29]
                // copies as to not alter the original distribution. This is helpful if more trees are added in the future to avoid having to denormalise
                // set ignoreZeroSum to true to produce a uniform distribution if there is no evaluation of the instance (i.e. it never ended up in an out-of-bag test data set because too few trees were built, say)
                distribution = normalise(copy(distribution), true);
                // get the prediction, rand tie breaking if necessary
                final double prediction = argMax(distribution, rand);
                final double classValue = trainData.get(i).getLabelIndex();
                trainResults.addPrediction(classValue, distribution, prediction, trainEstimatePredictionTimes[i] + (System.nanoTime() - timeStamp), null);
            }
            evaluationTimer.stop();
        }
        // save the final checkpoint
        if(workDone) {
            checkpointTimer.start();
            if(forceSaveCheckpoint()) getLog().info("saved final checkpoint");
            checkpointTimer.stop();
        }
        // sanity check that all timers have been stopped
        evaluationTimer.checkStopped();
        testTimer.checkStopped();
        checkpointTimer.checkStopped();
        // finish up
        LogUtils.logTimeContract(runTimer.elapsedTime(), trainTimeLimit, getLog(), "train");
        runTimer.stop();
        ResultUtils.setInfo(trainResults, this, trainData);
    }

    @Override
    public double[] distributionForInstance(final TimeSeriesInstance instance) throws Exception {
        // start timer
        testTimer.resetAndStart();
        // track how long every stage (i.e. every tree prediction) takes, recording the longest
        long longestTestStageTimeNanos = 0;
        // time each stage of the prediction
        final StopWatch testStageTimer = new StopWatch();
        final double[] finalDistribution = new double[getNumClasses()];
        // while there's remaining constituents to be examined and remaining test time
        for(int i = 0;
            i < constituents.size()
            &&
            (testTimeLimit <= 0 || testTimer.elapsedTime() + longestTestStageTimeNanos < testTimeLimit)
                ; i++) {
            testStageTimer.resetAndStart();
            // let the constituent vote
            final double[] distribution = vote(i, instance);
            // add the vote to the total votes
            add(finalDistribution, distribution);
            // update timings
            testStageTimer.stop();
            longestTestStageTimeNanos = Math.max(longestTestStageTimeNanos, testStageTimer.elapsedTime());
        }
        // normalise the final vote, i.e. [71,29] --> [.71,.29]
        normalise(finalDistribution);
        // finished prediction so stop timer
        testTimer.stop();
        return finalDistribution;
    }
    
    private double[] vote(int constituentIndex, TimeSeriesInstance instance) throws Exception {
        ProximityTree tree = constituents.get(constituentIndex).getProximityTree();
        double[] distribution = tree.distributionForInstance(instance);
        return vote(constituentIndex, distribution);
    }
    
    private double[] vote(int constituentIndex, double[] distribution) {
        // vote for the highest probability class
        final int index = argMax(distribution, getRandom());
        return oneHot(distribution.length, index);
    }

    @Override public boolean isModelFullyBuilt() {
        return constituents != null && constituents.size() == numTreeLimit;
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

    public Builder<ProximityTree> getProximityTreeFactory() {
        return proximityTreeBuilder;
    }

    public void setProximityTreeFactory(final Builder<ProximityTree> proximityTreeBuilder) {
        this.proximityTreeBuilder = Objects.requireNonNull(proximityTreeBuilder);
    }

    @Override public long getRunTime() {
        return runTimer.elapsedTime();
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
        // train time is the overall run time minus any time spent estimating the train error and minus any time spent checkpointing
        return getRunTime() - getTrainEstimateTime() - getCheckpointTime();
    }

    @Override public long getTrainEstimateTime() {
        return evaluationTimer.elapsedTime();
    }

    @Override public long getTestTime() {
        return testTimer.elapsedTime();
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

    @Override public long getCheckpointTime() {
        return checkpointTimer.elapsedTime();
    }

    @Override public String getParameters() {
        return CHECKPOINT_TIME_ID + "," + checkpointTimer.elapsedTime()
                + "," + super.getParameters();
    }
    
    public static final String CHECKPOINT_TIME_ID = "checkpointTime";
}
