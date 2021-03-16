package tsml.classifiers.distance_based.elastic_ensemble;

import evaluation.evaluators.Evaluator;
import evaluation.evaluators.InternalEstimateEvaluator;
import evaluation.storage.ClassifierResults;
import tsml.classifiers.TSClassifier;
import tsml.classifiers.TrainEstimateTimeable;
import tsml.classifiers.distance_based.distances.dtw.spaces.*;
import tsml.classifiers.distance_based.distances.ed.spaces.EDistanceSpace;
import tsml.classifiers.distance_based.distances.erp.spaces.ERPDistanceSpace;
import tsml.classifiers.distance_based.distances.lcss.spaces.LCSSDistanceSpace;
import tsml.classifiers.distance_based.distances.msm.spaces.MSMDistanceSpace;
import tsml.classifiers.distance_based.distances.twed.spaces.TWEDistanceSpace;
import tsml.classifiers.distance_based.distances.wdtw.spaces.WDDTWDistanceSpace;
import tsml.classifiers.distance_based.distances.wdtw.spaces.WDTWDistanceSpace;
import tsml.classifiers.distance_based.optimised.KnnAgent;
import tsml.classifiers.distance_based.optimised.OptimisedClassifier;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.configs.Configs;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.CheckpointConfig;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointed;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTest;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ProgressiveBuild;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.iteration.RandomSearch;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatchable;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import utilities.ArrayUtilities;
import utilities.Utilities;

import java.util.*;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class ElasticEnsemble extends BaseClassifier implements ContractedTrain, ContractedTest, ProgressiveBuild, Checkpointed,
                                                          MemoryWatchable, TrainEstimateTimeable {

    public final static Configs<ElasticEnsemble> CONFIGS = buildConfigs().immutable();

    public static Configs<ElasticEnsemble> buildConfigs() {
        final Configs<ElasticEnsemble> configs = new Configs<>();
        
        configs.add("EE", "Elastic ensemble with default constituents (ED, DTW, Full DTW, DDTW, Full DDTW, ERP, LCSS, MSM, TWED, WDTW, WDDTW", ElasticEnsemble::new, ee -> {
            ee.setTestTimeLimit(-1);
            ee.setTrainTimeLimit(-1);
            ee.setDistanceMeasureSpaceBuilders(newArrayList(
                    new EDistanceSpace(),
                    new DTWDistanceFullWindowSpace(),
                    new DTWDistanceSpace(),
                    new DDTWDistanceFullWindowSpace(),
                    new DDTWDistanceSpace(),
                    new WDTWDistanceSpace(),
                    new WDDTWDistanceSpace(),
                    new LCSSDistanceSpace(),
                    new ERPDistanceSpace(),
                    new TWEDistanceSpace(),
                    new MSMDistanceSpace()  
            ));
        });
        
        return configs;
    }

    public ElasticEnsemble() {
        super(true);
        CONFIGS.get("EE").configure(this);
    }

    private final StopWatch runTimer = new StopWatch();
    private final StopWatch evaluationTimer = new StopWatch();
    private final StopWatch testTimer = new StopWatch();
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    private final CheckpointConfig checkpointConfig = new CheckpointConfig();
    private long trainTimeLimit = -1;
    private long testTimeLimit = -1;
    private long longestTrainStageTime = 0;
    private List<ParamSpaceBuilder> distanceMeasureSpaceBuilders = new ArrayList<>();
    private List<OptimisedClassifier> constiteunts;
    private List<OptimisedClassifier> remainingConstituents;

    @Override public CheckpointConfig getCheckpointConfig() {
        return checkpointConfig;
    }

    @Override public long getMaxMemoryUsage() {
        return memoryWatcher.getMaxMemoryUsage();
    }

    @Override public long getTrainTime() {
        return getRunTime() - getCheckpointingTime() - getTrainEstimateTime();
    }

    @Override public long getTrainEstimateTime() {
        throw new UnsupportedOperationException();
    }

    @Override public long getRunTime() {
        return runTimer.elapsedTime();
    }

    @Override public long getTestTime() {
        return testTimer.elapsedTime();
    }

    @Override public boolean isFullyBuilt() {
        return remainingConstituents.isEmpty();
    }

    @Override public void buildClassifier(final TimeSeriesInstances trainData) throws Exception {

        final long timeStamp = System.nanoTime();
        memoryWatcher.start();
        checkpointConfig.setLogger(getLogger());
        // several scenarios for entering this method:
        // 1) from scratch: isRebuild() is true
        // 1a) checkpoint found and loaded, resume from wherever left off
        // 1b) checkpoint not found, therefore initialise classifier and build from scratch
        // 2) rebuild off, i.e. buildClassifier has been called before and already handled 1a or 1b. We can safely continue building from current state. This is often the case if a smaller contract has been executed (e.g. 1h), then the contract is extended (e.g. to 2h) and we enter into this method again. There is no need to reinitialise / discard current progress - simply continue on under new constraints.
        if(isRebuild()) {
            // case (1)
            // load from a checkpoint
            if(loadCheckpoint()) {
                memoryWatcher.start();
                checkpointConfig.setLogger(getLogger());
            } else {
                runTimer.reset();
                evaluationTimer.reset();
                checkpointConfig.resetCheckpointingTime();
                memoryWatcher.reset();
                // case (1b)
                // let super build anything necessary (will handle isRebuild accordingly in super class)
                super.buildClassifier(trainData);
                // if rebuilding
                // then init vars

                checkRandom();
                longestTrainStageTime = 0;
                // for each distance measure space
                constiteunts = new ArrayList<>();
                remainingConstituents = new LinkedList<>(); // the classifiers which are not fully built
                for(ParamSpaceBuilder builder : distanceMeasureSpaceBuilders) {
                    // build the agent to guide knn tuning
                    final KnnAgent agent = new KnnAgent();
                    agent.setParamSpaceBuilder(builder);
                    agent.setSearch(new RandomSearch());
                    agent.setEvaluatorBuilder(InternalEstimateEvaluator::new);
                    agent.setScorer(ClassifierResults::getAcc);
                    // build the optimised classifier, which uses the agent to do the optimisation
                    final OptimisedClassifier classifier = new OptimisedClassifier();
                    classifier.setAgent(agent);
                    classifier.setSeed(getSeed());
                    // kick off the classifier
                    classifier.beforeBuild();
                    if(!classifier.isFullyBuilt()) {
                        remainingConstituents.add(classifier);
                    }
                    constiteunts.add(classifier);
                }
            }  // else case (1a)

        } // else case (2)
        runTimer.start(timeStamp);
        
        // loop through tuned knns until no further increments remain or out of time
        final StopWatch trainStageTimer = new StopWatch();
        boolean workDone = false;
        // multiply up the longest train stage time to leave time for consolidating results into 1
        while(insideTrainTimeLimit(System.nanoTime() + longestTrainStageTime * constiteunts.size()) && !remainingConstituents.isEmpty()) {
            trainStageTimer.resetAndStart();
            final OptimisedClassifier classifier = remainingConstituents.remove(0);
            classifier.nextBuildStep();
            if(classifier.hasNextBuildStep()) {
                remainingConstituents.add(classifier);
            }
            workDone = true;
            saveCheckpoint();
            longestTrainStageTime = Math.max(longestTrainStageTime, trainStageTimer.elapsedTime());
        }
        
        if(workDone || trainResults.getPredClassVals() == null) {
            // init the train results
            trainResults = new ClassifierResults();
            final double[][] distributions = new double[trainData.numInstances()][trainData.numClasses()];
            final long[] predictionTimes = new long[trainData.numInstances()];
            // consolidate train results via ensembling
            for(OptimisedClassifier classifier : constiteunts) {
                // finalise the build for the constituent
                classifier.afterBuild();
                // get the train results for the constituent
                final ClassifierResults trainResults = classifier.getTrainResults();
                final double acc = trainResults.getAcc();
                for(int i = 0; i < trainData.numInstances(); i++) {
                    final double prediction = trainResults.getPredClassValue(i);
                    distributions[i][(int) prediction] += acc;
                    predictionTimes[i] += trainResults.getPredictionTime(i);
                }
            }
            for(int i = 0; i < distributions.length; i++) {
                // normalise predictions
                ArrayUtilities.normalise(distributions[i]);
                final int prediction = Utilities.argMax(distributions[i], getRandom());
                final long predictionTime = predictionTimes[i];
                final int labelIndex = trainData.get(i).getLabelIndex();
                trainResults.addPrediction(labelIndex, distributions[i], prediction, predictionTime, null);
            }
        }
        
        memoryWatcher.stop();
        runTimer.stop();
        
        if(workDone || trainResults.getPredClassVals() == null) {
            forceSaveCheckpoint();
            ResultUtils.setInfo(trainResults, this, trainData);
        }
        
    }

    @Override public double[] distributionForInstance(final TimeSeriesInstance inst) throws Exception {
        final double[] distribution = new double[inst.numClasses()];
        for(OptimisedClassifier classifier : constiteunts) {
            final double[] constituentDistribution = classifier.distributionForInstance(inst);
            final int prediction = Utilities.argMax(constituentDistribution, getRandom());
            distribution[prediction] += classifier.getTrainResults().getAcc();
        }
        ArrayUtilities.normalise(distribution);
        return distribution;
    }

    public boolean withinTrainContract(long time) {
        return insideTrainTimeLimit(time);
    }
    
    public List<ParamSpaceBuilder> getDistanceMeasureSpaceBuilders() {
        return distanceMeasureSpaceBuilders;
    }

    public void setDistanceMeasureSpaceBuilders(
            final List<ParamSpaceBuilder> distanceMeasureSpaceBuilders) {
        this.distanceMeasureSpaceBuilders = distanceMeasureSpaceBuilders;
        Objects.requireNonNull(distanceMeasureSpaceBuilders);
    }

    public static class Constituent {
        public Constituent(final Evaluator evaluator, final TSClassifier classifier) {
            this.evaluator = Objects.requireNonNull(evaluator);
            this.classifier = Objects.requireNonNull(classifier);
        }

        private final Evaluator evaluator;
        private final TSClassifier classifier;

        public Evaluator getEvaluator() {
            return evaluator;
        }

        public TSClassifier getClassifier() {
            return classifier;
        }
    }

    public long getTestTimeLimit() {
        return testTimeLimit;
    }

    public void setTestTimeLimit(final long testTimeLimit) {
        this.testTimeLimit = testTimeLimit;
    }

    public long getTrainTimeLimit() {
        return trainTimeLimit;
    }

    public void setTrainTimeLimit(final long trainTimeLimit) {
        this.trainTimeLimit = trainTimeLimit;
    }
}
