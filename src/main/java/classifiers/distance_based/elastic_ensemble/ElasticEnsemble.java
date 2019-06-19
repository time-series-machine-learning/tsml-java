package classifiers.distance_based.elastic_ensemble;

import classifiers.distance_based.elastic_ensemble.iteration.*;
import classifiers.distance_based.elastic_ensemble.iteration.linear.LinearIterator;
import classifiers.distance_based.elastic_ensemble.iteration.random.RandomIterator;
import classifiers.distance_based.elastic_ensemble.iteration.random.RoundRobinIterator;
import classifiers.distance_based.elastic_ensemble.selection.BestPerTypeSelector;
import classifiers.distance_based.elastic_ensemble.selection.Selector;
import classifiers.distance_based.knn.Knn;
import classifiers.template_classifier.TemplateClassifier;
import distances.derivative_time_domain.ddtw.CachedDdtw;
import distances.derivative_time_domain.wddtw.CachedWddtw;
import distances.time_domain.dtw.Dtw;
import distances.time_domain.erp.Erp;
import distances.time_domain.lcss.Lcss;
import distances.time_domain.msm.Msm;
import distances.time_domain.twe.Twe;
import distances.time_domain.wdtw.Wdtw;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSpace;
import utilities.ArrayUtilities;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.function.Function;

import static utilities.ArrayUtilities.argMax;

public class ElasticEnsemble extends TemplateClassifier {

    private final static String NUM_PARAMETER_SETS_KEY = "numParameterSets";
    private final static String NEIGHBOURHOOD_SIZE_KEY = "neighbourhoodSize";
    private final static String NUM_PARAMETER_SETS_PERCENTAGE_KEY = "numParameterSetsPercentage";
    private final static String NEIGHBOURHOOD_SIZE_PERCENTAGE_KEY = "neighbourhoodSizePercentage";
    private final List<Function<Instances, ParameterSpace>> parameterSpaceGetters = new ArrayList<>();
    private final List<ParameterSpace> parameterSpaces = new ArrayList<>();
    private final List<Candidate> candidates = new ArrayList<>();
    private final List<Candidate> fullyTrainedCandidates = new ArrayList<>();
    private final List<Candidate> constituents = new ArrayList<>();
    private boolean removeDuplicateParameterSets = true;
    private long maxPhaseTime = 0;
    private Selector<Candidate> selector = new BestPerTypeSelector<>(Candidate::getParameterSpace, (candidate, other) -> {
        int comparison = Integer.compare(candidate.getKnn().getNeighbourhoodSize(), other.getKnn().getNeighbourhoodSize());
        if (comparison != 0) {
            return comparison;
        }
        comparison = Integer.compare(candidate.getKnn().getNeighbourhoodSize(), other.getKnn().getNeighbourhoodSize());
        if (comparison <= 0) {
            comparison = Comparator.comparingDouble(ClassifierResults::getAcc).compare(candidate.getKnn().getTrainResults(), other.getKnn().getTrainResults());
        }
        return comparison;
    });
    private ParameterSpacesIterationStrategy parameterSpacesIterationStrategy = ParameterSpacesIterationStrategy.RANDOM;
    private DistanceMeasureSearchStrategy distanceMeasureSearchStrategy = DistanceMeasureSearchStrategy.RANDOM;
    private Knn.NeighbourSearchStrategy neighbourSearchStrategy = Knn.NeighbourSearchStrategy.RANDOM;
    public final static int DEFAULT_NUM_PARAMETER_SETS = -1;
    private int numParameterSets = DEFAULT_NUM_PARAMETER_SETS;
    private int parameterSetCount = 0;
    public final static int DEFAULT_NEIGHBOURHOOD_SIZE = -1;
    private int neighbourhoodSize = DEFAULT_NEIGHBOURHOOD_SIZE;
    public final static double DEFAULT_NUM_PARAMETER_SETS_PERCENTAGE = -1;
    private double numParameterSetsPercentage = DEFAULT_NUM_PARAMETER_SETS_PERCENTAGE;
    public final static double DEFAULT_NEIGHBOURHOOD_SIZE_PERCENTAGE = -1;
    private double neighbourhoodSizePercentage = DEFAULT_NEIGHBOURHOOD_SIZE_PERCENTAGE;
    private Iterator<IterableParameterSpace> parameterSpacesIterator;
    public final static int DEFAULT_TRAIN_SUB_SET_SIZE = -1;
    public final static int DEFAULT_TRAIN_SUB_SET_SIZE_PERCENTAGE = -1;
    private boolean progressive = false;

    public int getTrainSubSetSize() {
        return trainSubSetSize;
    }

    public void setTrainSubSetSize(final int trainSubSetSize) {
        this.trainSubSetSize = trainSubSetSize;
    }

    private int trainSubSetSize = DEFAULT_TRAIN_SUB_SET_SIZE;
    private double trainSubSetSizePercentage = DEFAULT_TRAIN_SUB_SET_SIZE_PERCENTAGE;
    private List<Instance> trainSet;

    public ElasticEnsemble() {
        this(getClassicParameterSpaceGetters());
    }

    public ElasticEnsemble(Function<Instances, ParameterSpace>... parameterSpaceGetters) {
        this(Arrays.asList(parameterSpaceGetters));
    }

    public ElasticEnsemble(List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        setParameterSpaceGetters(parameterSpaceGetters);
    }

    public static List<Function<Instances, ParameterSpace>> getClassicParameterSpaceGetters() {
        return new ArrayList<>(Arrays.asList(
                instances -> Dtw.euclideanParameterSpace(),
                instances -> Dtw.fullWindowParameterSpace(),
                Dtw::discreteParameterSpace,
                instances -> CachedDdtw.fullWindowParameterSpace(),
                CachedDdtw::discreteParameterSpace,
                instances -> Wdtw.discreteParameterSpace(),
                instances -> CachedWddtw.discreteParameterSpace(),
                Lcss::discreteParameterSpace,
                Erp::discreteParameterSpace,
                instances -> Msm.discreteParameterSpace(),
                instances -> Twe.discreteParameterSpace()
        ));
    }

    public static List<Function<Instances, ParameterSpace>> getDefaultParameterSpaceGetters() {
        return new ArrayList<>(Arrays.asList(
                Dtw::allDiscreteParameterSpace,
                CachedDdtw::allDiscreteParameterSpace,
                instances -> Wdtw.discreteParameterSpace(),
                instances -> CachedWddtw.discreteParameterSpace(),
                Lcss::discreteParameterSpace,
                Erp::discreteParameterSpace,
                instances -> Msm.discreteParameterSpace(),
                instances -> Twe.discreteParameterSpace()
        ));
    }

    public static List<ParameterSpace> getParameterSpaces(Instances instances, List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        List<ParameterSpace> parameterSpaces = new ArrayList<>();
        for (Function<Instances, ParameterSpace> getter : parameterSpaceGetters) {
            ParameterSpace parameterSpace = getter.apply(instances);
            parameterSpaces.add(parameterSpace);
        }
        return parameterSpaces;
    }

    @Override
    public String toString() {
        return "ee";
    }

    public boolean isRemoveDuplicateParameterSets() {
        return removeDuplicateParameterSets;
    }

    public void setRemoveDuplicateParameterSets(final boolean removeDuplicateParameterSets) {
        this.removeDuplicateParameterSets = removeDuplicateParameterSets;
    }

    public int getNeighbourhoodSize() {
        return neighbourhoodSize;
    }

    public void setNeighbourhoodSize(final int neighbourhoodSize) {
        this.neighbourhoodSize = neighbourhoodSize;
    }

    public int getNumParameterSets() {
        return numParameterSets;
    }

    public void setNumParameterSets(final int numParameterSets) {
        this.numParameterSets = numParameterSets;
    }

    public List<Function<Instances, ParameterSpace>> getParameterSpaceGetters() {
        return parameterSpaceGetters;
    }

    public void setParameterSpaceGetters(List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        this.parameterSpaceGetters.clear();
        this.parameterSpaceGetters.addAll(parameterSpaceGetters);
    }

    public Selector<Candidate> getSelector() {
        return selector;
    }

    public void setSelector(final Selector<Candidate> selector) {
        this.selector = selector;
    }

    public ParameterSpacesIterationStrategy getParameterSpacesIterationStrategy() {
        return parameterSpacesIterationStrategy;
    }

    public void setParameterSpacesIterationStrategy(final ParameterSpacesIterationStrategy parameterSpacesIterationStrategy) {
        this.parameterSpacesIterationStrategy = parameterSpacesIterationStrategy;
    }

    public DistanceMeasureSearchStrategy getDistanceMeasureSearchStrategy() {
        return distanceMeasureSearchStrategy;
    }

    public void setDistanceMeasureSearchStrategy(final DistanceMeasureSearchStrategy distanceMeasureSearchStrategy) {
        this.distanceMeasureSearchStrategy = distanceMeasureSearchStrategy;
    }

    public Knn.NeighbourSearchStrategy getNeighbourSearchStrategy() {
        return neighbourSearchStrategy;
    }

    public void setNeighbourSearchStrategy(final Knn.NeighbourSearchStrategy neighbourSearchStrategy) {
        this.neighbourSearchStrategy = neighbourSearchStrategy;
    }

    @Override
    public String[] getOptions() { // todo update
        return ArrayUtilities.concat(super.getOptions(), new String[]{
                NUM_PARAMETER_SETS_KEY,
                String.valueOf(getNumParameterSets()),
                NEIGHBOURHOOD_SIZE_KEY,
                String.valueOf(getNeighbourhoodSize())
        });
    }

    @Override
    public void setOptions(final String[] options) throws
            Exception { // todo update
        super.setOptions(options);
        for (int i = 0; i < options.length - 1; i += 2) {
            String key = options[i];
            String value = options[i + 1];
            if (key.equals(NUM_PARAMETER_SETS_KEY)) {
                setNumParameterSets(Integer.parseInt(value));
            } else if (key.equals(NEIGHBOURHOOD_SIZE_KEY)) {
                setNeighbourhoodSize(Integer.parseInt(value));
            } else if (key.equals(NEIGHBOURHOOD_SIZE_PERCENTAGE_KEY)) {
                setNeighbourhoodSizePercentage(Double.parseDouble(value));
            } else if (key.equals(NUM_PARAMETER_SETS_PERCENTAGE_KEY)) {
                setNumParameterSetsPercentage(Double.parseDouble(value));
            }
        }
    }

    private void setupNeighbourhoodSize() {
        if (neighbourhoodSizePercentage >= 0) {
            setNeighbourhoodSize((int) (neighbourhoodSizePercentage * trainSet.size()));
        }
    }

    private void setupNumParameterSets() {
        if (numParameterSetsPercentage >= 0) {
            int size = 0;
            for (ParameterSpace parameterSpace : parameterSpaces) {
                size += parameterSpace.size();
            }
            numParameterSets = (int) (numParameterSetsPercentage * size);
        }
    }

    public double getNumParameterSetsPercentage() {
        return numParameterSetsPercentage;
    }

    public void setNumParameterSetsPercentage(final double numParameterSetsPercentage) {
        this.numParameterSetsPercentage = numParameterSetsPercentage;
    }

    public double getNeighbourhoodSizePercentage() {
        return neighbourhoodSizePercentage;
    }

    public void setNeighbourhoodSizePercentage(final double neighbourhoodSizePercentage) {
        this.neighbourhoodSizePercentage = neighbourhoodSizePercentage;
    }

    private Iterator<IterableParameterSpace> getParameterSpacesIterator(List<IterableParameterSpace> iterableParameterSpaces) {
        switch (parameterSpacesIterationStrategy) {
            case RANDOM: return parameterSpacesIterator = new RandomIterator<>(iterableParameterSpaces, getTrainRandom());
            case ROUND_ROBIN: return parameterSpacesIterator = new RoundRobinIterator<>(iterableParameterSpaces);
            default: throw new IllegalStateException(parameterSpacesIterationStrategy.name() + " not implemented");
        }
    }

    private ParameterSetIterator getParameterSetIterator(ParameterSpace parameterSpace) {
        ArrayList<Integer> values =
                new ArrayList<>(Arrays.asList(ArrayUtilities.box(ArrayUtilities.range(parameterSpace.size() - 1))));
        switch (distanceMeasureSearchStrategy) {
            case RANDOM: return new ParameterSetIterator(parameterSpace, new RandomIterator<>(values, getTrainRandom()));
//            case SPREAD: return new ParameterSetIterator(parameterSpace, new SpreadIterator<>(values));
            case LINEAR: return new ParameterSetIterator(parameterSpace, new LinearIterator<>(values));
            default: throw new IllegalStateException(distanceMeasureSearchStrategy.name() + " not implemented yet");
        }
    }

    private boolean hasRemainingParameterSets() {
        return (parameterSetCount < numParameterSets || numParameterSets < 0) && parameterSpacesIterator.hasNext();
    }

    private void setupTrainSubSetSize() {
        if(trainSubSetSizePercentage >= 0) {
            setTrainSubSetSize((int) (trainSet.size() * trainSubSetSizePercentage));
        }
    }

    @Override
    public void buildClassifier(final Instances trainSet) throws
            Exception {
        Random random = getTrainRandom();
        if (trainSetChanged(trainSet)) {
            getTrainStopWatch().reset();
            this.trainSet = trainSet;
            candidates.clear();
            selector.setRandom(random); // todo make this into enum system
            selector.clear();
            fullyTrainedCandidates.clear();
            parameterSpaces.clear();
            parameterSetCount = 0;
            parameterSpaces.addAll(getParameterSpaces(trainSet, parameterSpaceGetters));
            if (removeDuplicateParameterSets) {
                for (ParameterSpace parameterSpace : parameterSpaces) {
                    parameterSpace.removeDuplicateValues();
                }
            }
            List<IterableParameterSpace> iterableParameterSpaces = new ArrayList<>();
            for (ParameterSpace parameterSpace : parameterSpaces) {
                Iterator<String[]> iterator = getParameterSetIterator(parameterSpace);
                if (iterator.hasNext()) {
                    iterableParameterSpaces.add(new IterableParameterSpace(parameterSpace, iterator));
                }
            }
            parameterSpacesIterator = getParameterSpacesIterator(iterableParameterSpaces);
            setupNeighbourhoodSize();
            setupNumParameterSets();
            setupTrainSubSetSize();
            getTrainStopWatch().lap();
        }
        if (!fullyTrainedCandidates.isEmpty() && neighbourhoodSize > fullyTrainedCandidates.get(0).getKnn().getNeighbourhoodSize()) {
            candidates.addAll(fullyTrainedCandidates);
            fullyTrainedCandidates.clear();
        }
        getTrainStopWatch().lap();
        if (getNeighbourhoodSize() != 0) {
//            int count = 0;
            boolean remainingParameters = hasRemainingParameterSets();
            boolean remainingCandidates = !candidates.isEmpty();
            while ((remainingParameters || remainingCandidates) && remainingTrainContractNanos() > maxPhaseTime) {
//                System.out.println(count++);
                long startTime = System.nanoTime();
                Knn knn;
                Candidate candidate;
                boolean choice = true;
                if (remainingParameters && remainingCandidates) {
                    choice = random.nextBoolean();
                } else if (remainingCandidates) {
                    choice = false;
                }
                int knnIndex;
                if (choice) {
                    IterableParameterSpace iterableParameterSpace = parameterSpacesIterator.next();
                    Iterator<String[]> parameterSetIterator = iterableParameterSpace.getIterator();
                    ParameterSpace parameterSpace = iterableParameterSpace.getParameterSpace();
                    String[] parameters = parameterSetIterator.next();
                    parameterSetIterator.remove();
                    if (!parameterSetIterator.hasNext()) {
                        parameterSpacesIterator.remove();
                    } // todo random guess if no params or constituents
                    knn = new Knn();
                    knn.setOptions(parameters);
                    if(progressive) {
                        knn.setNeighbourhoodSize(1);
                    } else {
                        knn.setNeighbourhoodSize(neighbourhoodSize);
                    }
                    knn.setEarlyAbandon(true);
                    knn.setNeighbourSearchStrategy(neighbourSearchStrategy);
                    knn.setSeed(random.nextInt());
                    knn.setTrainSubSetSize(trainSubSetSize); // todo make knn adapt when train set size changes / make enum strategy
                    candidate = new Candidate(knn, parameterSpace);
                    candidates.add(candidate);
                    knnIndex = candidates.size() - 1;
                    parameterSetCount++;
                } else {
                    knnIndex = random.nextInt(candidates.size());
                    candidate = candidates.get(knnIndex);
                    knn = candidate.getKnn();
                    int sampleSize = knn.getNeighbourhoodSize() + 1;
                    knn.setNeighbourhoodSize(sampleSize);
                }
                if ((knn.getNeighbourhoodSize() + 1 > getNeighbourhoodSize() && getNeighbourhoodSize() >= 0) || knn.getNeighbourhoodSize() + 1 > trainSet.size()) {
                    fullyTrainedCandidates.add(candidates.remove(knnIndex));
                }
                knn.setTrainContractNanos(remainingTrainContractNanos());
                knn.buildClassifier(trainSet);
                Candidate trainedCandidate = new Candidate(candidate);
                selector.add(trainedCandidate);
                maxPhaseTime = Long.max(System.nanoTime() - startTime, maxPhaseTime);
                remainingParameters = hasRemainingParameterSets();
                remainingCandidates = !candidates.isEmpty();
                getTrainStopWatch().lap();
            }
        }
        constituents.clear();
        constituents.addAll(selector.getSelected());
        String savePath = getSavePath();
        if (savePath != null) {
            Utilities.mkdir(savePath);
        }
        if (savePath != null) {
            for (int i = 0; i < constituents.size(); i++) {
                Knn knn = constituents.get(i).getKnn();
                ClassifierResults results = knn.getTrainResults();
                results.writeFullResultsToFile(savePath + "/train" + i + ".csv");
            }
        }
        ClassifierResults trainResults = new ClassifierResults();
        setTrainResults(trainResults);
        for (int i = 0; i < trainSet.size(); i++) {
            long predictionTime = System.nanoTime();
            double[] distribution = new double[trainSet.numClasses()];
            for(Candidate constituent : constituents) {
                Knn knn = constituent.getKnn();
                double[] candidateDistribution;
//                if(samplesTrainSet()) {
                    candidateDistribution = knn.distributionForInstance(trainSet.get(i));
//                } else { todo use pre-computed train results if not sampling train set in knns
//                    ClassifierResults constituentTrainResults = knn.getTrainResults();
//                    candidateDistribution = constituentTrainResults.getProbabilityDistribution(i);
//                    predictionTime -= constituentTrainResults.getPredictionTimeInNanos(i);
//                }
                ArrayUtilities.multiply(candidateDistribution, knn.getTrainResults().getAcc());
                ArrayUtilities.normaliseInplace(candidateDistribution);
                ArrayUtilities.add(distribution, candidateDistribution);
            }
            ArrayUtilities.normaliseInplace(distribution);
            predictionTime = System.nanoTime() - predictionTime;
            trainResults.addPrediction(trainSet.get(i).classValue(), distribution, Utilities.argMax(distribution, getTrainRandom()), predictionTime, null);
        }
        getTrainStopWatch().lap();
        setClassifierResultsMetaInfo(trainResults);
        if (getTrainResultsPath() != null) {
            getTrainResults().writeFullResultsToFile(getTrainResultsPath());
        }
    }

    // todo use test stopwatch below
    public ClassifierResults getTestResults(Instances testInstances) throws
            Exception {
        String savePath = getSavePath();
        if (savePath != null) {
            Utilities.mkdir(savePath);
            int i = 0;
            for (Candidate constituent : constituents) {
                constituent.getKnn().getTestResults(testInstances).writeFullResultsToFile(savePath + "/test" + i + ".csv");
                i++;
            }
        }
        getTestStopWatch().reset();
        ClassifierResults results = new ClassifierResults();
        for (Instance testInstance : testInstances) {
            long time = System.nanoTime();
            double classValue = testInstance.classValue();
            testInstance.setClassMissing();
            double[] distribution = distributionForInstance(testInstance);
            testInstance.setClassValue(classValue);
            time = System.nanoTime() - time;
            results.addPrediction(classValue, distribution, argMax(distribution), time, null);
        }
        getTestStopWatch().lap();
        setClassifierResultsMetaInfo(results);
        return results;
    }

    private boolean samplesTrainSet() {
        return trainSubSetSize >= 0 && trainSubSetSize < trainSet.size();
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
            Exception {
        double[] distribution = new double[testInstance.numClasses()];
        for(Candidate constituent : constituents) {
            Knn knn = constituent.getKnn();
            double[] candidateDistribution;
            candidateDistribution = knn.distributionForInstance(testInstance);
            ArrayUtilities.multiply(candidateDistribution, knn.getTrainResults().getAcc());
            ArrayUtilities.add(distribution, candidateDistribution);
        }
        ArrayUtilities.normaliseInplace(distribution);
        return distribution;
    }

    public double getTrainSubSetSizePercentage() {
        return trainSubSetSizePercentage;
    }

    public void setTrainSubSetSizePercentage(final double trainSubSetSizePercentage) {
        this.trainSubSetSizePercentage = trainSubSetSizePercentage;
    }

    public enum ParameterSpacesIterationStrategy {
        RANDOM,
        ROUND_ROBIN;

        public static ParameterSpacesIterationStrategy fromString(String str) {
            for (ParameterSpacesIterationStrategy s : ParameterSpacesIterationStrategy.values()) {
                if (s.name()
                        .equals(str)) {
                    return s;
                }
            }
            throw new IllegalArgumentException("No enum value by the name of " + str);
        }
    }

    public enum DistanceMeasureSearchStrategy {
        RANDOM,
        LINEAR,
        SPREAD;

        public static DistanceMeasureSearchStrategy fromString(String str) {
            for (DistanceMeasureSearchStrategy s : DistanceMeasureSearchStrategy.values()) {
                if (s.name()
                        .equals(str)) {
                    return s;
                }
            }
            throw new IllegalArgumentException("No enum value by the name of " + str);
        }
    }

    private static class IterableParameterSpace {
        private final ParameterSpace parameterSpace;
        private final Iterator<String[]> iterator;

        private IterableParameterSpace(final ParameterSpace parameterSpace,
                                       final Iterator<String[]> iterator) {
            this.parameterSpace = parameterSpace;
            this.iterator = iterator;
        }

        public Iterator<String[]> getIterator() {
            return iterator;
        }

        public ParameterSpace getParameterSpace() {
            return parameterSpace;
        }
    }

    private class Candidate {
        private final Knn knn;
        private final ParameterSpace parameterSpace;

        private Candidate(final Knn knn, final ParameterSpace parameterSpace) {
            this.knn = knn;
            this.parameterSpace = parameterSpace;
        }

        private Candidate(Candidate candidate) {
            this(candidate.knn.copy(), candidate.parameterSpace);
        }

        public Knn getKnn() {
            return knn;
        }

        public ParameterSpace getParameterSpace() {
            return parameterSpace;
        }

    }
}
