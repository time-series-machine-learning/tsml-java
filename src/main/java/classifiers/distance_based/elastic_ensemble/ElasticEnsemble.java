package classifiers.distance_based.elastic_ensemble;

import classifiers.distance_based.elastic_ensemble.iteration.*;
import classifiers.distance_based.elastic_ensemble.iteration.linear.LinearIterator;
import classifiers.distance_based.elastic_ensemble.iteration.random.RandomIterator;
import classifiers.distance_based.elastic_ensemble.iteration.random.RoundRobinIterator;
import classifiers.distance_based.elastic_ensemble.selection.BestPerTypeSelector;
import classifiers.distance_based.elastic_ensemble.selection.Selector;
import classifiers.distance_based.knn.Knn;
import classifiers.distance_based.knn.sampling.DistributedRandomSampler;
import classifiers.distance_based.knn.sampling.LinearSampler;
import classifiers.distance_based.knn.sampling.RandomSampler;
import classifiers.distance_based.knn.sampling.RoundRobinRandomSampler;
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

public class ElasticEnsemble extends TemplateClassifier {

    private final static String NUM_PARAMETER_SETS_KEY = "numParametersLimit";
    private final static String NEIGHBOURHOOD_SIZE_KEY = "trainNeighbourhoodSizeLimit";
    private final static String NUM_PARAMETER_SETS_PERCENTAGE_KEY = "numParametersLimitPercentage";
    private final static String NEIGHBOURHOOD_SIZE_PERCENTAGE_KEY = "neighbourhoodSizeLimitPercentage";
    private final List<Function<Instances, ParameterSpace>> parameterSpaceGetters = new ArrayList<>();
    private final List<ParameterSpace> parameterSpaces = new ArrayList<>();
    private final List<Candidate> candidates = new ArrayList<>();
    private final List<Candidate> constituents = new ArrayList<>();
    private boolean removeDuplicateParameterSets = true;
    private Selector<Candidate> candidateSelector = new BestPerTypeSelector<>(Candidate::getParameterSpace, (candidate, other) -> {
        int comparison = Integer.compare(candidate.getKnn().getTrainNeighbourhoodSizeLimit(), other.getKnn().getTrainNeighbourhoodSizeLimit());
        if (comparison != 0) {
            return comparison;
        }
        comparison = Integer.compare(candidate.getKnn().getTrainNeighbourhoodSizeLimit(), other.getKnn().getTrainNeighbourhoodSizeLimit());
        if (comparison <= 0) {
            comparison = Comparator.comparingDouble(ClassifierResults::getAcc).compare(candidate.getKnn().getTrainResults(), other.getKnn().getTrainResults());
        }
        return comparison;
    });
    private ParameterSpacesIterationStrategy parameterSpaceIterationStrategy = ParameterSpacesIterationStrategy.RANDOM;
    private DistanceMeasureSearchStrategy parameterSearchStrategy = DistanceMeasureSearchStrategy.RANDOM; // need to define this per param space
    private Knn.NeighbourSearchStrategy neighbourSearchStrategy = Knn.NeighbourSearchStrategy.RANDOM; // need to define this per param space
    public final static int DEFAULT_NUM_PARAMETER_SETS = -1;
    private int numParametersLimit = DEFAULT_NUM_PARAMETER_SETS;
    private int parameterCount = 0;
    public final static int DEFAULT_NEIGHBOURHOOD_SIZE = -1;
    private int trainNeighbourhoodSizeLimit = DEFAULT_NEIGHBOURHOOD_SIZE;
    public final static double DEFAULT_NUM_PARAMETER_SETS_PERCENTAGE = -1;
    private double numParametersLimitPercentage = DEFAULT_NUM_PARAMETER_SETS_PERCENTAGE;
    public final static double DEFAULT_NEIGHBOURHOOD_SIZE_PERCENTAGE = -1;
    private double neighbourhoodSizeLimitPercentage = DEFAULT_NEIGHBOURHOOD_SIZE_PERCENTAGE;
    private Iterator<IterableParameterSpace> parameterSpaceIterator;
    public final static int DEFAULT_TRAIN_SUB_SET_SIZE = -1;
    public final static int DEFAULT_TRAIN_SUB_SET_SIZE_PERCENTAGE = -1;
    private final List<Instance> trainNeighbours = new ArrayList<>();
    private boolean progressive = false;

    public int getTrainEstimateSetSize() {
        return trainEstimateSetSize;
    }

    public void setTrainEstimateSetSize(final int trainEstimateSetSize) {
        this.trainEstimateSetSize = trainEstimateSetSize;
    }

    private int trainEstimateSetSize = DEFAULT_TRAIN_SUB_SET_SIZE;
    private double trainEstimateSetSizePercentage = DEFAULT_TRAIN_SUB_SET_SIZE_PERCENTAGE;
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

    public int getTrainNeighbourhoodSizeLimit() {
        return trainNeighbourhoodSizeLimit;
    }

    public void setTrainNeighbourhoodSizeLimit(final int trainNeighbourhoodSizeLimit) {
        this.trainNeighbourhoodSizeLimit = trainNeighbourhoodSizeLimit;
    }

    public int getNumParametersLimit() {
        return numParametersLimit;
    }

    public void setNumParametersLimit(final int numParametersLimit) {
        this.numParametersLimit = numParametersLimit;
    }

    public List<Function<Instances, ParameterSpace>> getParameterSpaceGetters() {
        return parameterSpaceGetters;
    }

    public void setParameterSpaceGetters(List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        this.parameterSpaceGetters.clear();
        this.parameterSpaceGetters.addAll(parameterSpaceGetters);
    }

    public Selector<Candidate> getCandidateSelector() {
        return candidateSelector;
    }

    public void setCandidateSelector(final Selector<Candidate> candidateSelector) {
        this.candidateSelector = candidateSelector;
    }

    public ParameterSpacesIterationStrategy getParameterSpaceIterationStrategy() {
        return parameterSpaceIterationStrategy;
    }

    public void setParameterSpaceIterationStrategy(final ParameterSpacesIterationStrategy parameterSpaceIterationStrategy) {
        this.parameterSpaceIterationStrategy = parameterSpaceIterationStrategy;
    }

    public DistanceMeasureSearchStrategy getParameterSearchStrategy() {
        return parameterSearchStrategy;
    }

    public void setParameterSearchStrategy(final DistanceMeasureSearchStrategy parameterSearchStrategy) {
        this.parameterSearchStrategy = parameterSearchStrategy;
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
                String.valueOf(getNumParametersLimit()),
                NEIGHBOURHOOD_SIZE_KEY,
                String.valueOf(getTrainNeighbourhoodSizeLimit())
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
                setNumParametersLimit(Integer.parseInt(value));
            } else if (key.equals(NEIGHBOURHOOD_SIZE_KEY)) {
                setTrainNeighbourhoodSizeLimit(Integer.parseInt(value));
            } else if (key.equals(NEIGHBOURHOOD_SIZE_PERCENTAGE_KEY)) {
                setNeighbourhoodSizeLimitPercentage(Double.parseDouble(value));
            } else if (key.equals(NUM_PARAMETER_SETS_PERCENTAGE_KEY)) {
                setNumParametersLimitPercentage(Double.parseDouble(value));
            }
        }
    }

    private void setupNeighbourhoodSize() {
        if (neighbourhoodSizeLimitPercentage >= 0) {
            setTrainNeighbourhoodSizeLimit((int) (neighbourhoodSizeLimitPercentage * trainSet.size()));
        }
    }

    private void setupNumParameterSets() {
        if (numParametersLimitPercentage >= 0) {
            int size = 0;
            for (ParameterSpace parameterSpace : parameterSpaces) {
                size += parameterSpace.size();
            }
            numParametersLimit = (int) (numParametersLimitPercentage * size);
        }
    }

    public double getNumParametersLimitPercentage() {
        return numParametersLimitPercentage;
    }

    public void setNumParametersLimitPercentage(final double numParametersLimitPercentage) {
        this.numParametersLimitPercentage = numParametersLimitPercentage;
    }

    public double getNeighbourhoodSizeLimitPercentage() {
        return neighbourhoodSizeLimitPercentage;
    }

    public void setNeighbourhoodSizeLimitPercentage(final double neighbourhoodSizeLimitPercentage) {
        this.neighbourhoodSizeLimitPercentage = neighbourhoodSizeLimitPercentage;
    }

    private Iterator<IterableParameterSpace> getParameterSpacesIterator(List<IterableParameterSpace> iterableParameterSpaces) {
        switch (parameterSpaceIterationStrategy) {
            case RANDOM: return parameterSpaceIterator = new RandomIterator<>(iterableParameterSpaces, getTrainRandom().nextLong());
            case ROUND_ROBIN: return parameterSpaceIterator = new RoundRobinIterator<>(iterableParameterSpaces);
            default: throw new IllegalStateException(parameterSpaceIterationStrategy.name() + " not implemented");
        }
    }

    private ParameterSetIterator getParameterSetIterator(ParameterSpace parameterSpace) {
        ArrayList<Integer> values =
                new ArrayList<>(Arrays.asList(ArrayUtilities.box(ArrayUtilities.range(parameterSpace.size() - 1))));
        switch (parameterSearchStrategy) {
            case RANDOM: return new ParameterSetIterator(parameterSpace, new RandomIterator<>(values, getTrainRandom().nextLong()));
//            case SPREAD: return new ParameterSetIterator(parameterSpace, new SpreadIterator<>(values));
            case LINEAR: return new ParameterSetIterator(parameterSpace, new LinearIterator<>(values));
            default: throw new IllegalStateException(parameterSearchStrategy.name() + " not implemented yet");
        }
    }

    private boolean hasRemainingParameterSets() {
        return (parameterCount < numParametersLimit || numParametersLimit < 0) && parameterSpaceIterator.hasNext();
    }

    private void setupTrainEstimateSetSize() {
        if(trainEstimateSetSizePercentage >= 0) {
            setTrainEstimateSetSize((int) (trainSet.size() * trainEstimateSetSizePercentage));
        }
    }

    private void checkTrainSet(Instances trainSet) {
        if (trainSetChanged(trainSet)) {
            getTrainStopWatch().reset();
            this.trainSet = trainSet;
            candidates.clear();
            trainNeighbours.clear();
            candidateSelector.setRandom(getTrainRandom()); // todo make this into enum system
            candidateSelector.clear();
            parameterSpaces.clear();
            parameterCount = 0;
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
            parameterSpaceIterator = getParameterSpacesIterator(iterableParameterSpaces);
            setupNeighbourhoodSize();
            setupNumParameterSets();
            setupTrainEstimateSetSize();
            setupNeighbourSearchStrategy();
            getTrainStopWatch().lap();
        }
    }

    private void setupNeighbourSearchStrategy() {
        DynamicIterator<Instance, ?> neighboursIterator;
        switch (neighbourSearchStrategy) {
            case RANDOM:
                neighboursIterator = new RandomSampler(getTrainRandom().nextLong());
                break;
            case LINEAR:
                neighboursIterator = new LinearSampler();
                break;
            case ROUND_ROBIN_RANDOM:
                neighboursIterator = new RoundRobinRandomSampler(getTrainRandom());
                break;
            case DISTRIBUTED_RANDOM:
                neighboursIterator = new DistributedRandomSampler(getTrainRandom());
                break;
            default:
                throw new UnsupportedOperationException();
        }
        while (trainNeighbours.size() < trainNeighbourhoodSizeLimit && neighboursIterator.hasNext()) {
            trainNeighbours.add(neighboursIterator.next());
        }
    }

    @Override
    public void buildClassifier(final Instances trainSet) throws
            Exception {
        checkTrainSet(trainSet);
        // todo fully trained candidates for setter changes post build
        getTrainStopWatch().lap();
        if (getTrainNeighbourhoodSizeLimit() != 0) {
//            int count = 0;
            boolean remainingParameters = hasRemainingParameterSets();
            boolean remainingCandidates = !candidates.isEmpty();
            while ((remainingParameters || remainingCandidates)) {
//                System.out.println(count++);
                Knn knn;
                Candidate candidate;
                int knnIndex;
//                boolean choice = true;
//                if (remainingParameters && remainingCandidates) {
//                    choice = random.nextBoolean();
//                } else if (remainingCandidates) {
//                    choice = false;
//                }
//                if (choice) {
                if (remainingParameters) {
                    IterableParameterSpace iterableParameterSpace = parameterSpaceIterator.next();
                    Iterator<String[]> parameterSetIterator = iterableParameterSpace.getIterator();
                    ParameterSpace parameterSpace = iterableParameterSpace.getParameterSpace();
                    String[] parameters = parameterSetIterator.next();
                    parameterSetIterator.remove();
                    if (!parameterSetIterator.hasNext()) {
                        parameterSpaceIterator.remove();
                    } // todo random guess if no params or constituents
                    knn = new Knn();
                    knn.setOptions(parameters);
//                    if(progressive) {
//                        knn.setTrainNeighbourhoodSizeLimit(1);
//                    } else {
                        knn.setTrainNeighbourhoodSizeLimit(trainNeighbourhoodSizeLimit);
//                    }
                    knn.setEarlyAbandon(true);
                    knn.setNeighbourSearchStrategy(neighbourSearchStrategy);
                    knn.setSeed(getTrainRandom().nextInt());
                    knn.setPredefinedTrainNeighbourhood(trainNeighbours);
                    knn.setTrainEstimateSetSizeLimit(trainEstimateSetSize); // todo make knn adapt when train set size changes / make enum strategy
                    candidate = new Candidate(knn, parameterSpace);
                    candidates.add(candidate);
                    knnIndex = candidates.size() - 1;
                    parameterCount++;
                } else {
                    knnIndex = getTrainRandom().nextInt(candidates.size());
                    candidate = candidates.get(knnIndex);
                    knn = candidate.getKnn();
                    int sampleSize = knn.getTrainNeighbourhoodSizeLimit() + 1;
                    knn.setTrainNeighbourhoodSizeLimit(sampleSize);
                }
                if ((knn.getTrainNeighbourhoodSizeLimit() + 1 > getTrainNeighbourhoodSizeLimit()
                        && getTrainNeighbourhoodSizeLimit() >= 0)
                        || knn.getTrainNeighbourhoodSizeLimit() + 1 > trainSet.size()) {
//                    fullyTrainedCandidates.add(
                            candidates.remove(knnIndex);
//                    );
                }
                knn.setTrainContractNanos(remainingTrainContractNanos());
                knn.buildClassifier(trainSet);
                Candidate trainedCandidate = new Candidate(candidate);
                candidateSelector.add(trainedCandidate);
                remainingParameters = hasRemainingParameterSets();
                remainingCandidates = !candidates.isEmpty();
                getTrainStopWatch().lap();
            }
        }
        constituents.clear();
        constituents.addAll(candidateSelector.getSelected());
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
    }

    private boolean samplesTrainSet() {
        return trainEstimateSetSize >= 0 && trainEstimateSetSize < trainSet.size();
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

    public double getTrainEstimateSetSizePercentage() {
        return trainEstimateSetSizePercentage;
    }

    public void setTrainEstimateSetSizePercentage(final double trainEstimateSetSizePercentage) {
        this.trainEstimateSetSizePercentage = trainEstimateSetSizePercentage;
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
