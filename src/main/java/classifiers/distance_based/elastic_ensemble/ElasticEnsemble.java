package classifiers.distance_based.elastic_ensemble;

import classifiers.distance_based.elastic_ensemble.iteration.AbstractIterator;
import classifiers.distance_based.elastic_ensemble.iteration.ParameterSetIterator;
import classifiers.distance_based.elastic_ensemble.iteration.feedback.AbstractFeedbackIterator;
import classifiers.distance_based.elastic_ensemble.iteration.feedback.ThresholdIterator;
import classifiers.distance_based.elastic_ensemble.iteration.limited.LimitedIterator;
import classifiers.distance_based.elastic_ensemble.iteration.linear.LinearIterator;
import classifiers.distance_based.elastic_ensemble.iteration.linear.RoundRobinIterator;
import classifiers.distance_based.elastic_ensemble.iteration.random.RandomIterator;
import classifiers.distance_based.knn.Knn;
import classifiers.distance_based.knn.TrainEstimationSource;
import classifiers.template.TemplateClassifier;
import classifiers.template.TemplateClassifierInterface;
import classifiers.template.configuration.ConfigState;
import classifiers.template.configuration.TemplateConfig;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.ContractClassifier;
import utilities.ArrayUtilities;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;

public class ElasticEnsemble
    extends TemplateClassifier {

    @Override
    public ElasticEnsemble copy() throws
                           Exception {
        throw new UnsupportedOperationException();
    }

    private AbstractIterator<CandidateIterator> candidateIteratorIterator = null;

    private Instances trainSet; // todo needed?
    private final ConfigState<ElasticEnsembleConfig> configState = new ConfigState<>(ElasticEnsembleConfig::new);
    private ElasticEnsembleConfig config = null;
    private List<Instance> trainNeighbourhood = null;
    private List<Instance> trainEstimateSet = null;

    private void setupNeighbourSearchStrategy() {
        AbstractIterator<Instance> neighboursIterator;
        switch (config.getKnnConfiguration()
                      .getTrainNeighbourSearchStrategy()) {
            case RANDOM:
                iterator = new RandomIterator<>(getTrainRandom());
                break;
            case ROUND_ROBIN:
                iterator = new RoundRobinIterator<>();
                break;
            default:
                throw new UnsupportedOperationException();
        }
        for (CandidateIterator candidateParameterSetIterator : candidateParameterSetIterators) {
            if (candidateParameterSetIterator.hasNext()) {
                candidateIteratorIterator.add(candidateParameterSetIterator);
            }
        }
        return iterator;
    }

    private void setup(Instances trainSet) {
        configState.shift();
        if (trainSetChanged(trainSet) || configState.mustResetTrain()) {
            getTrainStopWatch().reset();
            config = configState.getCurrentConfig();
            this.trainSet = trainSet;
            List<CandidateIterator> candidateIterators = new ArrayList<>();
            trainNeighbourhood = buildTrainNeighbourhood(trainSet);
            trainEstimateSet = buildTrainEstimateSet(trainSet);
            for (CandidateIterator.Builder candidateIteratorBuilder : config.getCandidateIteratorBuilders()) {
                CandidateIterator candidateIterator = candidateIteratorBuilder.build(trainSet, getTrainRandom());
                candidateIterators.add(candidateIterator);
            }
            candidateIteratorIterator = buildCandidateIteratorIterator(candidateIterators);
            config.getKnnConfiguration()
                  .setupNeighbourhoodSize(trainSet);
            config.getKnnConfiguration()
                  .setupTrainEstimateSetSize(trainSet);
            getTrainStopWatch().lap();
        }
    }

    public static class CandidateIterator
        extends AbstractFeedbackIterator<AbstractClassifier, Double, Boolean> { // todo shift over to config struct
        public String getName() {
            return name;
        }

        @Override
        public Boolean feedback(final Double value) {
            return iterator.feedback(value);
        }

        private final Supplier<AbstractClassifier> supplier;
        private final TemplateConfig config;
        private final AbstractFeedbackIterator<ParameterSet, Double, Boolean> iterator;

        private CandidateIterator(final Supplier<AbstractClassifier> supplier, final TemplateConfig config,
                                  final String name, int threshold, int limit,
                                  AbstractIterator<ParameterSet> iterator) {
            this.supplier = supplier;
            this.config = config;
            this.name = name;
            this.iterator = new ThresholdIterator<>(new LimitedIterator<>(iterator, limit), threshold);
        }

        private final String name;

        private CandidateIterator(CandidateIterator other) throws
                                                           Exception {
            name = other.name;
            iterator = other.iterator.iterator();
            config = other.config.copy();
            supplier = other.supplier;
        }

        @Override
        public CandidateIterator iterator() {
            try {
                return new CandidateIterator(this);
            } catch (Exception e) {
                throw new IllegalStateException(e);
            }
        }

        @Override
        public AbstractClassifier next() {
            ParameterSet parameterSet = iterator.next();
            AbstractClassifier classifier = supplier.get();
            try {
                classifier.setOptions(config.getOptions());
                classifier.setOptions(parameterSet.getOptions());
            } catch (Exception e) {
                throw new IllegalStateException(e);
            }
            return classifier;
        }

        @Override
        public boolean hasNext() {
            return iterator.hasNext();
        }

        @Override
        public void add(final AbstractClassifier classifier) {
            iterator.add(new ParameterSet(classifier.getOptions()));
        }

        @Override
        public void remove() {
            iterator.remove();
        }

        public static class Builder<A extends AbstractClassifier> {
            private final String name;
            private ParameterSpace parameterSpace;
            private Function<Instances, ParameterSpace> parameterSpaceGetter;
            private int parameterSetCountThreshold = -1;
            private final Supplier<AbstractClassifier> supplier;
            private final TemplateConfig config;
            private ParameterSetSearchStrategy parameterSetSearchStrategy = ParameterSetSearchStrategy.RANDOM;
            private int parameterSetCountLimit = -1;

            public Builder(final String name,
                           final Supplier<AbstractClassifier> supplier,
                           final TemplateConfig config) {
                this.name = name;
                this.supplier = supplier;
                this.config = config;
            }

            public Builder setParameterSpaceGetter(Function<Instances, ParameterSpace> parameterSpaceGetter) {
                this.parameterSpaceGetter = parameterSpaceGetter;
                return this;
            }

            public Builder setParameterSpace(ParameterSpace parameterSpace) {
                this.parameterSpace = parameterSpace;
                return this;
            }

            public CandidateIterator build(Instances instances, Random random) {
                if(parameterSpaceGetter != null) {
                    if(parameterSpace != null) {
                        throw new IllegalStateException("both parameter space and getter set, which should be used?!");
                    }
                    parameterSpace = parameterSpaceGetter.apply(instances);
                }
                ParameterSetIterator iterator = buildParameterSetIterator(random);
                return new CandidateIterator(supplier, config, name, parameterSetCountThreshold, parameterSetCountLimit, iterator);
            }

            public Builder setParameterSetCountThreshold(int parameterSetCountThreshold) {
                this.parameterSetCountThreshold = parameterSetCountThreshold;
                return this;
            }

            public Builder setParameterSetSearchStrategy(ParameterSetSearchStrategy parameterSetSearchStrategy) {
                this.parameterSetSearchStrategy = parameterSetSearchStrategy;
                return this;
            }

            public ParameterSetIterator buildParameterSetIterator(Random random) {
                List<Integer> values = ArrayUtilities.sequence(parameterSpace.size());
                switch (parameterSetSearchStrategy) {
                    case RANDOM:
                        return new ParameterSetIterator(parameterSpace, new RandomIterator<>(random, values));
//            case SPREAD: return new ParameterSetIterator(parameterSpace, new SpreadIterator<>(values));
                    case LINEAR: return new ParameterSetIterator(parameterSpace, new LinearIterator<>(values));
                    default: throw new IllegalStateException(parameterSetSearchStrategy.name() + " not implemented yet");
                }
            }

            public ParameterSpace getParameterSpace() {
                return parameterSpace;
            }

            public Function<Instances, ParameterSpace> getParameterSpaceGetter() {
                return parameterSpaceGetter;
            }

            public int getParameterSetCountThreshold() {
                return parameterSetCountThreshold;
            }

            public ParameterSetSearchStrategy getParameterSetSearchStrategy() {
                return parameterSetSearchStrategy;
            }

            public String getName() {
                return name;
            }

            public int getParameterSetCountLimit() {
                return parameterSetCountLimit;
            }

            public Builder setParameterSetCountLimit(final int parameterSetCountLimit) {
                this.parameterSetCountLimit = parameterSetCountLimit;
                return this;
            }

            public TemplateConfig getConfig() {
                return config;
            }

        }
    }


    private boolean hasRemainingCandidateParameterSets() {
        return candidateIteratorIterator.hasNext();
    }

    private void evalNextCandidate() throws
                                     Exception {
        CandidateIterator candidateIterator = candidateIteratorIterator.next();
        AbstractClassifier candidate = candidateIterator.next();
        candidateIterator.remove();
        if(!candidateIterator.hasNext()) {
            candidateIteratorIterator.remove();
        }
        if(candidate instanceof Knn) { // todo build into candidate iterator? probs best
            ((Knn) candidate).getConfig().setPredefinedTrainEstimateSet(trainEstimateSet);
            ((Knn) candidate).getConfig().setPredefinedTrainEstimateSet(trainNeighbourhood);
        } else {
            throw new UnsupportedOperationException();
        }
        if(candidate instanceof ContractClassifier) {
            ((Knn) candidate).setTimeLimit(remainingTrainContractNanos());
        }
        candidate.buildClassifier(trainSet);
        ClassifierResults trainResults;
        if(candidate instanceof TemplateClassifierInterface) {
            trainResults = ((Knn) candidate).getTrainResults();
        } else {
            throw new UnsupportedOperationException();
        }
        boolean improvement = candidateIterator.feedback(config.getTrainResultsMetricGetter().apply(trainResults));
        if(improvement) {
            // todo record best param
            // todo dump all this in the tuners for individ dms
        }
        throw new UnsupportedOperationException();
    }

    private List<Instance> buildTrainNeighbourhood(Collection<Instance> trainSet) {
        DynamicIterator<Instance, ?> iterator = config.getKnnConfiguration()
                                                      .buildNeighbourSearchStrategy(trainSet, getTrainRandom());
        List<Instance> trainNeighbourhood = new ArrayList<>();
        ArrayUtilities.addAllAndRemove(trainNeighbourhood, iterator);
        return trainNeighbourhood;
    }

    private List<Instance> buildTrainEstimateSet(Collection<Instance> trainSet) {
        DynamicIterator<Instance, ?> iterator = config.getKnnConfiguration()
                                                       .buildTrainEstimationStrategy(trainSet, getTrainRandom());
        List<Instance> trainEstimateSet = new ArrayList<>();
        if(config.getKnnConfiguration().getTrainEstimationSource().equals(TrainEstimationSource.FROM_TRAIN_NEIGHBOURHOOD)) {
            trainEstimateSet.addAll(trainNeighbourhood);
        } else {
            ArrayUtilities.addAllAndRemove(trainEstimateSet, iterator);
        }
        return trainEstimateSet;
    }

    @Override
    public void setOption(final String key, final String value) throws
                                                                Exception {

    }

    @Override
    public void buildClassifier(final Instances trainSet) throws
                                                                Exception {
        setup(trainSet);
        while (hasRemainingCandidateParameterSets() && withinTrainContract()) {
            evalNextCandidate();
            getTrainStopWatch().lap();
        }
    }

    public boolean evaluateCandidate(AbstractClassifier classifier) {
        if(classifier instanceof Knn) {
            ((Knn) classifier).getConfig().setPredefinedTrainNeighbourhood(trainNeighbourhood);
            ((Knn) classifier).getConfig().setPredefinedTrainEstimateSet(trainEstimateSet);
        }
        throw new UnsupportedOperationException();
    }

    private static class TrainedCandidate {
        private final ClassifierResults trainResults;
        private final Classifier classifier;

        private TrainedCandidate(final ClassifierResults trainResults,
                                 final Classifier classifier) {
            this.trainResults = trainResults;
            this.classifier = classifier;
        }

        public Classifier getClassifier() {
            return classifier;
        }

        public ClassifierResults getTrainResults() {
            return trainResults;
        }
    }

    //    private final static String NUM_PARAMETER_SETS_KEY = "p";
//    private final static String NEIGHBOURHOOD_SIZE_KEY = "n";
//    private final static String NUM_PARAMETER_SETS_PERCENTAGE_KEY = "pp";
//    private final static String NEIGHBOURHOOD_SIZE_PERCENTAGE_KEY = "np";
//
//    private final List<CandidateParameterSpaceIteratorBuilder> candidateParameterSpaceIteratorBuilders = new ArrayList<>();
//    private final List<CandidateParameterSpaceIterator> iterators = new ArrayList<>();
//
//    private final List<Function<Instances, ParameterSpace>> parameterSpaceGetters = new ArrayList<>();
//
//    private final List<ParameterSpace> parameterSpaces = new ArrayList<>();
//    private final List<Candidate> candidates = new ArrayList<>();
//    private final List<Candidate> constituents = new ArrayList<>();
//    private boolean removeDuplicateParameterSets = true;
//    private Selector<Candidate> candidateSelector = new BestPerTypeSelector<>(Candidate::getParameterSpace, (candidate, other) -> {
//        int comparison = Integer.compare(candidate.getKnn().getTrainNeighbourhoodSizeLimit(), other.getKnn().getTrainNeighbourhoodSizeLimit());
//        if (comparison != 0) {
//            return comparison;
//        }
//        comparison = Integer.compare(candidate.getKnn().getTrainNeighbourhoodSizeLimit(), other.getKnn().getTrainNeighbourhoodSizeLimit());
//        if (comparison <= 0) {
//            comparison = Comparator.comparingDouble(ClassifierResults::getAcc).compare(candidate.getKnn().getTrainResults(), other.getKnn().getTrainResults());
//        }
//        return comparison;
//    });
//    private ParameterSpacesIterationStrategy parameterSpaceIterationStrategy = ParameterSpacesIterationStrategy.RANDOM;
//    private ParameterSetSearchStrategy parameterSetSearchStrategy = ParameterSetSearchStrategy.RANDOM; // need to define this per param space
//    private Knn.NeighbourSearchStrategy neighbourSearchStrategy = Knn.NeighbourSearchStrategy.RANDOM; // need to define this per param space
//    public final static int DEFAULT_NUM_PARAMETER_SETS = -1;
//    private int numParametersLimit = DEFAULT_NUM_PARAMETER_SETS;
//    private int parameterCount = 0;
//    public final static int DEFAULT_NEIGHBOURHOOD_SIZE = -1;
//    private int trainNeighbourhoodSizeLimit = DEFAULT_NEIGHBOURHOOD_SIZE;
//    public final static double DEFAULT_NUM_PARAMETER_SETS_PERCENTAGE = -1;
//    private double numParametersLimitPercentage = DEFAULT_NUM_PARAMETER_SETS_PERCENTAGE;
//    public final static double DEFAULT_NEIGHBOURHOOD_SIZE_PERCENTAGE = -1;
//    private double neighbourhoodSizeLimitPercentage = DEFAULT_NEIGHBOURHOOD_SIZE_PERCENTAGE;
//    private Iterator<IterableParameterSpace> parameterSpaceIterator;
//    public final static int DEFAULT_TRAIN_SUB_SET_SIZE = -1;
//    public final static int DEFAULT_TRAIN_SUB_SET_SIZE_PERCENTAGE = -1;
//    private final List<Instance> trainNeighbourhood = new ArrayList<>();
//    private boolean progressive = false;
//    private int parameterIterationThreshold = 15;
//    private int neighbourIterationThreshold = 10;
//    private Map<ParameterSpace, Integer> parameterIterationCounts = null;
//
//    public int getTrainEstimateSetSize() {
//        return trainEstimateSetSize;
//    }
//
//    public void setTrainEstimateSetSize(final int trainEstimateSetSize) {
//        this.trainEstimateSetSize = trainEstimateSetSize;
//    }
//
//    private int trainEstimateSetSize = DEFAULT_TRAIN_SUB_SET_SIZE;
//    private double trainEstimateSetSizePercentage = DEFAULT_TRAIN_SUB_SET_SIZE_PERCENTAGE;
//    private Instances trainSet;
//
//    public ElasticEnsemble() {
//        this(getClassicParameterSpaceGetters());
//    }
//
//    @Override
//    public void setOption(String key, String value) throws Exception {
//        switch (key) {
//            case NUM_PARAMETER_SETS_KEY:
//                setNumParametersLimit(Integer.parseInt(value));
//                break;
//            case NEIGHBOURHOOD_SIZE_KEY:
//                setTrainNeighbourhoodSizeLimit(Integer.parseInt(value));
//                break;
//            case NEIGHBOURHOOD_SIZE_PERCENTAGE_KEY:
//                setNeighbourhoodSizeLimitPercentage(Double.parseDouble(value));
//                break;
//            case NUM_PARAMETER_SETS_PERCENTAGE_KEY:
//                setNumParametersLimitPercentage(Double.parseDouble(value));
//                break;
//        }
//    }
//
//    public ElasticEnsemble(Function<Instances, ParameterSpace>... parameterSpaceGetters) {
//        this(Arrays.asList(parameterSpaceGetters));
//    }
//
//    public ElasticEnsemble(List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
//        setParameterSpaceGetters(parameterSpaceGetters);
//    }
//
//    public static List<Function<Instances, ParameterSpace>> getClassicParameterSpaceGetters() {
//        return new ArrayList<>(Arrays.asList(
//                instances -> Dtw.edParameterSpace(),
//                instances -> Dtw.fullWarpParameterSpace(),
//                Dtw::warpParameterSpace,
//                instances -> CachedDdtw.fullWarpParameterSpace(),
//                CachedDdtw::warpParameterSpace,
//                instances -> Wdtw.parameterSpace(),
//                instances -> CachedWddtw.parameterSpace(),
//                Lcss::parameterSpace,
//                Erp::parameterSpace,
//                instances -> Msm.parameterSpace(),
//                instances -> Twe.parameterSpace()
//        ));
//    }
//
//    public static List<Function<Instances, ParameterSpace>> getDefaultParameterSpaceGetters() {
//        return new ArrayList<>(Arrays.asList(
//                Dtw::allWarpParameterSpace,
//                CachedDdtw::allWarpParameterSpace,
//                instances -> Wdtw.parameterSpace(),
//                instances -> CachedWddtw.parameterSpace(),
//                Lcss::parameterSpace,
//                Erp::parameterSpace,
//                instances -> Msm.parameterSpace(),
//                instances -> Twe.parameterSpace()
//        ));
//    }
//
//    public static List<ParameterSpace> getParameterSpaces(Instances instances, List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
//        List<ParameterSpace> parameterSpaces = new ArrayList<>();
//        for (Function<Instances, ParameterSpace> getter : parameterSpaceGetters) {
//            ParameterSpace parameterSpace = getter.apply(instances);
//            parameterSpaces.add(parameterSpace);
//        }
//        return parameterSpaces;
//    }
//
//    @Override
//    public String toString() {
//        return "ee";
//    }
//
//    public boolean isRemoveDuplicateParameterSets() {
//        return removeDuplicateParameterSets;
//    }
//
//    public void setRemoveDuplicateParameterSets(final boolean removeDuplicateParameterSets) {
//        this.removeDuplicateParameterSets = removeDuplicateParameterSets;
//    }
//
//    public int getTrainNeighbourhoodSizeLimit() {
//        return trainNeighbourhoodSizeLimit;
//    }
//
//    public void setTrainNeighbourhoodSizeLimit(final int trainNeighbourhoodSizeLimit) {
//        this.trainNeighbourhoodSizeLimit = trainNeighbourhoodSizeLimit;
//    }
//
//    public int getNumParametersLimit() {
//        return numParametersLimit;
//    }
//
//    public void setNumParametersLimit(final int numParametersLimit) {
//        this.numParametersLimit = numParametersLimit;
//    }
//
//    public List<Function<Instances, ParameterSpace>> getParameterSpaceGetters() {
//        return parameterSpaceGetters;
//    }
//
//    public void setParameterSpaceGetters(List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
//        this.parameterSpaceGetters.clear();
//        this.parameterSpaceGetters.addAll(parameterSpaceGetters);
//    }
//
//    public Selector<Candidate> getCandidateSelector() {
//        return candidateSelector;
//    }
//
//    public void setCandidateSelector(final Selector<Candidate> candidateSelector) {
//        this.candidateSelector = candidateSelector;
//    }
//
//    public ParameterSpacesIterationStrategy getParameterSpaceIterationStrategy() {
//        return parameterSpaceIterationStrategy;
//    }
//
//    public void setParameterSpaceIterationStrategy(final ParameterSpacesIterationStrategy parameterSpaceIterationStrategy) {
//        this.parameterSpaceIterationStrategy = parameterSpaceIterationStrategy;
//    }
//
//    public ParameterSetSearchStrategy getParameterSetSearchStrategy() {
//        return parameterSetSearchStrategy;
//    }
//
//    public void setParameterSetSearchStrategy(final ParameterSetSearchStrategy parameterSetSearchStrategy) {
//        this.parameterSetSearchStrategy = parameterSetSearchStrategy;
//    }
//
//    public Knn.NeighbourSearchStrategy getNeighbourSearchStrategy() {
//        return neighbourSearchStrategy;
//    }
//
//    public void setNeighbourSearchStrategy(final Knn.NeighbourSearchStrategy neighbourSearchStrategy) {
//        this.neighbourSearchStrategy = neighbourSearchStrategy;
//    }
//
//    @Override
//    public String[] getOptions() { // todo update
//        return ArrayUtilities.concat(super.getOptions(), new String[]{
//                NUM_PARAMETER_SETS_KEY,
//                String.valueOf(getNumParametersLimit()),
//                NEIGHBOURHOOD_SIZE_KEY,
//                String.valueOf(getTrainNeighbourhoodSizeLimit()),
//                NUM_PARAMETER_SETS_PERCENTAGE_KEY,
//                String.valueOf(getNumParametersLimitPercentage()),
//                NEIGHBOURHOOD_SIZE_PERCENTAGE_KEY,
//                String.valueOf(getNeighbourhoodSizeLimitPercentage())
//        });
//    }
//
//    private void setupNeighbourhoodSize() {
//        if (neighbourhoodSizeLimitPercentage >= 0) {
//            setTrainNeighbourhoodSizeLimit((int) (neighbourhoodSizeLimitPercentage * trainSet.size()));
//        }
//    }
//
//    private void setupNumParameterSets() {
//        if (numParametersLimitPercentage >= 0) {
//            int size = 0;
//            for (ParameterSpace parameterSpace : parameterSpaces) {
//                size += parameterSpace.size();
//            }
//            numParametersLimit = (int) (numParametersLimitPercentage * size);
//        }
//    }
//
//    public double getNumParametersLimitPercentage() {
//        return numParametersLimitPercentage;
//    }
//
//    public void setNumParametersLimitPercentage(final double numParametersLimitPercentage) {
//        this.numParametersLimitPercentage = numParametersLimitPercentage;
//    }
//
//    public double getNeighbourhoodSizeLimitPercentage() {
//        return neighbourhoodSizeLimitPercentage;
//    }
//
//    public void setNeighbourhoodSizeLimitPercentage(final double neighbourhoodSizeLimitPercentage) {
//        this.neighbourhoodSizeLimitPercentage = neighbourhoodSizeLimitPercentage;
//    }
//
//    private Iterator<IterableParameterSpace> getParameterSpacesIterator(List<IterableParameterSpace> iterableParameterSpaces) {
//        switch (parameterSpaceIterationStrategy) {
//            case RANDOM: return parameterSpaceIterator = new RandomIterator<>(iterableParameterSpaces, getTrainRandom().nextLong());
//            case ROUND_ROBIN: return parameterSpaceIterator = new RoundRobinIterator<>(iterableParameterSpaces);
//            default: throw new IllegalStateException(parameterSpaceIterationStrategy.name() + " not implemented");
//        }
//    }
//
//    private ParameterSetIterator getParameterSetIterator(ParameterSpace parameterSpace) {
//        ArrayList<Integer> values =
//                new ArrayList<>(Arrays.asList(ArrayUtilities.box(ArrayUtilities.range(parameterSpace.size() - 1))));
//        switch (parameterSetSearchStrategy) {
//            case RANDOM: return new ParameterSetIterator(parameterSpace, new RandomIterator<>(values, getTrainRandom().nextLong()));
////            case SPREAD: return new ParameterSetIterator(parameterSpace, new SpreadIterator<>(values));
//            case LINEAR: return new ParameterSetIterator(parameterSpace, new LinearIterator<>(values));
//            default: throw new IllegalStateException(parameterSetSearchStrategy.name() + " not implemented yet");
//        }
//    }
//
//    private boolean hasRemainingParameterSets() {
//        return (parameterCount < numParametersLimit || numParametersLimit < 0) && parameterSpaceIterator.hasNext();
//    }
//
//    private void setupTrainEstimateSetSize() {
//        if(trainEstimateSetSizePercentage >= 0) {
//            setTrainEstimateSetSize((int) (trainSet.size() * trainEstimateSetSizePercentage));
//        }
//    }
//
//
//
//
//    private boolean evaluateCandidate(Candidate candidate) throws Exception {
//        Knn knn = candidate.getKnn();
//        knn.setTrainContractNanos(remainingTrainContractNanos());
//        knn.buildClassifier(trainSet);
//        Candidate trainedCandidate = candidate;//new Candidate(candidate);
//        boolean improvement = candidateSelector.add(trainedCandidate);
//        return improvement;
//    }
//
//    @Override
//    public void buildClassifier(final Instances trainSet) throws
//            Exception {
//        setup(trainSet);
//        // todo fully trained candidates for setter changes post build
//        getTrainStopWatch().lap();
//        if (getTrainNeighbourhoodSizeLimit() != 0) {
////            int count = 0;
//            boolean remainingParameters = hasRemainingParameterSets();
//            boolean remainingCandidates = !candidates.isEmpty();
//            while ((remainingParameters || remainingCandidates)) {
////                System.out.println(count++);
//                Knn knn;
//                Candidate candidate;
//                int knnIndex;
////                boolean choice = true;
////                if (remainingParameters && remainingCandidates) {
////                    choice = random.nextBoolean();
////                } else if (remainingCandidates) {
////                    choice = false;
////                }
////                if (choice) {
//                if (remainingParameters) {
//                    IterableParameterSpace iterableParameterSpace = parameterSpaceIterator.next();
//                    Iterator<String[]> parameterSetIterator = iterableParameterSpace.getIterator();
//                    ParameterSpace parameterSpace = iterableParameterSpace.getParameterSpace();
//                    String[] parameters = parameterSetIterator.next();
//                    parameterSetIterator.remove();
//                    int parameterIterationCount = parameterIterationCounts.get(parameterSpace);
//                    parameterIterationCounts.put(parameterSpace, parameterIterationCount + 1);
//                    if (!parameterSetIterator.hasNext() || parameterIterationCount > parameterIterationThreshold) {
//                        parameterSpaceIterator.remove();
//                    } // todo random guess if no params or constituents
//                    knn = new Knn();
//                    knn.setOptions(parameters);
////                    if(progressive) {
////                        knn.setTrainNeighbourhoodSizeLimit(1);
////                    } else {
//                        knn.setTrainNeighbourhoodSizeLimit(trainNeighbourhoodSizeLimit);
////                    }
//                    knn.setEarlyAbandon(true);
//                    knn.setTrainNeighbourSearchStrategy(neighbourSearchStrategy);
//                    knn.setSeed(getTrainRandom().nextInt());
//                    knn.setPredefinedTrainNeighbourhood(trainNeighbourhood);
//                    knn.setTrainEstimateSetSizeLimit(trainEstimateSetSize); // todo make knn adapt when train set size changes / make enum strategy
//                    candidate = new Candidate(knn, parameterSpace);
//                    candidates.add(candidate);
//                    knnIndex = candidates.size() - 1;
//                    parameterCount++;
//                    boolean improvement = evaluateCandidate(candidate);
//                    if(improvement) {
//                        parameterIterationCounts.put(parameterSpace, 0);
//                    }
//                } else {
//                    knnIndex = getTrainRandom().nextInt(candidates.size());
//                    candidate = candidates.get(knnIndex);
//                    knn = candidate.getKnn();
//                    int sampleSize = knn.getTrainNeighbourhoodSizeLimit() + 1;
//                    knn.setTrainNeighbourhoodSizeLimit(sampleSize);
//                    evaluateCandidate(candidate);
//                }
//                if ((knn.getTrainNeighbourhoodSizeLimit() + 1 > getTrainNeighbourhoodSizeLimit()
//                        && getTrainNeighbourhoodSizeLimit() >= 0)
//                        || knn.getTrainNeighbourhoodSizeLimit() + 1 > trainSet.size()) {
////                    fullyTrainedCandidates.add(
//                            candidates.remove(knnIndex);
////                    );
//                }
//                remainingParameters = hasRemainingParameterSets();
//                remainingCandidates = !candidates.isEmpty();
//                getTrainStopWatch().lap();
//            }
//        }
//        constituents.clear();
//        constituents.addAll(candidateSelector.getSelected());
//        String savePath = getSavePath();
//        if (savePath != null) {
//            Utilities.mkdir(savePath);
//        }
//        if (savePath != null) {
//            for (int i = 0; i < constituents.size(); i++) {
//                Knn knn = constituents.get(i).getKnn();
//                ClassifierResults results = knn.getTrainResults();
//                results.writeFullResultsToFile(savePath + "/train" + i + ".csv");
//            }
//        }
//        ClassifierResults trainResults = new ClassifierResults();
//        setTrainResults(trainResults);
//        for (int i = 0; i < trainSet.size(); i++) {
//            long predictionTime = System.nanoTime();
//            double[] distribution = new double[trainSet.numClasses()];
//            for(Candidate constituent : constituents) {
//                Knn knn = constituent.getKnn();
//                double[] candidateDistribution;
////                if(samplesTrainSet()) {
//                    candidateDistribution = knn.distributionForInstance(trainSet.get(i));
////                } else { todo use pre-computed train results if not sampling train set in knns
////                    ClassifierResults constituentTrainResults = knn.getTrainResults();
////                    candidateDistribution = constituentTrainResults.getProbabilityDistribution(i);
////                    predictionTime -= constituentTrainResults.getPredictionTimeInNanos(i);
////                }
//                ArrayUtilities.multiply(candidateDistribution, knn.getTrainResults().getAcc());
//                ArrayUtilities.normaliseInplace(candidateDistribution);
//                ArrayUtilities.add(distribution, candidateDistribution);
//            }
//            ArrayUtilities.normaliseInplace(distribution);
//            predictionTime = System.nanoTime() - predictionTime;
//            trainResults.addPrediction(trainSet.get(i).classValue(), distribution, Utilities.argMax(distribution, getTrainRandom()), predictionTime, null);
//        }
//        getTrainStopWatch().lap();
//        setClassifierResultsMetaInfo(trainResults);
//    }
//
//    private boolean samplesTrainSet() {
//        return trainEstimateSetSize >= 0 && trainEstimateSetSize < trainSet.size();
//    }
//
//    @Override
//    public double[] distributionForInstance(final Instance testInstance) throws
//            Exception {
//        double[] distribution = new double[testInstance.numClasses()];
//        for(Candidate constituent : constituents) {
//            Knn knn = constituent.getKnn();
//            double[] candidateDistribution;
//            candidateDistribution = knn.distributionForInstance(testInstance);
//            ArrayUtilities.multiply(candidateDistribution, knn.getTrainResults().getAcc());
//            ArrayUtilities.add(distribution, candidateDistribution);
//        }
//        ArrayUtilities.normaliseInplace(distribution);
//        return distribution;
//    }
//
//    public double getTrainEstimateSetSizePercentage() {
//        return trainEstimateSetSizePercentage;
//    }
//
//    public void setTrainEstimateSetSizePercentage(final double trainEstimateSetSizePercentage) {
//        this.trainEstimateSetSizePercentage = trainEstimateSetSizePercentage;
//    }
//

//

//
//
//
//    private static class IterableParameterSpace {
//        private final ParameterSpace parameterSpace;
//        private final Iterator<String[]> iterator;
//
//        private IterableParameterSpace(final ParameterSpace parameterSpace,
//                                       final Iterator<String[]> iterator) {
//            this.parameterSpace = parameterSpace;
//            this.iterator = iterator;
//        }
//
//        public Iterator<String[]> getIterator() {
//            return iterator;
//        }
//
//        public ParameterSpace getParameterSpace() {
//            return parameterSpace;
//        }
//    }
//
//    private class Candidate {
//        private final Knn knn;
//        private final ParameterSpace parameterSpace;
//
//        private Candidate(final Knn knn, final ParameterSpace parameterSpace) {
//            this.knn = knn;
//            this.parameterSpace = parameterSpace;
//        }
//
//        private Candidate(Candidate candidate) {
//            this(candidate.knn.copy(), candidate.parameterSpace);
//        }
//
//        public Knn getKnn() {
//            return knn;
//        }
//
//        public ParameterSpace getParameterSpace() {
//            return parameterSpace;
//        }
//
//    }
}
