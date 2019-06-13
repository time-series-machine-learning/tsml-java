package classifiers.distance_based.elastic_ensemble;

import classifiers.distance_based.elastic_ensemble.iteration.LinearIterator;
import classifiers.distance_based.elastic_ensemble.iteration.RandomIterator;
import classifiers.distance_based.elastic_ensemble.iteration.RoundRobinIterator;
import classifiers.distance_based.elastic_ensemble.iteration.SpreadIterator;
import classifiers.distance_based.elastic_ensemble.selection.BestPerTypeSelector;
import classifiers.distance_based.elastic_ensemble.selection.Selector;
import classifiers.distance_based.knn.Knn;
import classifiers.template_classifier.TemplateClassifier;
import distances.derivative_time_domain.ddtw.CachedDdtw;
import distances.time_domain.dtw.Dtw;
import distances.derivative_time_domain.wddtw.CachedWddtw;
import distances.time_domain.erp.Erp;
import distances.time_domain.lcss.Lcss;
import distances.time_domain.msm.Msm;
import distances.time_domain.twe.Twe;
import distances.time_domain.wdtw.Wdtw;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import timeseriesweka.classifiers.ensembles.voting.MajorityVoteByConfidence;
import timeseriesweka.classifiers.ensembles.voting.ModuleVotingScheme;
import timeseriesweka.classifiers.ensembles.weightings.ModuleWeightingScheme;
import timeseriesweka.classifiers.ensembles.weightings.TrainAcc;
import utilities.ArrayUtilities;
import utilities.StringUtilities;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.function.Function;

import static utilities.ArrayUtilities.argMax;

public class ElasticEnsemble extends TemplateClassifier {

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

    public ElasticEnsemble() {
        this(getClassicParameterSpaceGetters());
    }

    public ElasticEnsemble(Function<Instances, ParameterSpace>... parameterSpaceGetters) {
        this(Arrays.asList(parameterSpaceGetters));
    }

    private final List<Function<Instances, ParameterSpace>> parameterSpaceGetters = new ArrayList<>();
    private final List<ParameterSpace> parameterSpaces = new ArrayList<>();

    public ElasticEnsemble(List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        setParameterSpaceGetters(parameterSpaceGetters);
    }

    public void setParameterSpaceGetters(List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        this.parameterSpaceGetters.clear();
        this.parameterSpaceGetters.addAll(parameterSpaceGetters);
    }

    public static List<ParameterSpace> getParameterSpaces(Instances instances, List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        List<ParameterSpace> parameterSpaces = new ArrayList<>();
        for(Function<Instances, ParameterSpace> getter : parameterSpaceGetters) {
            ParameterSpace parameterSpace = getter.apply(instances);
            parameterSpaces.add(parameterSpace);
        }
        return parameterSpaces;
    }

    @Override
    public String toString() {
        return "ee";
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

    public boolean isRemoveDuplicateParameterSets() {
        return removeDuplicateParameterSets;
    }

    public void setRemoveDuplicateParameterSets(final boolean removeDuplicateParameterSets) {
        this.removeDuplicateParameterSets = removeDuplicateParameterSets;
    }

    private boolean removeDuplicateParameterSets = true;
    private EnsembleModule[] modules = null;
    private ModuleWeightingScheme weightingScheme = new TrainAcc();
    private ModuleVotingScheme votingScheme = new MajorityVoteByConfidence();
    private long phaseTime = 0;
    private final List<Candidate> candidates = new ArrayList<>();
    private Selector<TrainedCandidate> selector = new BestPerTypeSelector<>(Candidate::getParameterSpace, (candidate, other) -> {
        int comparison = Integer.compare(candidate.getKnn().getNeighbourhoodSize(), other.getKnn().getNeighbourhoodSize());
        if(comparison != 0) {
            return comparison;
        }
        comparison = Integer.compare(candidate.getKnn().getNeighbourhoodSize(), other.getKnn().getNeighbourhoodSize());
        if(comparison <= 0) {
            comparison = Comparator.comparingDouble(ClassifierResults::getAcc).compare(candidate.getTrainResults(), other.getTrainResults());
        }
        return comparison;
    });
    private final List<Candidate> fullyTrainedCandidates = new ArrayList<>();

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

    private ParameterSpacesIterationStrategy parameterSpacesIterationStrategy = ParameterSpacesIterationStrategy.ROUND_ROBIN;

    public List<Function<Instances, ParameterSpace>> getParameterSpaceGetters() {
        return parameterSpaceGetters;
    }

    public ModuleWeightingScheme getWeightingScheme() {
        return weightingScheme;
    }

    public void setWeightingScheme(final ModuleWeightingScheme weightingScheme) {
        this.weightingScheme = weightingScheme;
    }

    public ModuleVotingScheme getVotingScheme() {
        return votingScheme;
    }

    public void setVotingScheme(final ModuleVotingScheme votingScheme) {
        this.votingScheme = votingScheme;
    }

    public Selector<TrainedCandidate> getSelector() {
        return selector;
    }

    public void setSelector(final Selector<TrainedCandidate> selector) {
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

    private DistanceMeasureSearchStrategy distanceMeasureSearchStrategy = DistanceMeasureSearchStrategy.RANDOM;
    private Knn.NeighbourSearchStrategy neighbourSearchStrategy = Knn.NeighbourSearchStrategy.RANDOM;
    private int numParameterSets = -1;
    private int parameterSetCount = 0;
    private int neighbourhoodSize = -1;
    private final static String NUM_PARAMETER_SETS_KEY = "numParameterSets";
    private final static String NEIGHBOURHOOD_SIZE_KEY = "neighbourhoodSize";
    private final static String NUM_PARAMETER_SETS_PERCENTAGE_KEY = "numParameterSetsPercentage";
    private final static String NEIGHBOURHOOD_SIZE_PERCENTAGE_KEY = "neighbourhoodSizePercentage";
    private final static String CONSTITUENT_PARAMETERS_KEY = "constituentParametersTag";

    @Override
    public String[] getOptions() {
        List<String> constituentParameters = new ArrayList<>();
        for(TrainedCandidate constituent : constituents) {
            constituentParameters.add(CONSTITUENT_PARAMETERS_KEY);
            constituentParameters.add(String.valueOf(constituent.getTrainResults().getAcc()));
            constituentParameters.add(StringUtilities.join(",", constituent.getKnn().getOptions()));
        }
        constituentParameters.add(CONSTITUENT_PARAMETERS_KEY);
        return ArrayUtilities.concat(super.getOptions(), new String[] {
            NUM_PARAMETER_SETS_KEY,
            String.valueOf(getNumParameterSets()),
            NEIGHBOURHOOD_SIZE_KEY,
            String.valueOf(getNeighbourhoodSize())
        }, constituentParameters.toArray(new String[0]));
    }

    @Override
    public void setOptions(final String[] options) throws
                                                   Exception {
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

    private void setupNeighbourhoodSize(Instances trainInstances) {
        if(neighbourhoodSizePercentage >= 0) {
            setNeighbourhoodSize((int) (neighbourhoodSizePercentage * trainInstances.size()));
        }
    }

    private void setupNumParameterSets() {
        if(numParameterSetsPercentage >= 0) {
            int size = 0;
            for(ParameterSpace parameterSpace : parameterSpaces) {
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

    private double numParameterSetsPercentage = -1;

    public double getNeighbourhoodSizePercentage() {
        return neighbourhoodSizePercentage;
    }

    public void setNeighbourhoodSizePercentage(final double neighbourhoodSizePercentage) {
        this.neighbourhoodSizePercentage = neighbourhoodSizePercentage;
    }

    private double neighbourhoodSizePercentage = -1;
    private final List<TrainedCandidate> constituents = new ArrayList<>();

    private boolean limitedNumParameterSets() {
        return numParameterSets >= 0;
    }

    private boolean withinNumParameterSets() {
        return parameterSetCount < numParameterSets;
    }

    private boolean remainingParameterSets() {
        return parameterSpacesIterator.hasNext() && (!limitedNumParameterSets() || withinNumParameterSets());
    }

    private Iterator<IterableParameterSpace> parameterSpacesIterator;

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

    private Iterator<IterableParameterSpace> getParameterSpacesIterator(List<IterableParameterSpace> iterableParameterSpaces) {
        switch (parameterSpacesIterationStrategy) {
            case RANDOM:
                return parameterSpacesIterator = new RandomIterator<>(iterableParameterSpaces, getTrainRandom());
            case ROUND_ROBIN:
                return parameterSpacesIterator = new RoundRobinIterator<>(iterableParameterSpaces);
            default:
                throw new IllegalStateException(parameterSpacesIterationStrategy.name() + " not implemented");
        }
    }

    private ParameterSetIterator getParameterSetIterator(ParameterSpace parameterSpace) {
        ArrayList<Integer> values =
            new ArrayList<>(Arrays.asList(ArrayUtilities.box(ArrayUtilities.range(parameterSpace.size() - 1))));
        switch (distanceMeasureSearchStrategy) {
            case RANDOM:
                return new ParameterSetIterator(parameterSpace, new RandomIterator<>(values,
                    getTrainRandom()));
            case SPREAD:
                return new ParameterSetIterator(parameterSpace, new SpreadIterator<>(
                    values));
            case LINEAR:
                return new ParameterSetIterator(parameterSpace, new LinearIterator<>(values));
            default:
                throw new IllegalStateException(distanceMeasureSearchStrategy.name() + " not implemented yet");
        }
    }

    @Override
    public void buildClassifier(final Instances trainInstances) throws
                                                      Exception {
        Random random = getTrainRandom();
        if(trainSetChanged(trainInstances)) {
            getTrainStopWatch().reset();
            candidates.clear();
            selector.setRandom(random);
            fullyTrainedCandidates.clear();
            parameterSpaces.clear();
            parameterSetCount = 0;
            parameterSpaces.addAll(getParameterSpaces(trainInstances, parameterSpaceGetters));
            if(removeDuplicateParameterSets) {
                for(ParameterSpace parameterSpace : parameterSpaces) {
                    parameterSpace.removeDuplicateValues();
                }
            }
            List<IterableParameterSpace> iterableParameterSpaces = new ArrayList<>();
            for(ParameterSpace parameterSpace : parameterSpaces) {
                Iterator<String[]> iterator = getParameterSetIterator(parameterSpace);
                if(iterator.hasNext()) {
                    iterableParameterSpaces.add(new IterableParameterSpace(parameterSpace, iterator));
                }
            }
            parameterSpacesIterator = getParameterSpacesIterator(iterableParameterSpaces);
            setupNeighbourhoodSize(trainInstances);
            setupNumParameterSets();
            getTrainStopWatch().lap();
        }
        if(!fullyTrainedCandidates.isEmpty() && neighbourhoodSize > fullyTrainedCandidates.get(0).getKnn().getNeighbourhoodSize()) {
            candidates.addAll(fullyTrainedCandidates);
            fullyTrainedCandidates.clear();
        }
        boolean remainingParameters = remainingParameterSets();
        boolean remainingCandidates = !candidates.isEmpty();
        getTrainStopWatch().lap();
        int count = 0;
        if(getNeighbourhoodSize() != 0) {
            while((remainingParameters || remainingCandidates) && remainingTrainContractNanos() > phaseTime) {
                System.out.println(count++);
                long startTime = System.nanoTime();
                Knn knn;
                Candidate candidate;
                boolean choice = true;
                if(remainingParameters && remainingCandidates) {
                    choice = random.nextBoolean();
                } else if(remainingCandidates) {
                    choice = false;
                }
                int knnIndex;
                if(choice) {
                    IterableParameterSpace iterableParameterSpace = parameterSpacesIterator.next();
                    Iterator<String[]> parameterSetIterator = iterableParameterSpace.getIterator();
                    ParameterSpace parameterSpace = iterableParameterSpace.getParameterSpace();
                    String[] parameters = parameterSetIterator.next();
                    parameterSetIterator.remove();
                    if(!parameterSetIterator.hasNext()) {
                        parameterSpacesIterator.remove();
                    } // todo random guess if no params or constituents
                    knn = new Knn();
                    knn.setOptions(parameters);
                    knn.setNeighbourhoodSize(1);
                    knn.setEarlyAbandon(true);
                    knn.setNeighbourSearchStrategy(neighbourSearchStrategy);
                    knn.setSeed(random.nextInt());
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
                if((knn.getNeighbourhoodSize() + 1 > getNeighbourhoodSize() && getNeighbourhoodSize() >= 0) || knn.getNeighbourhoodSize() + 1 > trainInstances.size()) {
                    fullyTrainedCandidates.add(candidates.remove(knnIndex));
                }
                knn.setTrainContractNanos(remainingTrainContractNanos());
                knn.buildClassifier(trainInstances);
                ClassifierResults trainResults = knn.getTrainResults();
                TrainedCandidate trainedCandidate = new TrainedCandidate(candidate, trainResults);
                selector.add(trainedCandidate);
                phaseTime = Long.max(System.nanoTime() - startTime, phaseTime);
                remainingParameters = remainingParameterSets();
                remainingCandidates = !candidates.isEmpty();
                getTrainStopWatch().lap();
            }
        }
        constituents.clear();
        constituents.addAll(selector.getSelected());
        modules = new EnsembleModule[constituents.size()];
        String savePath = getSavePath();
        if(savePath != null) {
            Utilities.mkdir(savePath);
        }
        for(int i = 0; i < constituents.size(); i++) {
            Knn knn = constituents.get(i).getKnn();
            modules[i] = new EnsembleModule(knn.toString(), knn, knn.getParameters());
            modules[i].trainResults = knn.getTrainResults();
            if(savePath != null) {
                modules[i].trainResults.writeFullResultsToFile(savePath + "/train" + i + ".csv");
            }
        }
        weightingScheme.defineWeightings(modules, trainInstances.numClasses());
        votingScheme.trainVotingScheme(modules, trainInstances.numClasses());
        ClassifierResults trainResults = new ClassifierResults();
        setTrainResults(trainResults);
        for(int i = 0; i < trainInstances.size(); i++) {
            long predictionTime = System.nanoTime();
            double[] distribution = votingScheme.distributionForTrainInstance(modules, i);
            predictionTime = System.nanoTime() - predictionTime;
            for (EnsembleModule module : modules) {
                predictionTime += module.trainResults.getPredictionTimeInNanos(i);
            }
            trainResults.addPrediction(trainInstances.get(i).classValue(), distribution, Utilities.argMax(distribution, getTrainRandom()), predictionTime, null);
        }
        getTrainStopWatch().lap();
        setClassifierResultsMetaInfo(trainResults);
        if(getTrainResultsPath() != null) {
            getTrainResults().writeFullResultsToFile(getTrainResultsPath());
        }
    }

    // todo use test stopwatch below
    public ClassifierResults getTestResults(Instances testInstances) throws
                                                                     Exception {
        String savePath = getSavePath();
        if(savePath != null) {
            Utilities.mkdir(savePath);
            int i = 0;
            for(TrainedCandidate constituent : constituents) {
                constituent.getKnn().getTestResults(testInstances).writeFullResultsToFile(savePath + "/test" + i + ".csv");
                i++;
            }
        }
        getTestStopWatch().reset();
        ClassifierResults results = new ClassifierResults();
        for(Instance testInstance : testInstances) {
            long time = System.nanoTime();
            double[] distribution = distributionForInstance(testInstance);
            time = System.nanoTime() - time;
            results.addPrediction(testInstance.classValue(), distribution, argMax(distribution), time, null);
        }
        getTestStopWatch().lap();
        setClassifierResultsMetaInfo(results);
        return results;
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                     Exception {
        return votingScheme.distributionForInstance(modules, testInstance);
    }

    private class Candidate {
        private final Knn knn;

        public Knn getKnn() {
            return knn;
        }

        public ParameterSpace getParameterSpace() {
            return parameterSpace;
        }

        private final ParameterSpace parameterSpace;

        private Candidate(final Knn knn, final ParameterSpace parameterSpace) { this.knn = knn;
            this.parameterSpace = parameterSpace;
        }

        private Candidate(Candidate candidate) {
            this(candidate.knn.copy(), candidate.parameterSpace);
        }

    }

    private class TrainedCandidate
        extends Candidate {
        private ClassifierResults trainResults;

        private TrainedCandidate(Candidate candidate, ClassifierResults trainResults) {
            super(new Candidate(candidate));
            this.trainResults = trainResults;
        }

        public ClassifierResults getTrainResults() {
            return trainResults;
        }

    }
}
