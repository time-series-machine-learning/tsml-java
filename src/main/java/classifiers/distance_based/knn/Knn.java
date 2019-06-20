package classifiers.distance_based.knn;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.knn.sampling.*;
import classifiers.template_classifier.TemplateClassifier;
import distances.DistanceMeasure;
import distances.time_domain.dtw.Dtw;
import evaluation.storage.ClassifierResults;
import utilities.ArrayUtilities;
import utilities.StringUtilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

import static utilities.ArrayUtilities.argMax;
import static utilities.ArrayUtilities.indexOfMax;

public class Knn
    extends TemplateClassifier {

    public static final int DEFAULT_K = 1;
    public static final boolean DEFAULT_EARLY_ABANDON = true;
    public static final String K_KEY = "k";
    public static final int DEFAULT_NEIGHBOURHOOD_SIZE = -1;
    public static final String NEIGHBOURHOOD_SIZE_KEY = "neighbourhoodSizeLimit";
    public static final String NEIGHBOURHOOD_SIZE_PERCENTAGE_KEY = "neighbourhoodSizePercentage";
    public static final String DISTANCE_MEASURE_KEY = "distanceMeasure";
    public static final String NEIGHBOUR_SEARCH_STRATEGY_KEY = "neighbourSearchStrategy";
    public static final String TRAIN_SET_SIZE_KEY = "trainSubSetSizeLimit";
    public static final String TRAIN_SET_SIZE_PERCENTAGE_KEY = "trainSubSetSizePercentage";
    public static final String EARLY_ABANDON_KEY = "earlyAbandon";
    public static final int DEFAULT_TRAIN_SET_SIZE = -1;
    public static final int DEFAULT_TRAIN_SET_SIZE_PERCENTAGE = -1;
    public static final double DEFAULT_NEIGHBOURHOOD_SIZE_PERCENTAGE = -1;
    private final List<KNearestNeighbours> trainKNearestNeighbours = new ArrayList<>();
    private final List<Instance> trainSubSet = new ArrayList<>();
    private int k;
    private DistanceMeasure distanceMeasure;
    private boolean earlyAbandon;
    private int neighbourhoodSizeLimit;
    private double trainSubSetSizePercentage;
    private int trainSubSetSizeLimit;
    private NeighbourSearchStrategy neighbourSearchStrategy = NeighbourSearchStrategy.RANDOM;
    private DynamicIterator<Instance, ?> trainSubSetSampler;
    private DynamicIterator<Instance, ?> trainSetSampler;
    private final List<Instance> neighbourhood = new ArrayList<>();
    private double neighbourhoodSizePercentage = DEFAULT_NEIGHBOURHOOD_SIZE_PERCENTAGE;

    public TrainSetSampleStrategy getTrainSetSampleStrategy() {
        return trainSetSampleStrategy;
    }

    public void setTrainSetSampleStrategy(TrainSetSampleStrategy trainSetSampleStrategy) {
        this.trainSetSampleStrategy = trainSetSampleStrategy;
    }

    private TrainSetSampleStrategy trainSetSampleStrategy = TrainSetSampleStrategy.RANDOM;
    private Instances trainSet;

    public Knn() {
        setDistanceMeasure(new Dtw(0));
        setK(DEFAULT_K);
        setNeighbourhoodSizeLimit(DEFAULT_NEIGHBOURHOOD_SIZE);
        setEarlyAbandon(DEFAULT_EARLY_ABANDON);
        setTrainSubSetSizePercentage(DEFAULT_TRAIN_SET_SIZE_PERCENTAGE);
        setTrainSubSetSizeLimit(DEFAULT_TRAIN_SET_SIZE);
        setNeighbourSearchStrategy(NeighbourSearchStrategy.RANDOM);
    }

    @Override
    public String toString() {
        return "knn";
    }

    public Knn copy() {
        Knn knn = new Knn();
        try {
            knn.copyFromSerObject(this);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return knn;
    }

    public int getK() {
        return k;
    }

    public String[] getOptions() { // todo
        return ArrayUtilities.concat(super.getOptions(), new String[] {
            K_KEY,
            String.valueOf(k),
            NEIGHBOURHOOD_SIZE_KEY,
            String.valueOf(neighbourhoodSizeLimit),
            NEIGHBOUR_SEARCH_STRATEGY_KEY,
            neighbourSearchStrategy.name(),
            TRAIN_SET_SIZE_KEY,
            String.valueOf(trainSubSetSizeLimit),
            EARLY_ABANDON_KEY,
            String.valueOf(earlyAbandon),
            NEIGHBOURHOOD_SIZE_PERCENTAGE_KEY,
            String.valueOf(neighbourhoodSizePercentage),
            TRAIN_SET_SIZE_PERCENTAGE_KEY,
            String.valueOf(trainSubSetSizePercentage),
            DISTANCE_MEASURE_KEY,
            distanceMeasure.toString(),
            StringUtilities.join(",", distanceMeasure.getOptions()),
            });
    }

    public void setK(final int k) {
        this.k = k;
    }

    public void setOptions(final String[] options) throws
                                                   Exception { // todo
        super.setOptions(options);
        for (int i = 0; i < options.length - 1; i += 2) {
            String key = options[i];
            String value = options[i + 1];
            if (key.equals(K_KEY)) {
                setK(Integer.parseInt(value));
            } else if (key.equals(NEIGHBOURHOOD_SIZE_KEY)) {
                setK(Integer.parseInt(value));
            } else if (key.equals(DISTANCE_MEASURE_KEY)) {
                setDistanceMeasure(DistanceMeasure.fromString(value));
            } else if (key.equals(NEIGHBOUR_SEARCH_STRATEGY_KEY)) {
                setNeighbourSearchStrategy(NeighbourSearchStrategy.fromString(value));
            } else if (key.equals(EARLY_ABANDON_KEY)) {
                setEarlyAbandon(Boolean.parseBoolean(value));
            } else if (key.equals(TRAIN_SET_SIZE_KEY)) {
                setTrainSubSetSizeLimit(Integer.parseInt(value));
            } else if (key.equals(TRAIN_SET_SIZE_PERCENTAGE_KEY)) {
                setTrainSubSetSizePercentage(Double.parseDouble(value));
            } else if (key.equals(NEIGHBOURHOOD_SIZE_PERCENTAGE_KEY)) {
                setNeighbourhoodSizePercentage(Double.valueOf(value));
            }
        }
        distanceMeasure.setOptions(options);
    }

    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure;
    }

    public void setDistanceMeasure(final DistanceMeasure distanceMeasure) {
        this.distanceMeasure = distanceMeasure;
    }

    public boolean getEarlyAbandon() {
        return earlyAbandon;
    }

    public void setEarlyAbandon(final boolean earlyAbandon) {
        this.earlyAbandon = earlyAbandon;
    }

    public int getNeighbourhoodSizeLimit() {
        return neighbourhoodSizeLimit;
    }

    public void setNeighbourhoodSizeLimit(final int neighbourhoodSizeLimit) {
        this.neighbourhoodSizeLimit = neighbourhoodSizeLimit;
    }

    public NeighbourSearchStrategy getNeighbourSearchStrategy() {
        return neighbourSearchStrategy;
    }

    public void setNeighbourSearchStrategy(final NeighbourSearchStrategy neighbourSearchStrategy) {
        this.neighbourSearchStrategy = neighbourSearchStrategy;
    }

    public double getTrainSubSetSizePercentage() {
        return trainSubSetSizePercentage;
    }

    public int getTrainSubSetSizeLimit() {
        return trainSubSetSizeLimit;
    }

    public void setTrainSubSetSizeLimit(final int trainSubSetSizeLimit) {
        this.trainSubSetSizeLimit = trainSubSetSizeLimit;
    }

    public void setTrainSubSetSizePercentage(final double trainSubSetSizePercentage) {
        this.trainSubSetSizePercentage = trainSubSetSizePercentage;
    }

    public double getNeighbourhoodSizePercentage() {
        return neighbourhoodSizePercentage;
    }

    public void setNeighbourhoodSizePercentage(double neighbourhoodSizePercentage) {
        this.neighbourhoodSizePercentage = neighbourhoodSizePercentage;
    }

    private boolean hasRemainingTrainSubSet() {
        return (trainSubSet.size() < trainSubSetSizeLimit || (trainSubSetSizeLimit < 0 && trainSubSet.size() < trainSet.size()))
               && trainSetSampler.hasNext();
    }

    private boolean hasRemainingNeighbours() {
        return (neighbourhood.size() < neighbourhoodSizeLimit || neighbourhoodSizeLimit < 0)
               && trainSubSetSampler.hasNext();
    }

    private Instance nextTrainInstance() {
        Instance instance = trainSetSampler.next();
        trainSetSampler.remove();
        return instance;
    }

    private void nextTrainInstanceEstimate() {
        Instance trainInstance = nextTrainInstance();
        KNearestNeighbours KNearestNeighbours = new KNearestNeighbours(trainInstance);
        trainKNearestNeighbours.add(KNearestNeighbours);
        trainSubSet.add(trainInstance);
        KNearestNeighbours.addAll(neighbourhood);
        trainSubSetSampler.add(trainInstance);
    }

    private Instance nextNeighbour() {
        Instance neighbour = trainSubSetSampler.next();
        trainSubSetSampler.remove();
        return neighbour;
    }

    private void nextNeighbourSearch() {
        Instance neighbour = nextNeighbour();
        neighbourhood.add(neighbour);
        for (KNearestNeighbours trainKNearestNeighbours : this.trainKNearestNeighbours) {
            if (!trainKNearestNeighbours.getTarget()
                    .equals(neighbour)) {
                trainKNearestNeighbours.add(neighbour);
            }
        }
    }

    private void buildTrainResults() throws Exception {
        ClassifierResults trainResults = new ClassifierResults();
        setTrainResults(trainResults);
        for (KNearestNeighbours KNearestNeighbours : trainKNearestNeighbours) {
            KNearestNeighbours.trim();
            Prediction prediction = KNearestNeighbours.predict();
            double[] distribution = prediction.getDistribution();
            trainResults.addPrediction(KNearestNeighbours.getTarget().classValue(),
                    distribution,
                    ArrayUtilities.indexOfMax(distribution, getTrainRandom()),
                    prediction.getPredictionTimeNanos(),
                    null);
        }
        setClassifierResultsMetaInfo(trainResults);
    }

    private void checkTrainSet(Instances trainSet) {
        if (trainSetChanged(trainSet)) { // todo call in setters if exceeding certain
            // stuff, e.g. neighbourhood size inc
            getTrainStopWatch().reset();
            this.trainSet = trainSet;
            trainSubSet.clear();
            trainKNearestNeighbours.clear();
            neighbourhood.clear();
            setupNeighbourhoodSize();
            setupTrainSetSize();
            setupTrainSetSampleStrategy();
            setupNeighbourSearchStrategy();
        }
    }

    @Override
    public void buildClassifier(Instances trainSet) throws
                                                           Exception {
        checkTrainSet(trainSet);
        boolean remainingTrainSubSet = hasRemainingTrainSubSet();
        boolean remainingNeighbours = hasRemainingNeighbours();
        while ((remainingTrainSubSet || remainingNeighbours)) {
//            boolean choice = true;
//            if(remainingTrainSubSet && remainingNeighbours) {
//                choice = getTrainRandom().nextBoolean(); // todo change to strategy
//            } else if(remainingNeighbours) { // todo empty train set should be rand predict
//                choice = false;
//            }
//            if(choice) {
            if(remainingTrainSubSet) {
                nextTrainInstanceEstimate();
            } else {
                nextNeighbourSearch();
            }
            remainingTrainSubSet = hasRemainingTrainSubSet();
            remainingNeighbours = hasRemainingNeighbours();
            getTrainStopWatch().lap();
        }
        buildTrainResults();
        getTrainStopWatch().lap();
    }

    private void setupNeighbourSearchStrategy() {
        switch (neighbourSearchStrategy) {
            case RANDOM:
                trainSubSetSampler = new RandomSampler(getTrainRandom());
                break;
            case LINEAR:
                trainSubSetSampler = new LinearSampler();
                break;
            case ROUND_ROBIN_RANDOM:
                trainSubSetSampler = new RoundRobinRandomSampler(getTrainRandom());
                break;
            case DISTRIBUTED_RANDOM:
                trainSubSetSampler = new DistributedRandomSampler(getTrainRandom());
                break;
            default:
                throw new UnsupportedOperationException();
        }
    }

    private void setupNeighbourhoodSize() {
        if (neighbourhoodSizePercentage >= 0) {
            setNeighbourhoodSizeLimit((int) (trainSet.size() * neighbourhoodSizePercentage));
        }
    }

    private void setupTrainSetSize() {
        if (trainSubSetSizePercentage >= 0) {
            setTrainSubSetSizeLimit((int) (trainSet.size() * trainSubSetSizePercentage));
        }
    }

    private void setupTrainSetSampleStrategy() {
        switch (trainSetSampleStrategy) {
            case RANDOM:
                trainSetSampler = new RandomSampler(getTrainRandom());
                break;
            case LINEAR:
                trainSetSampler = new LinearSampler();
                break;
            case ROUND_ROBIN_RANDOM:
                trainSetSampler = new RoundRobinRandomSampler(getTrainRandom());
                break;
            case DISTRIBUTED_RANDOM:
                trainSetSampler = new DistributedRandomSampler(getTrainRandom());
                break;
            default:
                throw new UnsupportedOperationException();
        }
        for(Instance trainInstance : trainSet) {
            trainSetSampler.add(trainInstance);
        }
    }

    @Override
    public void copyFromSerObject(final Object obj) throws
                                                    Exception {
        super.copyFromSerObject(obj);
        Knn other = (Knn) obj;
        trainKNearestNeighbours.clear();
        for (KNearestNeighbours KNearestNeighbours : other.trainKNearestNeighbours) {
            trainKNearestNeighbours.add(new KNearestNeighbours(KNearestNeighbours));
        }
        setK(other.getK());
        setDistanceMeasure(DistanceMeasure.fromString(other.getDistanceMeasure().toString()));
        distanceMeasure.setOptions(other.getDistanceMeasure()
                                        .getOptions());
        setEarlyAbandon(other.getEarlyAbandon());
        setNeighbourhoodSizeLimit(other.getNeighbourhoodSizeLimit());
        setNeighbourSearchStrategy(other.getNeighbourSearchStrategy());
        setTrainSubSetSizePercentage(other.getTrainSubSetSizePercentage());
        setTrainSubSetSizeLimit(other.getTrainSubSetSizeLimit());
        setNeighbourhoodSizePercentage(other.getNeighbourhoodSizePercentage());
        setTrainSetSampleStrategy(other.getTrainSetSampleStrategy());
        trainSubSet.clear();
        trainSubSet.addAll(other.trainSubSet);
        neighbourhood.clear();
        neighbourhood.addAll(other.neighbourhood);
        trainSetSampler = other.trainSetSampler.iterator();
        trainSubSetSampler = other.trainSubSetSampler.iterator();
        trainSet = other.trainSet;
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                         Exception {
        KNearestNeighbours testKNearestNeighbours = new KNearestNeighbours(testInstance);
        testKNearestNeighbours.addAll(neighbourhood);
        testKNearestNeighbours.trim();
        return testKNearestNeighbours.predict().getDistribution();
    }

    public enum NeighbourSearchStrategy {
        RANDOM,
        LINEAR,
        ROUND_ROBIN_RANDOM,
        DISTRIBUTED_RANDOM;

        public static NeighbourSearchStrategy fromString(String str) {
            for (NeighbourSearchStrategy s : NeighbourSearchStrategy.values()) {
                if (s.name()
                     .equals(str)) {
                    return s;
                }
            }
            throw new IllegalArgumentException("No enum value by the name of " + str);
        }
    }

    public enum TrainSetSampleStrategy {
        RANDOM,
        LINEAR,
        ROUND_ROBIN_RANDOM,
        DISTRIBUTED_RANDOM;

        public static TrainSetSampleStrategy fromString(String str) {
            for (TrainSetSampleStrategy s : values()) {
                if (s.name()
                        .equals(str)) {
                    return s;
                }
            }
            throw new IllegalArgumentException("No enum value by the name of " + str);
        }
    }

    private class KNearestNeighbours {

        private final Instance target;
        private final TreeMap<Double, Collection<Instance>> kNeighbours = new TreeMap<>();
        private TreeMap<Double, Collection<Instance>> trimmedKNeighbours = kNeighbours;
        private int size = 0;
        private long searchTimeNanos = 0;
        private Collection<Instance> furthestNeighbours = null;
        private double furthestDistance = Double.POSITIVE_INFINITY;

        private KNearestNeighbours(final Instance target) {
            this.target = target;
        }

        public KNearestNeighbours(KNearestNeighbours other) {
            this(other.target);
            for (Map.Entry<Double, Collection<Instance>> entry : other.kNeighbours.entrySet()) {
                kNeighbours.put(entry.getKey(), new ArrayList<>(entry.getValue()));
            }
            size = other.size;
            searchTimeNanos = other.searchTimeNanos;
            furthestDistance = kNeighbours.lastKey();
            furthestNeighbours = kNeighbours.lastEntry().getValue();
            trim();
        }

        public void trim() {
            long startTime = System.nanoTime();
            if(size <= k) {
                trimmedKNeighbours = kNeighbours;
            } else {
                trimmedKNeighbours = new TreeMap<>();
                List<Instance> furthestNeighbours = null;
                for (Map.Entry<Double, Collection<Instance>> entry : kNeighbours.entrySet()) {
                    furthestNeighbours = new ArrayList<>(entry.getValue());
                    trimmedKNeighbours.put(entry.getKey(), furthestNeighbours);
                }
                if(furthestNeighbours != null) {
                    int size = furthestNeighbours.size();
                    while (size > k) {
                        size--;
                        int index = getTrainRandom().nextInt(furthestNeighbours.size());
                        furthestNeighbours.remove(index);
                    }
                }
            }
            searchTimeNanos = System.nanoTime() - startTime;
        }

        public Instance getTarget() {
            return target;
        }

        public int size() {
            return size;
        }

        public Prediction predict() {
            long startTime = System.nanoTime();
            double[] distribution = new double[target.numClasses()];
            TreeMap<Double, Collection<Instance>> neighbours = trimmedKNeighbours;
            if (neighbours.size() == 0) {
                distribution[getTestRandom().nextInt(distribution.length)]++;
            } else {
                for (Map.Entry<Double, Collection<Instance>> entry : neighbours.entrySet()) {
                    for (Instance instance : entry.getValue()) {
                        // todo weighted
                        distribution[(int) instance.classValue()]++;
                    }
                }
                ArrayUtilities.normaliseInplace(distribution);
            }
            long predictTimeNanos = System.nanoTime() - startTime;
            return new Prediction(distribution, predictTimeNanos);
        }

        public void addAll(final List<Instance> instances) {
            for (Instance instance : instances) {
                add(instance);
            }
        }

        public double add(Instance instance) {
            long startTime = System.nanoTime();
            double maxDistance = earlyAbandon ? furthestDistance : Double.POSITIVE_INFINITY;
            double distance = distanceMeasure.distance(target, instance, maxDistance);
            searchTimeNanos += System.nanoTime() - startTime;
            add(instance, distance);
            return distance;
        }

        public void add(Instance instance, double distance) {
            long startTime = System.nanoTime();
            if((distance <= furthestDistance || size < k) && k > 0) {
                Collection<Instance> equalDistanceNeighbours = kNeighbours.get(distance);
                if (equalDistanceNeighbours == null) {
                    equalDistanceNeighbours = new ArrayList<>();
                    kNeighbours.put(distance, equalDistanceNeighbours);
                    if(size == 0) {
                        furthestDistance = distance;
                        furthestNeighbours = equalDistanceNeighbours;
                    } else {
                        furthestDistance = Math.max(furthestDistance, distance);
                    }
                }
                equalDistanceNeighbours.add(instance);
                size++;
                if(distance < furthestDistance && size > k) { // if we've got too many neighbours AND just added a neighbour closer than the furthest then try and knock off the furthest lot
                    int numFurthestNeighbours = furthestNeighbours.size();
                    if (size - k >= numFurthestNeighbours) {
                        kNeighbours.pollLastEntry();
                        size -= numFurthestNeighbours;
                        Map.Entry<Double, Collection<Instance>> furthestNeighboursEntry = kNeighbours.lastEntry();
                        furthestNeighbours = furthestNeighboursEntry.getValue();
                        furthestDistance = furthestNeighboursEntry.getKey();
                    }
                }
                trimmedKNeighbours = null;
            }
            searchTimeNanos += System.nanoTime() - startTime;
        }
    }

    private static class Prediction {
        private final double[] distribution;

        public double[] getDistribution() {
            return distribution;
        }

        public long getPredictionTimeNanos() {
            return predictionTimeNanos;
        }

        private final long predictionTimeNanos;

        private Prediction(double[] distribution, long predictionTimeNanos) {
            this.distribution = distribution;
            this.predictionTimeNanos = predictionTimeNanos;
        }
    }




}
