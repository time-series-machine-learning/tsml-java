package classifiers.distance_based.knn;

import classifiers.template_classifier.TemplateClassifier;
import distances.DistanceMeasure;
import distances.time_domain.dtw.Dtw;
import evaluation.storage.ClassifierResults;
import utilities.ArrayUtilities;
import utilities.StringUtilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

import static java.lang.Double.max;
import static utilities.ArrayUtilities.argMax;

public class Knn
    extends TemplateClassifier {

    public static final int DEFAULT_K = 1;
    public static final boolean DEFAULT_EARLY_ABANDON = false;
    public static final String K_KEY = "k";
    public static final int DEFAULT_SAMPLE_SIZE = -1;
    public static final String SAMPLE_SIZE_KEY = "sampleSize";
    public static final String DISTANCE_MEASURE_KEY = "distanceMeasure";
    public static final String SAMPLING_STRATEGY_KEY = "samplingStrategy";
    private final List<NearestNeighbourSet> trainNearestNeighbourSets = new ArrayList<>();
    private final List<NearestNeighbourSet> testNearestNeighbourSets = new ArrayList<>();
    private final List<Instance> untestedTrainInstances = new ArrayList<>();
    private int k;
    private DistanceMeasure distanceMeasure;
    private boolean earlyAbandon;
    private int sampleSize;
    private SamplingStrategy samplingStrategy = SamplingStrategy.RANDOM;
    private final List<Instance> remainingTrainInstances = new ArrayList<>();
    private final List<Instance> trainInstances = new ArrayList<>();
    private Integer testInstancesHash = null;

    public Knn copy() {
        Knn knn = new Knn();
        try {
            knn.copyFromSerObject(this);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return knn;
    }

    @Override
    public void copyFromSerObject(final Object obj) throws
                                                    Exception {
        super.copyFromSerObject(obj);
        Knn other = (Knn) obj;
        trainNearestNeighbourSets.clear();
        for(NearestNeighbourSet nearestNeighbourSet : other.trainNearestNeighbourSets) {
            trainNearestNeighbourSets.add(nearestNeighbourSet.copy());
        }
        testNearestNeighbourSets.clear();
        for(NearestNeighbourSet nearestNeighbourSet : other.testNearestNeighbourSets) {
            testNearestNeighbourSets.add(nearestNeighbourSet.copy());
        }
        untestedTrainInstances.clear();
        untestedTrainInstances.addAll(other.untestedTrainInstances);
        setK(other.getK());
        setDistanceMeasure(other.getDistanceMeasure());
        setEarlyAbandon(other.getEarlyAbandon());
        setSampleSize(other.getSampleSize());
        setSamplingStrategy(other.getSamplingStrategy());
        remainingTrainInstances.clear();
        remainingTrainInstances.addAll(other.remainingTrainInstances);
        trainInstances.clear();
        trainInstances.addAll(other.trainInstances);
        remainingTrainInstances.clear();
        remainingTrainInstances.addAll(other.remainingTrainInstances);
        testInstancesHash = other.testInstancesHash;
        trainInstancesHash = other.trainInstancesHash;
    }

    public String[] getOptions() {
        return ArrayUtilities.concat(super.getOptions(), new String[] {
            K_KEY,
            String.valueOf(k),
            SAMPLE_SIZE_KEY,
            String.valueOf(sampleSize),
            SAMPLING_STRATEGY_KEY,
            samplingStrategy.name(),
            DISTANCE_MEASURE_KEY,
            distanceMeasure.toString(),
            StringUtilities.join(",", distanceMeasure.getOptions()),
            });
    }
    private Integer trainInstancesHash = null;

    public Knn() {
        setDistanceMeasure(new Dtw(0));
        setK(DEFAULT_K);
        setSampleSize(DEFAULT_SAMPLE_SIZE);
        setEarlyAbandon(DEFAULT_EARLY_ABANDON);
        setSamplingStrategy(SamplingStrategy.RANDOM);
    }

    public SamplingStrategy getSamplingStrategy() {
        return samplingStrategy;
    }    public void setOptions(final String[] options) throws
                                                   Exception {
        super.setOptions(options);
        for (int i = 0; i < options.length - 1; i += 2) {
            String key = options[i];
            String value = options[i + 1];
            if (key.equals(K_KEY)) {
                setK(Integer.parseInt(value));
            } else if (key.equals(SAMPLE_SIZE_KEY)) {
                setK(Integer.parseInt(value));
            } else if (key.equals(DISTANCE_MEASURE_KEY)) {
                setDistanceMeasure(DistanceMeasure.fromString(value));
            } else if(key.equals(SAMPLING_STRATEGY_KEY)) {
                setSamplingStrategy(SamplingStrategy.fromString(value));
            }
        }
        distanceMeasure.setOptions(options);
    }

    public void setSamplingStrategy(final SamplingStrategy samplingStrategy) {
        this.samplingStrategy = samplingStrategy;
    }

    public int getK() {
        return k;
    }

    public void setK(final int k) {
        this.k = k;
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

    public int getSampleSize() {
        return sampleSize;
    }

    public void setSampleSize(final int sampleSize) {
        this.sampleSize = sampleSize;
    }

    public boolean usesAllNeighbours() {
        return k <= 0;
    }

    private long phaseTime = 0;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        long startTime = System.nanoTime();
        int hash = data.hashCode();
        if(trainInstancesHash == null || hash != trainInstancesHash) {
            trainInstancesHash = hash;
            setTrainTimeNanos(0);
            trainInstances.clear();
            trainInstances.addAll(data);
            remainingTrainInstances.clear();
            remainingTrainInstances.addAll(data);
            trainNearestNeighbourSets.clear();
            for(Instance instance : data) {
                trainNearestNeighbourSets.add(new NearestNeighbourSet(instance));
            }
        }
        while(withinSampleSize() && samplesTrainInstances()
              && !remainingTrainInstances.isEmpty()
              && phaseTime < remainingTrainContract()) {
            long startPhaseTime = System.nanoTime();
            Instance instance = sampleInstance();
            untestedTrainInstances.add(instance);
            for(NearestNeighbourSet nearestNeighbourSet : trainNearestNeighbourSets) {
                nearestNeighbourSet.add(instance);
            }
            phaseTime = Long.max(System.nanoTime() - startPhaseTime, phaseTime);
        }
        ClassifierResults trainResults = new ClassifierResults();
        setTrainResults(trainResults);
        for(NearestNeighbourSet nearestNeighbourSet : trainNearestNeighbourSets) {
            double[] distribution = nearestNeighbourSet.predict();
            trainResults.addPrediction(nearestNeighbourSet.getTarget().classValue(), distribution, argMax(distribution), nearestNeighbourSet.getTime(), null);
        }
        incrementTrainTimeNanos(System.nanoTime() - startTime);
        setClassifierResultsMetaInfo(trainResults);
    }

    private boolean withinSampleSize() {
        return trainInstances.size() - remainingTrainInstances.size() < sampleSize;
    }

    public boolean samplesTrainInstances() {
        return sampleSize > 0;
    }

    private Instance sampleInstance() {
        List<Instance> instances = remainingTrainInstances;
        if(samplingStrategy.equals(SamplingStrategy.RANDOM)) {
            return instances.remove(getTrainRandom().nextInt(instances.size()));
        } else if(samplingStrategy.equals(SamplingStrategy.LINEAR)) {
            return instances.remove(0);
        } else {
            throw new UnsupportedOperationException();
        }
    }


    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                         Exception {
        NearestNeighbourSet testNearestNeighbourSet = new NearestNeighbourSet(testInstance);
        for (NearestNeighbourSet trainNearestNeighbourSet : trainNearestNeighbourSets) {
            testNearestNeighbourSet.add(trainNearestNeighbourSet.getTarget());
        }
        return testNearestNeighbourSet.predict();
    }

    public ClassifierResults getTestResults(Instances testInstances) throws
                                                              Exception {
        long startTime = System.nanoTime();
        int hash = testInstances.hashCode();
        if(testInstancesHash == null || hash != testInstancesHash) {
            testInstancesHash = hash;
            testNearestNeighbourSets.clear();
            for(Instance testInstance : testInstances) {
                testNearestNeighbourSets.add(new NearestNeighbourSet(testInstance));
            }
        }
        boolean change = !untestedTrainInstances.isEmpty();
        while (!untestedTrainInstances.isEmpty()) {
            Instance instance = untestedTrainInstances.remove(0);
            for(NearestNeighbourSet nearestNeighbourSet : testNearestNeighbourSets) {
                nearestNeighbourSet.add(instance);
            }
        }
        if(change) {
            for(NearestNeighbourSet nearestNeighbourSet : testNearestNeighbourSets) {
                nearestNeighbourSet.trimNeighbours();
            }
        }
        ClassifierResults results = new ClassifierResults();
        for(int i = 0; i < testNearestNeighbourSets.size(); i++) {
            NearestNeighbourSet nearestNeighbourSet = testNearestNeighbourSets.get(i);
            double[] distribution = nearestNeighbourSet.predict();
            long time = nearestNeighbourSet.getTime();
            results.addPrediction(nearestNeighbourSet.getTarget().classValue(), distribution, argMax(distribution), time, null);
        }
        incrementTestTimeNanos(System.nanoTime() - startTime);
        setClassifierResultsMetaInfo(results);
        return results;
    }
    public enum SamplingStrategy {
        RANDOM,
        LINEAR,
        STRATIFIED;

        public static SamplingStrategy fromString(String str) {
            for(SamplingStrategy s : SamplingStrategy.values()) {
                if(s.name().equals(str)) {
                    return s;
                }
            }
            throw new IllegalArgumentException("No enum value by the name of " + str);
        }
    }

    private class NearestNeighbourSet {
        private final Instance target;
        private final TreeMap<Double, List<Instance>> neighbours = new TreeMap<>();
        private TreeMap<Double, List<Instance>> trimmedNeighbours = null;
        private double maxDistance = Double.POSITIVE_INFINITY;
        private int size = 0;
        private long time = 0;
        private long predictTime = 0;

        public NearestNeighbourSet copy() {
            NearestNeighbourSet nearestNeighbourSet = new NearestNeighbourSet(target);
            for(Map.Entry<Double, List<Instance>> entry : neighbours.entrySet()) {
                nearestNeighbourSet.neighbours.put(entry.getKey(), new ArrayList<>(entry.getValue()));
            }
            nearestNeighbourSet.size = size;
            nearestNeighbourSet.time = time;
            nearestNeighbourSet.predictTime = predictTime;
            nearestNeighbourSet.trimNeighbours();
            return nearestNeighbourSet;
        }

        private NearestNeighbourSet(final Instance target) {this.target = target;}

        public Instance getTarget() {
            return target;
        }

        public int size() {
            return size;
        }

        public void trimNeighbours() {
            long startTime = System.nanoTime();
            int size = this.size;
            if(size == 0 || size == k) {
                trimmedNeighbours = neighbours;
                return;
            }
            trimmedNeighbours = new TreeMap<>();
            for (Map.Entry<Double, List<Instance>> entry : neighbours.entrySet()) {
                trimmedNeighbours.put(entry.getKey(), entry.getValue());
            }
            Map.Entry<Double, List<Instance>> last = trimmedNeighbours.lastEntry();
            List<Instance> furthestNeighbours = new ArrayList<>(last.getValue());
            trimmedNeighbours.put(last.getKey(), furthestNeighbours);
            while (size > k) {
                size--;
                int index = getTrainRandom().nextInt(furthestNeighbours.size());
                furthestNeighbours.remove(index);
            }
            time += System.nanoTime() - startTime;
        }

        public long getTime() {
            return time + predictTime;
        }

        public double[] predict() {
            long startTime = System.nanoTime();
            double[] distribution = new double[target.numClasses()];
            if(trimmedNeighbours == null) {
                trimNeighbours();
            }
            TreeMap<Double, List<Instance>> neighbours = trimmedNeighbours;
            if(neighbours.size() == 0) {
                distribution[getTestRandom().nextInt(distribution.length)]++;
                return distribution;
            }
            for (Map.Entry<Double, List<Instance>> entry : neighbours.entrySet()) {
                for (Instance instance : entry.getValue()) {
                    // todo weighted
                    distribution[(int) instance.classValue()]++;
                }
            }
            ArrayUtilities.normalise_inplace(distribution);
            predictTime = System.nanoTime() - startTime;
            return distribution;
        }

        public void addAll(final List<Instance> instances) {
            for(Instance instance : instances) {
                add(instance);
            }
        }

        public double add(Instance instance) {
            long startTime = System.nanoTime();
            double distance = distanceMeasure.distance(target, instance, maxDistance);
            time += System.nanoTime() - startTime;
            add(instance, distance);
            return distance;
        }

        public void add(Instance instance, double distance) {
            long startTime = System.nanoTime();
            List<Instance> equalDistanceNeighbours = neighbours.get(distance);
            if (equalDistanceNeighbours == null) {
                equalDistanceNeighbours = new ArrayList<>();
                neighbours.put(distance, equalDistanceNeighbours);
                if (earlyAbandon) { maxDistance = max(maxDistance, distance); }
            }
            equalDistanceNeighbours.add(instance);
            size++;
            int lastEntrySize = neighbours.lastEntry()
                                          .getValue()
                                          .size();
            if (size - k >= lastEntrySize) {
                neighbours.pollLastEntry();
                size -= lastEntrySize;
                if (earlyAbandon) {
                    maxDistance = max(maxDistance, neighbours.lastEntry()
                                                             .getKey());
                }
            }
            trimmedNeighbours = null;
            time += System.nanoTime() - startTime;
        }
    }






}
