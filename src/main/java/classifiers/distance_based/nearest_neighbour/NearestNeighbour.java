package classifiers.distance_based.nearest_neighbour;

import classifiers.template.ExtendedClassifier;
import distances.DistanceMeasure;
import distances.DistanceMeasureFactory;
import distances.dtw.Dtw;
import utilities.ArrayUtilities;
import utilities.StringUtilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

import static java.lang.Double.max;

public class NearestNeighbour
    extends ExtendedClassifier {

    public static final int DEFAULT_K = 1;
    public static final boolean DEFAULT_EARLY_ABANDON = false;
    public static final String K_KEY = "k";
    public static final int DEFAULT_SAMPLE_SIZE = -1;
    public static final String SAMPLE_SIZE_KEY = "sampleSize";
    public static final String DISTANCE_MEASURE_KEY = "distanceMeasure";
    public static final String DISTANCE_MEASURE_OPTIONS_KEY = "distanceMeasureOptionsStart";
    public static final String DISTANCE_MEASURE_OPTIONS_END_KEY = "distanceMeasureOptionsEnd";
    private final List<NearestNeighbourSet> trainNearestNeighbourSets = new ArrayList<>();
    private int k;
    private DistanceMeasure distanceMeasure;
    private boolean earlyAbandon;
    private int sampleSize;

    public String[] getOptions() {
        return ArrayUtilities.concat(super.getOptions(), new String[] {
            K_KEY,
            String.valueOf(k),
            SAMPLE_SIZE_KEY,
            String.valueOf(sampleSize),
            DISTANCE_MEASURE_KEY,
            distanceMeasure.toString(),
            DISTANCE_MEASURE_OPTIONS_KEY,
            "{",
            StringUtilities.join(",", distanceMeasure.getOptions()),
            });
    }

    public NearestNeighbour() {
        setDistanceMeasure(new Dtw(0));
        setK(DEFAULT_K);
        setSampleSize(DEFAULT_SAMPLE_SIZE);
        setEarlyAbandon(DEFAULT_EARLY_ABANDON);
    }

    public int getK() {
        return k;
    }

    public void setOptions(final String[] options) throws
                                                   Exception {
        super.setOptions(options);
        boolean parseDistanceMeasureOptions = false;
        List<String> distanceMeasureOptions = new ArrayList<>();
        for (int i = 0; i < options.length - 1; i += 2) {
            String key = options[i];
            String value = options[i + 1];
            if (parseDistanceMeasureOptions) {
                if (key.equals(DISTANCE_MEASURE_OPTIONS_END_KEY)) {
                    parseDistanceMeasureOptions = false;
                } else {
                    distanceMeasureOptions.add(key);
                    distanceMeasureOptions.add(value);
                }
            } else {
                if (key.equals(K_KEY)) {
                    setK(Integer.parseInt(value));
                } else if (key.equals(SAMPLE_SIZE_KEY)) {
                    setK(Integer.parseInt(value));
                } else if (key.equals(DISTANCE_MEASURE_KEY)) {
                    setDistanceMeasure(DistanceMeasureFactory.fromString(value));
                } else if (key.equals(DISTANCE_MEASURE_OPTIONS_KEY)) {
                    parseDistanceMeasureOptions = true;
                }
            }
        }
        if (distanceMeasureOptions.size() > 0) {
            distanceMeasure.setOptions(distanceMeasureOptions.toArray(new String[0]));
        }
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

    public boolean isEarlyAbandon() {
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

    @Override
    public void buildClassifier(final Instances trainInstances) throws
                                                                Exception {
        trainNearestNeighbourSets.clear();
        for (Instance instance : trainInstances) {
            if (!withinSampleSize() && samplesTrainInstances()) {
                return;
            }
            NearestNeighbourSet nextNearestNeighbourSet = new NearestNeighbourSet(instance);
            for (NearestNeighbourSet nearestNeighbourSet : trainNearestNeighbourSets) {
                double distance = nextNearestNeighbourSet.add(nearestNeighbourSet.getTarget());
                nearestNeighbourSet.add(nextNearestNeighbourSet.getTarget(), distance);
            }
            trainNearestNeighbourSets.add(nextNearestNeighbourSet);
        }
        for (NearestNeighbourSet nearestNeighbourSet : trainNearestNeighbourSets) {
            nearestNeighbourSet.trimNeighbours();
        }
    }

    private boolean withinSampleSize() {
        return trainNearestNeighbourSets.size() <= sampleSize;
    }

    public boolean samplesTrainInstances() {
        return sampleSize > 0;
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

    private class NearestNeighbourSet {
        private final Instance target;
        private final TreeMap<Double, List<Instance>> neighbours = new TreeMap<>();
        private TreeMap<Double, List<Instance>> trimmedNeighbours = null;
        private double maxDistance = Double.POSITIVE_INFINITY;
        private int size = 0;

        private NearestNeighbourSet(final Instance target) {this.target = target;}

        public Instance getTarget() {
            return target;
        }

        public int size() {
            return size;
        }

        public double add(Instance instance) {
            double distance = distanceMeasure.distance(target, instance, maxDistance);
            add(instance, distance);
            return distance;
        }

        public void add(Instance instance, double distance) {
            List<Instance> equalDistanceNeighbours = neighbours.get(distance);
            if (equalDistanceNeighbours == null) {
                equalDistanceNeighbours = new ArrayList<>();
                neighbours.put(distance, equalDistanceNeighbours);
                if (earlyAbandon) { maxDistance = max(maxDistance, distance); }
            }
            equalDistanceNeighbours.add(instance);
            size++;
            if (size - k >= neighbours.lastEntry()
                                      .getValue()
                                      .size()) {
                neighbours.pollLastEntry();
                if (earlyAbandon) {
                    maxDistance = max(maxDistance, neighbours.lastEntry()
                                                             .getKey());
                }
            }
            trimmedNeighbours = null;
        }

        public void trimNeighbours() {
            trimmedNeighbours = new TreeMap<>();
            int size = this.size;
            for (Map.Entry<Double, List<Instance>> entry : neighbours.entrySet()) {
                trimmedNeighbours.put(entry.getKey(), new ArrayList<>(entry.getValue()));
            }
            List<Instance> furthestNeighbours = trimmedNeighbours.lastEntry()
                                                                 .getValue();
            Random random = new Random(); // todo seed
            while (size > k) {
                size--;
                int index = random.nextInt(furthestNeighbours.size());
                furthestNeighbours.remove(index);
            }
        }

        public double[] predict() {
            double[] distribution = new double[target.numClasses()];
            TreeMap<Double, List<Instance>> neighbours = trimmedNeighbours;
            for (Map.Entry<Double, List<Instance>> entry : neighbours.entrySet()) {
                for (Instance instance : entry.getValue()) {
                    // todo weighted
                    distribution[(int) instance.classValue()]++;
                }
            }
            ArrayUtilities.normalise_inplace(distribution);
            return distribution;
        }
    }






}
