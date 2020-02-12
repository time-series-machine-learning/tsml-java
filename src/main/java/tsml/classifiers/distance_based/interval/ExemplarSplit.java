package tsml.classifiers.distance_based.interval;

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import utilities.Utilities;
import utilities.collections.PrunedMultimap;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class ExemplarSplit implements Split {

    private List<Instance> exemplars;
    private DistanceFunction distanceFunction;
    private List<Instances> parts;
    private Instances data;
    private double score = -1;
    private Random random = new Random(0);

    @Override public List<Instances> getParts() {
        return parts;
    }

    @Override public Instances getData() {
        return data;
    }

    @Override public void cleanUp() {
        data = null;
        parts = null;
    }

    @Override public double getScore() {
        return score;
    }

    @Override public void setScore(final double score) {
        this.score = score;
    }

    public List<Instance> getExemplars() {
        return exemplars;
    }

    public void setExemplars(final List<Instance> exemplars) {
        this.exemplars = exemplars;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public void setDistanceFunction(final DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
    }

    public Random getRandom() {
        return random;
    }

    public void setRandom(final Random random) {
        this.random = random;
    }

    @Override public List<Instances> split(final Instances data) {
        parts = new ArrayList<>();
        this.data = data;
        Map<Instance, Instances> splitByExemplarMap = new HashMap<>();
        for(Instance exemplar : exemplars) {
            Instances part = new Instances(data, 0);
            splitByExemplarMap.put(exemplar, part);
            parts.add(part);
        }
        PrunedMultimap<Double, Instance> map = PrunedMultimap.asc(ArrayList::new);
        map.setSoftLimit(1);
        map.setRandom(random);
        double min = DistanceMeasure.MAX_DISTANCE;
        for(Instance instance : data) {
            for(Instance exemplar : exemplars) {
                double distance = distanceFunction.distance(instance, exemplar, min); // todo grab df
                map.put(distance, instance);
                min = map.lastKey();
            }
            Instance exemplar = Utilities.randPickOne(map.values(), random);
            Instances instances = splitByExemplarMap.get(exemplar);
            instances.add(instance);
        }
        return parts;
    }
}
