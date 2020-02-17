package tsml.classifiers.distance_based.pf.relative;

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.pf.Split;
import utilities.Utilities;
import utilities.collections.PrunedMultimap;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

import java.util.*;

public class ExemplarSplit implements Split, Randomizable {

    private List<Instance> exemplars;
    private DistanceFunction distanceFunction;
    private List<Instances> parts;
    private Instances data;
    private double score = -1;
    private int seed = 0;
    private Random random = new Random(seed);
    private boolean earlyAbandon = true;

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

    public void setSeed(int seed) {
        this.seed = seed;
        random.setSeed(seed);
    }

    public int getSeed() {
        return seed;
    }

    @Override public List<Instances> split(final Instances data) {
        setData(data);
        return split();
    }

    @Override
    public void setData(Instances data) {
        this.data = data;
    }

    @Override
    public List<Instances> split() {
        if(data == null) {
            throw new IllegalStateException("data cannot be null");
        }
        if(exemplars == null) {
            throw new IllegalStateException("exemplars cannot be null");
        }
        parts = new ArrayList<>();
        Map<Instance, Instances> splitByExemplarMap = new HashMap<>();
        for(Instance exemplar : exemplars) {
            Instances part = new Instances(data, 0);
            splitByExemplarMap.put(exemplar, part);
            parts.add(part);
        }
        for(Instance instance : data) {
            Instance closestExemplar = findClosestExemplar(instance, random);
            Instances instances = splitByExemplarMap.get(closestExemplar);
            instances.add(instance);
        }
        return parts;
    }

    public List<Double> findDistanceToExemplars(Instance instance) {
        List<Double> distances = new ArrayList<>(exemplars.size());
        double min = DistanceMeasure.MAX_DISTANCE;
        for(Instance exemplar : exemplars) {
            double distance = distanceFunction.distance(instance, exemplar, min);
            if(earlyAbandon) {
                min = Math.min(min, distance);
            }
        }
        return distances;
    }

    public Instance findClosestExemplar(Instance instance, Random random) {
        PrunedMultimap<Double, Instance> map = PrunedMultimap.asc(ArrayList::new);
        map.setSoftLimit(1);
        map.setSeed(random.nextInt());
        List<Double> distances = findDistanceToExemplars(instance);
        if(distances.size() != exemplars.size()) {
            throw new IllegalStateException("shouldn't happen");
        }
        for(int i = 0; i < exemplars.size(); i++) {
            map.put(distances.get(i), exemplars.get(i));
        }
        return Utilities.randPickOne(map.values(), random);
    }

    public boolean isEarlyAbandon() {
        return earlyAbandon;
    }

    public void setEarlyAbandon(boolean earlyAbandon) {
        this.earlyAbandon = earlyAbandon;
    }
}
