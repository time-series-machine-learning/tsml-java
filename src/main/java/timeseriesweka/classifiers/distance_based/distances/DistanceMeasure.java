package timeseriesweka.classifiers.distance_based.distances;

import timeseriesweka.classifiers.distance_based.distances.ddtw.Ddtw;
import timeseriesweka.classifiers.distance_based.distances.dtw.Dtw;
import timeseriesweka.classifiers.distance_based.distances.erp.Erp;
import timeseriesweka.classifiers.distance_based.distances.lcss.Lcss;
import timeseriesweka.classifiers.distance_based.distances.msm.Msm;
import timeseriesweka.classifiers.distance_based.distances.twed.Twed;
import timeseriesweka.classifiers.distance_based.distances.wddtw.Wddtw;
import timeseriesweka.classifiers.distance_based.distances.wdtw.Wdtw;
import timeseriesweka.filters.cache.Cache;
import timeseriesweka.filters.cache.DupeCache;
import utilities.Options;
import weka.core.Instance;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Enumeration;

public abstract class DistanceMeasure
    implements Serializable,
               Options {

    public DistanceMeasure() { }

    private double limit = Double.POSITIVE_INFINITY;
    private Instance firstInstance;
    private Instance secondInstance;

    private Cache<Integer, Integer, Double> distanceCache = new DupeCache<>();

    private boolean storeHashInInstanceWeight = true;
    public final static String CACHED_DISTANCE_KEY = "cachedDistances";

    private int count = 2;
    private boolean cacheDistances = true;

    @Override
    public String[] getOptions() {
        return new String[] {CACHED_DISTANCE_KEY,
                             String.valueOf(cacheDistances)
        };
    }

    @Override
    public void setOption(final String key, final String value) {
        if(key.equals(CACHED_DISTANCE_KEY)) setCacheDistances(Boolean.parseBoolean(value));
    }

    protected abstract double measureDistance();

    public boolean isStoreHashInInstanceWeight() {
        return storeHashInInstanceWeight;
    }

    public void setStoreHashInInstanceWeight(final boolean storeHashInInstanceWeight) {
        this.storeHashInInstanceWeight = storeHashInInstanceWeight;
    }

    protected int hash(Instance instance) {
        if(storeHashInInstanceWeight) {
            int hash = (int) instance.weight();
            if(hash == 1) {
                hash = count++;
                instance.setWeight(hash);
            }
            return hash;
        } else {
            return Arrays.hashCode(instance.toDoubleArray());
        }
    }

    public double cachedDistance() {
        int firstHash = hash(firstInstance);
        int secondHash = hash(secondInstance);
        Double distance = distanceCache.get(firstHash, secondHash);
        if(distance == null) {
            distance = measureDistance();
            distanceCache.put(firstHash, secondHash, distance);
        }
        return distance;
    }

    public double distance() {
        if(firstInstance.classIndex() != firstInstance.numAttributes() - 1) {
            throw new IllegalStateException("class value must be at the end");
        }
        if(secondInstance.classIndex() != secondInstance.numAttributes() - 1) {
            throw new IllegalStateException("class value must be at the end");
        }
        double distance;
        if(cacheDistances) {
            distance = cachedDistance();
        } else {
            distance = measureDistance();
        }
        return distance;
    }

    public double distance(final Instance first, final Instance second) {
        setFirstInstance(first);
        setSecondInstance(second);
        return distance();
    }

    public double distance(final Instance first, final Instance second, final double limit) {
        setLimit(limit);
        return distance(first, second);
    }

    public static DistanceMeasure fromString(String str) {
        switch(str) {
            case Dtw.NAME: return new Dtw();
            case Ddtw.NAME: return new Ddtw();
            case Wdtw.NAME: return new Wdtw();
            case Wddtw.NAME: return new Wddtw();
            case Twed.NAME: return new Twed();
            case Msm.NAME: return new Msm();
            case Lcss.NAME: return new Lcss();
            case Erp.NAME: return new Erp();
            default: throw new IllegalArgumentException("unknown distance measure: " + str);
        }
    }

    public static final String DISTANCE_MEASURE_KEY = "distanceMeasure";

    @Override
    public abstract String toString();

    public double getLimit() {
        return limit;
    }

    public void setLimit(final double limit) {
        this.limit = limit;
    }

    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    public Instance getSecondInstance() {
        return secondInstance;
    }

    public void setSecondInstance(final Instance second) {
        this.secondInstance = second;

    }

    public Instance getFirstInstance() {
        return firstInstance;
    }

    public void setFirstInstance(final Instance first) {
        this.firstInstance = first;
    }

    public boolean isCacheDistances() {
        return cacheDistances;
    }

    public void setCacheDistances(final boolean cacheDistances) {
        this.cacheDistances = cacheDistances;
    }

    public Cache<Integer, Integer, Double> getDistanceCache() {
        return distanceCache;
    }

    public void setDistanceCache(final Cache<Integer, Integer, Double> distanceCache) {
        this.distanceCache = distanceCache;
    }
}
