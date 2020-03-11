package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.proximity.ProxTree;
import tsml.classifiers.distance_based.proximity.Split;
import tsml.classifiers.distance_based.utils.collections.PrunedMultimap;
import utilities.ArrayUtilities;
import weka.core.*;

import java.util.*;

/**
 * Purpose:
 */
public class RandomExemplarSimilaritySplitter extends Split implements Randomizable {
    private Random random = new Random();
    private Integer seed = null;
    private ProxTree proxTree;
    private ExemplarPicker exemplarPicker;
    private DistanceFunction distanceFunction;
    private boolean useEarlyAbandon = true;

    public boolean isUseEarlyAbandon() {
        return useEarlyAbandon;
    }

    public void setUseEarlyAbandon(final boolean useEarlyAbandon) {
        this.useEarlyAbandon = useEarlyAbandon;
    }

    public static abstract class ExemplarPicker implements Randomizable {

        private Integer seed;
        private Random random = new Random();

        @Override public void setSeed(final int seed) {
            this.seed = seed;
        }

        @Override public int getSeed() {
            return seed;
        }

        public Random getRandom() {
            return random;
        }

        public void setRandom(final Random random) {
            this.random = random;
        }

        public abstract List<Instance> pickExemplars(Instances instances);
    }

    public RandomExemplarSimilaritySplitter(ProxTree proxTree) {
        this.proxTree = proxTree;
    }

    @Override public void setSeed(final int seed) {
        this.seed = seed;
    }

    @Override public int getSeed() {
        return seed;
    }

    public Random getRandom() {
        return random;
    }

    public void setRandom(final Random random) {
        this.random = random;
    }

    @Override protected List<Instances> split(final Instances data) {
//        if(seed == null) {
//            throw new IllegalStateException("seed not set");
//        }
        final List<Instance> exemplars = exemplarPicker.pickExemplars(data);
        final Map<Instance, Integer> exemplarIndexMap = new HashMap<>();
        final List<Instances> split = new ArrayList<>();
        for(int i = 0; i < exemplars.size(); i++) {
            final Instance exemplar = exemplars.get(i);
            final Instances part = new Instances(data, 0);
            split.add(part);
            exemplarIndexMap.put(exemplar, i);
        }
        for(Instance instance : data) {
            double limit = DistanceMeasureable.getMaxDistance();
            PrunedMultimap<Double, Instance> map = PrunedMultimap.asc(ArrayList::new);
            map.setSoftLimit(1);
            for(Instance exemplar : exemplars) {
                distanceFunction.distance(exemplar, instance, limit);
                if(useEarlyAbandon) {
                    limit = map.lastKey();
                }
            }
            map.hardPruneToSoftLimit();
            final List<Instance> closestExemplars = ArrayUtilities.drain(map.values());
            if(closestExemplars.size() != 1) {
                throw new IllegalStateException("expected 1 only");
            }
            final Instance closestExemplar = closestExemplars.get(0);
            final int index = exemplarIndexMap.get(closestExemplar);
            split.get(index).add(instance);
        }
        return split;
    }
}
