package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.proximity.ReadOnlyRandomSource;
import tsml.classifiers.distance_based.proximity.splitting.Scorer;
import tsml.classifiers.distance_based.proximity.splitting.Split;
import tsml.classifiers.distance_based.utils.collections.PrunedMultimap;
import utilities.Utilities;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: split data into partitions using random exemplars.
 * <p>
 * Contributors: goastler
 */
public class RandomExemplarSimilaritySplit extends Split {
    private DistanceFunction distanceFunction;
    private List<List<Instance>> exemplars;
    private boolean useEarlyAbandon = true;
    private Random random;
    private Scorer scorer = Scorer.giniScore;

    @Override
    public Scorer getScorer() {
        return scorer;
    }

    @Override
    public RandomExemplarSimilaritySplit setScorer(final Scorer scorer) {
        this.scorer = scorer;
        return this;
    }

    public RandomExemplarSimilaritySplit(Instances data,
            DistanceFunction distanceFunction, List<List<Instance>> exemplars, Random random) {
        setData(data);
        setExemplars(exemplars);
        setDistanceFunction(distanceFunction);
        setRandom(random);
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public RandomExemplarSimilaritySplit setDistanceFunction(DistanceFunction distanceFunction) {
        Assert.assertNotNull(distanceFunction);
        this.distanceFunction = distanceFunction;
        return this;
    }

    public List<List<Instance>> getExemplars() {
        return exemplars;
    }

    public RandomExemplarSimilaritySplit setExemplars(final List<List<Instance>> exemplars) {
        Assert.assertNotNull(exemplars);
        for(List<Instance> exemplarGroup : exemplars) {
            Assert.assertNotNull(exemplarGroup);
            Assert.assertFalse(exemplarGroup.isEmpty());
        }
        this.exemplars = exemplars;
        return this;
    }

    @Override
    protected List<Instances> split() {
        final Instances data = getData();
        final List<Instances> partitions = new ArrayList<>();
        for(List<Instance> ignored : exemplars) {
            partitions.add(new Instances(data, 0));
        }
        getDistanceFunction().setInstances(data);
        for(Instance instance : data) {
            final int index = getPartitionIndexOf(instance);
            final Instances closestPartition = partitions.get(index);
            closestPartition.add(instance);
        }
        return partitions;
    }

    @Override
    public int getPartitionIndexOf(final Instance instance) {
        final List<List<Instance>> exemplars = getExemplars();
        final DistanceFunction distanceFunction = getDistanceFunction();
        double limit = DistanceMeasureable.getMaxDistance();
        final PrunedMultimap<Double, Integer> distanceToPartitionIndexMap = PrunedMultimap.asc();
        distanceToPartitionIndexMap.setSoftLimit(1);
        final boolean useEarlyAbandon = isUseEarlyAbandon();
        for(int i = 0; i < exemplars.size(); i++) {
            // todo extract min dist to exemplar in group into own interface
            double minDistance = DistanceMeasureable.getMaxDistance();
            for(Instance exemplar : exemplars.get(i)) {
                final double distance = distanceFunction.distance(instance, exemplar, limit);
                if(useEarlyAbandon) {
                    limit = Math.min(distance, limit);
                }
                minDistance = Math.min(distance, minDistance);
            }
            distanceToPartitionIndexMap.put(minDistance, i);
        }
        distanceToPartitionIndexMap.hardPruneToSoftLimit();
        final Double smallestDistance = distanceToPartitionIndexMap.firstKey();
        final Collection<Integer> closestPartitionIndices = distanceToPartitionIndexMap.get(smallestDistance);
        final Integer closestPartitionIndex = Utilities.randPickOne(closestPartitionIndices, getRandom());
        return closestPartitionIndex;
    }

    public boolean isUseEarlyAbandon() {
        return useEarlyAbandon;
    }

    public RandomExemplarSimilaritySplit setUseEarlyAbandon(final boolean useEarlyAbandon) {
        this.useEarlyAbandon = useEarlyAbandon;
        return this;
    }

    public Random getRandom() {
        return random;
    }

    public RandomExemplarSimilaritySplit setRandom(
        final Random random) {
        Assert.assertNotNull(random);
        this.random = random;
        return this;
    }
}
