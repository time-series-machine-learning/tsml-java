package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.proximity.ReadOnlyRandomSource;
import tsml.classifiers.distance_based.proximity.splitting.Split;
import tsml.classifiers.distance_based.utils.collections.PrunedMultimap;
import utilities.Utilities;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class RandomExemplarSimilaritySplit extends Split {
    private DistanceFunction distanceFunction;
    private List<Instance> exemplars;
    private boolean useEarlyAbandon = true;
    private ReadOnlyRandomSource randomSource;

    public RandomExemplarSimilaritySplit(double score, Instances data,
            DistanceFunction distanceFunction, List<Instance> exemplars, ReadOnlyRandomSource randomSource) {
        setScore(score);
        setData(data);
        setExemplars(exemplars);
        setDistanceFunction(distanceFunction);
        setRandomSource(randomSource);
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public RandomExemplarSimilaritySplit setDistanceFunction(DistanceFunction distanceFunction) {
        Assert.assertNotNull(distanceFunction);
        this.distanceFunction = distanceFunction;
        return this;
    }

    public List<Instance> getExemplars() {
        return exemplars;
    }

    public RandomExemplarSimilaritySplit setExemplars(final List<Instance> exemplars) {
        Assert.assertNotNull(exemplars);
        this.exemplars = exemplars;
        return this;
    }

    @Override
    protected List<Instances> findPartitions() {
        final Instances data = getData();
        final List<Instances> partitions = new ArrayList<>();
        for(Instance ignored : exemplars) {
            partitions.add(new Instances(data, 0));
        }
        for(Instance instance : data) {
            final int index = getPartitionIndexOf(instance);
            final Instances closestPartition = partitions.get(index);
            closestPartition.add(instance);
        }
        return partitions;
    }

    @Override
    public int getPartitionIndexOf(final Instance instance) {
        final List<Instance> exemplars = getExemplars();
        final DistanceFunction distanceFunction = getDistanceFunction();
        double limit = DistanceMeasureable.getMaxDistance();
        final PrunedMultimap<Double, Integer> distanceToPartitionIndexMap = PrunedMultimap.asc();
        distanceToPartitionIndexMap.setSoftLimit(1);
        for(int i = 0; i < exemplars.size(); i++) {
            final Instance exemplar = exemplars.get(i);
            final double distance = distanceFunction.distance(instance, exemplar, limit);
            if(isUseEarlyAbandon()) {
                limit = Math.min(distance, limit);
            }
            distanceToPartitionIndexMap.put(distance, i);
        }
        distanceToPartitionIndexMap.hardPruneToSoftLimit();
        final Double smallestDistance = distanceToPartitionIndexMap.firstKey();
        Collection<Integer> closestPartitionIndices = distanceToPartitionIndexMap.get(smallestDistance);
        final Integer closestPartitionIndex = Utilities.randPickOne(closestPartitionIndices, getRandomSource().getRandom());
        return closestPartitionIndex;
    }

    public boolean isUseEarlyAbandon() {
        return useEarlyAbandon;
    }

    public RandomExemplarSimilaritySplit setUseEarlyAbandon(final boolean useEarlyAbandon) {
        this.useEarlyAbandon = useEarlyAbandon;
        return this;
    }

    public ReadOnlyRandomSource getRandomSource() {
        return randomSource;
    }

    public RandomExemplarSimilaritySplit setRandomSource(
        final ReadOnlyRandomSource randomSource) {
        Assert.assertNotNull(randomSource);
        this.randomSource = randomSource;
        return this;
    }
}
