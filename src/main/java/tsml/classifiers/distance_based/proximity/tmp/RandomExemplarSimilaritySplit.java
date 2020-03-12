package tsml.classifiers.distance_based.proximity.tmp;

import java.util.List;
import java.util.Map;
import org.junit.Assert;
import utilities.ArrayUtilities;
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
    private Map<Instance, Integer> exemplarMap;

    public RandomExemplarSimilaritySplit(double score, Instances data,
        List<Instances> partitions, DistanceFunction distanceFunction,
        Map<Instance, Integer> exemplarMap) {
        super(score, data, partitions);
        Assert.assertNotNull(distanceFunction);
        Assert.assertNotNull(exemplarMap);
        Assert.assertEquals(ArrayUtilities.unique(exemplarMap.values()).size(), getPartitions().size());
        setDistanceFunction(distanceFunction);
        setExemplarMap(exemplarMap);
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    protected RandomExemplarSimilaritySplit setDistanceFunction(DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
        return this;
    }

    public Map<Instance, Integer> getExemplarMap() {
        return exemplarMap;
    }

    protected RandomExemplarSimilaritySplit setExemplarMap(Map<Instance, Integer> exemplarMap) {
        this.exemplarMap = exemplarMap;
        return this;
    }
}
