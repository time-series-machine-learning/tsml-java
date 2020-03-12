package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import com.google.common.collect.Lists;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.proximity.RandomSource;
import tsml.classifiers.distance_based.proximity.Split;
import tsml.classifiers.distance_based.utils.collections.PrunedMultimap;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import utilities.Utilities;
import weka.core.*;

import java.util.*;

/**
 * Purpose:
 */
public class RandomExemplarSimilaritySplit extends Split {

    /**
     * Purpose: build the splitter. This is responsible for picking the exemplars, picking the distance functions and
     * parameters.
     *
     */
    public static class Builder extends Split.Builder {
        private List<ParamSpace> paramSpaces = Lists.newArrayList();
        private ExemplarPicker exemplarPicker = new RandomExemplarPerClassPicker();
        private RandomSource randomSource;
        private boolean useEarlyAbandon = true;
        private Instances data;

        public Instances getData() {
            return data;
        }

        public RandomExemplarSimilaritySplit build() {
            final Instances data = getData();
            Assert.assertNotNull(data);
            final DistanceFunction distanceFunction = null; // todo
            final List<Instance> exemplars = null; // todo
            final RandomExemplarSimilaritySplit split = new RandomExemplarSimilaritySplit(exemplars,
                distanceFunction, getRandomSource());
            split.setUseEarlyAbandon(isUseEarlyAbandon());
            return split;
        }

        public List<ParamSpace> getParamSpaces() {
            return paramSpaces;
        }

        public Builder setParamSpaces(List<ParamSpace> paramSpaces) {
            this.paramSpaces = paramSpaces;
            return this;
        }

        public ExemplarPicker getExemplarPicker() {
            return exemplarPicker;
        }

        public Builder setExemplarPicker(
            ExemplarPicker exemplarPicker) {
            this.exemplarPicker = exemplarPicker;
            return this;
        }

        public RandomSource getRandomSource() {
            return randomSource;
        }

        public Builder setRandomSource(RandomSource randomSource) {
            this.randomSource = randomSource;
            return this;
        }

        public boolean isUseEarlyAbandon() {
            return useEarlyAbandon;
        }

        public Builder setUseEarlyAbandon(boolean earlyAbandon) {
            this.useEarlyAbandon = earlyAbandon;
            return this;
        }

        public Builder setData(Instances data) {
            this.data = data;
            return this;
        }
    }

    public RandomExemplarSimilaritySplit(List<Instance> exemplars, DistanceFunction distanceFunction,
        RandomSource randomSource) {
        setRandomSource(randomSource);
        setExemplars(exemplars);
        setDistanceFunction(distanceFunction);
    }

    private List<Instance> exemplars;
    private DistanceFunction distanceFunction;
    private boolean useEarlyAbandon = true;
    private RandomSource randomSource;

    public RandomSource getRandomSource() {
        return randomSource;
    }

    public RandomExemplarSimilaritySplit setRandomSource(RandomSource randomSource) {
        this.randomSource = randomSource;
        return this;
    }

    public List<Instance> getExemplars() {
        return exemplars;
    }

    public RandomExemplarSimilaritySplit setExemplars(List<Instance> exemplars) {
        Assert.assertNotNull(exemplars);
        Assert.assertFalse(exemplars.isEmpty());
        this.exemplars = exemplars;
        return this;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public RandomExemplarSimilaritySplit setDistanceFunction(DistanceFunction distanceFunction) {
        Assert.assertNotNull(distanceFunction);
        this.distanceFunction = distanceFunction;
        return this;
    }

    public boolean isUseEarlyAbandon() {
        return useEarlyAbandon;
    }

    public void setUseEarlyAbandon(final boolean useEarlyAbandon) {
        this.useEarlyAbandon = useEarlyAbandon;
    }

    /**
     * split the data using the exemplars and distance function. Note this function doesn't choose the exemplars or
     * distance function! That should be done before this function is called, ideally in a splitter builder.
     * @param data
     * @return
     */
    @Override protected List<Instances> splitData(final Instances data) {
        final List<Instance> exemplars = getExemplars();
        final Map<Instance, Integer> exemplarIndexMap = new HashMap<>(); // todo this need to be own field with
        // builder for using multiple exemplars per class
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
            final Instance closestExemplar = Utilities.randPickOne(map.values(), getRandomSource().getRandom());
            final int index = exemplarIndexMap.get(closestExemplar);
            split.get(index).add(instance);
        }
        return split;
    }
}
