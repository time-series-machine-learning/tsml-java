package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import java.util.List;
import java.util.Map;
import org.junit.Assert;
import tsml.classifiers.distance_based.proximity.ReadOnlyRandomSource;
import tsml.classifiers.distance_based.proximity.splitting.Splitter;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class RandomExemplarSimilaritySplitter extends Splitter {

    private List<ParamSpace> paramSpaces;
    private ReadOnlyRandomSource randomSource;
    private ExemplarPicker exemplarPicker;

    public RandomExemplarSimilaritySplitter(List<ParamSpace> paramSpaces, ReadOnlyRandomSource randomSource,
        ExemplarPicker exemplarPicker) {
        setExemplarPicker(exemplarPicker);
        setParamSpaces(paramSpaces);
        setRandomSource(randomSource);
    }

    @Override
    public RandomExemplarSimilaritySplit buildSplit(Instances data) {
        // pick exemplars
        // pick distance measure via param space
        // todo
        Map<Instance, Integer> exemplars = null;
        double score = -2;
        List<Instances> partitions = null;
        List<Instance> distanceFunction = null;
        RandomExemplarSimilaritySplit split = new RandomExemplarSimilaritySplit(score, data, partitions,
            distanceFunction, exemplars);
        return split;
    }

    public List<ParamSpace> getParamSpaces() {
        return paramSpaces;
    }

    public RandomExemplarSimilaritySplitter setParamSpaces(
        List<ParamSpace> paramSpaces) {
        Assert.assertNotNull(paramSpaces);
        Assert.assertFalse(paramSpaces.isEmpty());
        this.paramSpaces = paramSpaces;
        return this;
    }

    public ReadOnlyRandomSource getRandomSource() {
        return randomSource;
    }

    public RandomExemplarSimilaritySplitter setRandomSource(
        ReadOnlyRandomSource randomSource) {
        Assert.assertNotNull(randomSource);
        this.randomSource = randomSource;
        return this;
    }

    public ExemplarPicker getExemplarPicker() {
        return exemplarPicker;
    }

    public RandomExemplarSimilaritySplitter setExemplarPicker(
        ExemplarPicker exemplarPicker) {
        Assert.assertNotNull(exemplarPicker);
        this.exemplarPicker = exemplarPicker;
        return this;
    }
}
