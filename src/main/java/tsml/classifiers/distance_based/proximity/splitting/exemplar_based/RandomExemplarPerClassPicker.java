package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import org.junit.Assert;
import tsml.classifiers.distance_based.proximity.RandomSource;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class RandomExemplarPerClassPicker extends ExemplarPicker {

    private RandomSource random;

    public RandomExemplarPerClassPicker(RandomSource random) {
        setRandomSource(random);
    }

    @Override public List<Instance> pickExemplars(final Instances instances) {
        final Random random = getRandomSource().getRandom();
        final Map<Double, Instances> instancesByClass = Utilities.instancesByClass(instances);
        List<Instance> exemplars = new ArrayList<>();
        for(Double classLabel : instancesByClass.keySet()) {
            final Instances instanceClass = instancesByClass.get(classLabel);
            final Instance exemplar = Utilities.randPickOne(instanceClass, random);
            exemplars.add(exemplar);
        }
        return exemplars;
    }

    public RandomSource getRandomSource() {
        return random;
    }

    public RandomExemplarPerClassPicker setRandomSource(RandomSource random) {
        Assert.assertNotNull(random);
        this.random = random;
        return this;
    }
}
