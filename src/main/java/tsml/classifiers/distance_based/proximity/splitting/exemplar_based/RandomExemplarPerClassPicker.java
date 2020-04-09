package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import com.beust.jcommander.internal.Lists;
import java.util.List;
import org.junit.Assert;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Map;
import java.util.Random;

public class RandomExemplarPerClassPicker implements ExemplarPicker {

    private Random random;

    public RandomExemplarPerClassPicker(Random random) {
        setRandom(random);
    }

    @Override public List<List<Instance>> pickExemplars(final Instances instances) {
        final Random random = getRandom();
        final Map<Double, Instances> instancesByClass = Utilities.instancesByClass(instances);
        List<List<Instance>> exemplars = Lists.newArrayList(instancesByClass.size());
        for(Double classLabel : instancesByClass.keySet()) {
            final Instances instanceClass = instancesByClass.get(classLabel);
            final Instance exemplar = Utilities.randPickOne(instanceClass, random);
            exemplars.add(Lists.newArrayList(exemplar));
        }
        return exemplars;
    }

    public Random getRandom() {
        return random;
    }

    public RandomExemplarPerClassPicker setRandom(Random random) {
        Assert.assertNotNull(random);
        this.random = random;
        return this;
    }
}
