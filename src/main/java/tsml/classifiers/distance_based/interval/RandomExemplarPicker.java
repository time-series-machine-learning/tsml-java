package tsml.classifiers.distance_based.interval;

import utilities.InstanceTools;
import utilities.Randomised;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class RandomExemplarPicker implements ExemplarPicker, Randomised {

    private Random random = new Random(0);

    @Override public List<Instance> pickExemplars(final Instances data) {
        List<Instance> exemplars = new ArrayList<>();
        Map<Double, Instances> instancesByClass = Utilities.instancesByClass(data);
        for(int i = 0; i < data.numClasses(); i++) {
            Instances instances = instancesByClass.get(i);
            Instance exemplar = instances.get(random.nextInt(instances.size()));
            exemplars.add(exemplar);
        }
        return exemplars;
    }

    @Override public Random getRandom() {
        return random;
    }

    @Override public void setRandom(final Random random) {
        this.random = random;
    }
}
